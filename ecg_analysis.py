import matplotlib
matplotlib.use('Agg')
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import io
import base64
import math
import neurokit2 as nk
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor

# --- Funções de Processamento e Análise de ECG ---

def parse_ecg_from_xml(xml_content):
    """
    Extrai dados brutos do ECG e metadados do conteúdo XML.
    """
    if not xml_content:
        raise ValueError("Conteúdo XML vazio recebido.")

    root = ET.fromstring(xml_content)
    
    exame_tag = root.find("Exame")
    if exame_tag is None:
        raise ValueError("Tag <Exame> não encontrada no XML.")
    
    registros_tag = exame_tag.find("Registros")
    if registros_tag is None:
        raise ValueError("Tag <Registros> não encontrada dentro de <Exame>.")

    taxa_amostragem_str = registros_tag.get("TaxaAmostragem")
    match_taxa = re.search(r'(\d+)', taxa_amostragem_str) if taxa_amostragem_str else None
    if not match_taxa:
        raise ValueError(f"Taxa de amostragem inválida ou não encontrada: '{taxa_amostragem_str}'.")
    taxa_amostragem = float(match_taxa.group(1))

    sensibilidade_str = registros_tag.get("Sensibilidade")
    sensibilidade_valor = 0.005
    if sensibilidade_str:
        match_sens = re.search(r'(\d+(\.\d+)?)', sensibilidade_str)
        if match_sens:
            sensibilidade_microvolts = float(match_sens.group(1))
            sensibilidade_valor = sensibilidade_microvolts / 1000.0
    
    ecg_data = {}
    canais_tag = registros_tag.find("Registro/Canais")
    if canais_tag is None:
        raise ValueError("Nenhum canal encontrado em <Registro>/<Canais>.")
        
    for canal in canais_tag.findall("Canal"):
        nome = canal.get("Nome")
        amostras_tag = canal.find("Amostras")
        if nome and amostras_tag is not None and amostras_tag.text:
            try:
                raw_samples = np.array([float(x) for x in amostras_tag.text.strip().replace('\n', '').split(';') if x.strip()])
                ecg_data[nome] = raw_samples * sensibilidade_valor
            except (ValueError, TypeError):
                continue

    if not ecg_data:
        raise ValueError("Nenhum dado de canal válido foi processado.")

    paciente_tag = root.find("Paciente")
    metadata = {
        "nome_paciente": paciente_tag.findtext("Nome", "Desconhecido") if paciente_tag is not None else "Desconhecido",
        "sexo_paciente": paciente_tag.findtext("Sexo", "N/A") if paciente_tag is not None else "N/A",
        "data_exame": exame_tag.findtext("Data", "N/A"),
        "hora_exame": exame_tag.findtext("Hora", "N/A"),
        "idade_paciente": "N/A"
    }
    
    data_nascimento = paciente_tag.findtext("DataNascimento") if paciente_tag is not None else None
    if data_nascimento and metadata["data_exame"] != "N/A":
        try:
            # Lógica para extrair ano, considerando formatos DD/MM/AAAA ou AAAA
            nasc_parts = data_nascimento.split('/')
            ano_nasc = int(nasc_parts[-1])
            
            exame_parts = metadata["data_exame"].split('/')
            ano_exame = int(exame_parts[-1])

            if ano_nasc > 1900 and ano_exame > 1900: # Validação simples
                metadata["idade_paciente"] = ano_exame - ano_nasc
        except (ValueError, IndexError):
            metadata["idade_paciente"] = "N/A"

    return ecg_data, taxa_amostragem, metadata


def analyze_ecg_signals(ecg_data, taxa_amostragem):
    """
    Versão Híbrida: Tenta a medição de alta precisão e recorre a estimativas
    robustas e padronizadas caso a primeira falhe.
    """
    lead_name = "DII" if "DII" in ecg_data else list(ecg_data.keys())[0]
    signal = ecg_data.get(lead_name)

    if signal is None:
        raise ValueError(f"Derivação '{lead_name}' não encontrada ou vazia.")

    # ETAPA 1: Limpeza do Sinal e Detecção de Picos R (essencial e robusto)
    cleaned_ecg = nk.signal_filter(signal, sampling_rate=taxa_amostragem, lowcut=0.5, highcut=35, method='butterworth', order=2)
    _, rpeaks_info = nk.ecg_peaks(cleaned_ecg, sampling_rate=taxa_amostragem, correct_artifacts=True)
    picos_r = rpeaks_info['ECG_R_Peaks']

    if len(picos_r) < 2:
        return {"frequencia_cardiaca": "N/A", "duracao_qrs": "N/A", "intervalo_qtc": "N/A"}, {}

    metrics = {}
    
    # ETAPA 2: TENTATIVA DE DELINEAMENTO DE ALTA PRECISÃO (PLANO A)
    try:
        _, waves_info = nk.ecg_delineate(cleaned_ecg, picos_r, sampling_rate=taxa_amostragem, method='peak')
        info = {**rpeaks_info, **waves_info}
        delineation_ok = all(k in info and info[k] is not None and len(info[k]) > 0 for k in ['ECG_QRS_Onsets', 'ECG_QRS_Offsets', 'ECG_T_Offsets'])
        if delineation_ok:
            print("INFO: Delineamento de alta precisão bem-sucedido.")
        else:
            print("AVISO: Delineamento incompleto. Usando estratégia de fallback.")
            delineation_ok = False # Garante que o fallback seja usado
    except Exception:
        info = rpeaks_info
        delineation_ok = False
        print("AVISO: Delineamento de alta precisão falhou. Usando estratégia de fallback.")

    # ETAPA 3: CÁLCULO DAS MÉTRICAS
    rate = nk.ecg_rate(picos_r, sampling_rate=taxa_amostragem, desired_length=len(cleaned_ecg))
    metrics["frequencia_cardiaca"] = f"{np.nanmean(rate):.1f} bpm"

    # Duração QRS
    if delineation_ok:
        durations = np.array(info['ECG_QRS_Offsets']) - np.array(info['ECG_QRS_Onsets'])
        qrs_duration_ms = np.nanmean(durations) * (1000 / taxa_amostragem)
        metrics["duracao_qrs"] = f"{qrs_duration_ms:.0f} ms"
    else:
        # PLANO B: Estimativa padrão de 90ms
        metrics["duracao_qrs"] = "90 ms (estimado)"

    # Intervalo QTc
    rr_intervals_s = np.diff(picos_r) / taxa_amostragem
    mean_rr_s = np.nanmean(rr_intervals_s)
    
    if delineation_ok:
        qt_intervals_s = (np.array(info['ECG_T_Offsets']) - np.array(info['ECG_QRS_Onsets'])) / taxa_amostragem
        mean_qt_s = np.nanmean(qt_intervals_s)
        if mean_rr_s > 0:
            qtc = mean_qt_s / np.sqrt(mean_rr_s) # Fórmula de Bazett
            metrics["intervalo_qtc"] = f"{qtc * 1000:.0f} ms"
        else:
            metrics["intervalo_qtc"] = "N/A"
    else:
        # PLANO B: Estimativa baseada em fórmula
        qt_estimado_s = 0.39 * np.sqrt(mean_rr_s)
        qtc_estimado = qt_estimado_s / np.sqrt(mean_rr_s)
        metrics["intervalo_qtc"] = f"{qtc_estimado * 1000:.0f} ms (estimado)"

    # Onda P
    metrics["onda_p_presente"] = "Não avaliado"
    if 'ECG_P_Peaks' in info and info.get('ECG_P_Peaks') is not None and len(clean_indices(info['ECG_P_Peaks'])) > 0:
         metrics["onda_p_presente"] = "Presente"
    else:
        metrics["onda_p_presente"] = "Ausente ou não detectada"
    
    # Eixo QRS
    metrics["eixo_qrs"] = "N/A"
    if "DI" in ecg_data and "aVF" in ecg_data:
        d1, avf = ecg_data.get("DI"), ecg_data.get("aVF")
        if d1 is not None and avf is not None and len(picos_r) > 0:
            amp_d1, amp_avf = np.nanmean(d1[picos_r]), np.nanmean(avf[picos_r])
            if not (np.isnan(amp_d1) or np.isnan(amp_avf)):
                eixo_graus = np.degrees(np.arctan2(amp_avf, amp_d1))
                metrics["eixo_qrs"] = f"{eixo_graus:.1f}°"

    analysis_artifacts = {
        "picos_r": picos_r, "picos_p": info.get("ECG_P_Peaks", []), "picos_q": info.get("ECG_Q_Peaks", []),
        "picos_s": info.get("ECG_S_Peaks", []), "picos_t": info.get("ECG_T_Peaks", []),
        "r_peak_lead_name": lead_name
    }
    
    return metrics, analysis_artifacts

def clean_indices(picos):
    if picos is None: return []
    return [int(p) for p in picos if not np.isnan(p)]

def _get_age_based_norms(age):
    """
    Retorna um dicionário com os valores de referência do ECG baseados na idade.
    """
    try:
        age = int(age)
    except (ValueError, TypeError):
        age = 30 # Default para adulto

    if age <= (1/12): # Neonato (até 1 mês)
        return {'hr_min': 90, 'hr_max': 180, 'qrs_max_ms': 80, 'qtc_max_ms': 450, 'eixo_min': 0, 'eixo_max': 180}
    elif age <= 1: # Lactente (1-12 meses)
        return {'hr_min': 90, 'hr_max': 160, 'qrs_max_ms': 80, 'qtc_max_ms': 450, 'eixo_min': 0, 'eixo_max': 135}
    elif age <= 3:
        return {'hr_min': 80, 'hr_max': 150, 'qrs_max_ms': 90, 'qtc_max_ms': 450, 'eixo_min': 0, 'eixo_max': 110}
    elif age <= 12:
        return {'hr_min': 60, 'hr_max': 120, 'qrs_max_ms': 100, 'qtc_max_ms': 450, 'eixo_min': 0, 'eixo_max': 110}
    else: # Adulto (> 12 anos)
        return {'hr_min': 60, 'hr_max': 100, 'qrs_max_ms': 110, 'qtc_max_ms': 460, 'eixo_min': -30, 'eixo_max': 90}


def generate_analysis_report(metrics):
    """
    Gera o laudo textual completo e estratificado por idade, com classes HTML.
    """
    conclusions = {}
    is_overall_normal = True
    
    norms = _get_age_based_norms(metrics.get("idade_paciente"))

    # 1. Análise de Frequência Cardíaca
    fc_str = metrics.get("frequencia_cardiaca", "N/A")
    if "bpm" in fc_str:
        try:
            fc_valor = float(re.search(r'[\d\.]+', fc_str).group())
            if norms['hr_min'] <= fc_valor <= norms['hr_max']:
                conclusions["frequencia_cardiaca_html"] = f"<li>Frequência Cardíaca: <span class='normal'>Normal para a idade</span> ({fc_valor:.1f} bpm).</li>"
            else:
                is_overall_normal = False
                status = "Taquicardia" if fc_valor > norms['hr_max'] else "Bradicardia"
                conclusions["frequencia_cardiaca_html"] = f"<li>Frequência Cardíaca: <span class='abnormal'>{status} para a idade</span> ({fc_valor:.1f} bpm).</li>"
        except (ValueError, TypeError, AttributeError):
            is_overall_normal = False
            conclusions["frequencia_cardiaca_html"] = "<li>Frequência Cardíaca: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["frequencia_cardiaca_html"] = "<li>Frequência Cardíaca: <span class='abnormal'>Não detectada.</span></li>"

    # 2. Análise da Duração do QRS
    qrs_str = metrics.get("duracao_qrs", "N/A")
    if "ms" in qrs_str:
        try:
            qrs_valor = float(re.search(r'[\d\.]+', qrs_str).group())
            if "estimado" in qrs_str:
                conclusions["duracao_qrs_html"] = f"<li>Duração do QRS: <span class='abnormal'>Não Medido</span> ({qrs_str}).</li>"
                is_overall_normal = False
            elif qrs_valor < norms['qrs_max_ms']:
                conclusions["duracao_qrs_html"] = f"<li>Duração do QRS: <span class='normal'>Normal para a idade</span> ({qrs_valor:.0f} ms).</li>"
            else:
                is_overall_normal = False
                conclusions["duracao_qrs_html"] = f"<li>Duração do QRS: <span class='abnormal'>Alargado para a idade</span> ({qrs_valor:.0f} ms).</li>"
        except (ValueError, TypeError, AttributeError):
             is_overall_normal = False
             conclusions["duracao_qrs_html"] = "<li>Duração do QRS: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["duracao_qrs_html"] = "<li>Duração do QRS: <span class='abnormal'>Não detectada.</span></li>"
    
    # 3. Análise do Eixo QRS
    eixo_str = metrics.get("eixo_qrs", "N/A")
    if "°" in eixo_str:
        try:
            eixo_valor = float(eixo_str.replace("°", ""))
            if norms['eixo_min'] <= eixo_valor <= norms['eixo_max']:
                conclusions["eixo_qrs_html"] = f"<li>Eixo Elétrico QRS: <span class='normal'>Normal para a idade</span> ({eixo_valor:.1f}°).</li>"
            else:
                is_overall_normal = False
                desvio = "para a Direita" if eixo_valor > norms['eixo_max'] else "para a Esquerda"
                conclusions["eixo_qrs_html"] = f"<li>Eixo Elétrico QRS: <span class='abnormal'>Desvio de eixo {desvio}</span> ({eixo_valor:.1f}°).</li>"
        except (ValueError, TypeError):
            is_overall_normal = False
            conclusions["eixo_qrs_html"] = "<li>Eixo Elétrico QRS: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["eixo_qrs_html"] = "<li>Eixo Elétrico QRS: <span class='abnormal'>Não detectado.</span></li>"
        
    # 4. Análise do Intervalo QTc
    qtc_str = metrics.get("intervalo_qtc", "N/A")
    if "ms" in qtc_str:
        try:
            qtc_valor = float(re.search(r'[\d\.]+', qtc_str).group())
            if "estimado" in qtc_str:
                conclusions["intervalo_qtc_html"] = f"<li>Intervalo QTc: <span class='abnormal'>Não Medido</span> ({qtc_str}).</li>"
                is_overall_normal = False
            elif qtc_valor < norms['qtc_max_ms']:
                conclusions["intervalo_qtc_html"] = f"<li>Intervalo QTc: <span class='normal'>Normal</span> ({qtc_valor:.0f} ms).</li>"
            else:
                is_overall_normal = False
                conclusions["intervalo_qtc_html"] = f"<li>Intervalo QTc: <span class='abnormal'>Alargado</span> ({qtc_valor:.0f} ms).</li>"
        except (ValueError, TypeError, AttributeError):
            is_overall_normal = False
            conclusions["intervalo_qtc_html"] = "<li>Intervalo QTc: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["intervalo_qtc_html"] = "<li>Intervalo QTc: <span class='abnormal'>Não detectado.</span></li>"
        
    # 5. Análise da Onda P
    onda_p_status = metrics.get("onda_p_presente", "Não Avaliado")
    if "Presente" in onda_p_status:
        conclusions["onda_p_html"] = f"<li>Onda P: <span class='normal'>{onda_p_status}</span>.</li>"
    else:
        is_overall_normal = False
        conclusions["onda_p_html"] = f"<li>Onda P: <span class='abnormal'>{onda_p_status}</span>.</li>"

    if is_overall_normal:
        conclusions["overall_status_html"] = "<p><strong><span class='normal'>Resultado Geral: Dentro dos padrões de normalidade para a idade do paciente.</span></strong></p>"
    else:
        conclusions["overall_status_html"] = "<p><strong><span class='abnormal'>Resultado Geral: Anormalidades detectadas ou parâmetros não puderam ser medidos. Recomenda-se avaliação médica.</span></strong></p>"

    conclusions["attention_note"] = "<b>Atenção:</b> Esta análise é gerada por computador e serve apenas como demonstração. Não substitui, de forma alguma, a avaliação e o diagnóstico de um profissional de saúde qualificado."
    
    return conclusions

# A função de plotagem não precisa de mais alterações.
def generate_ecg_plot(ecg_data, taxa_amostragem, metrics, artifacts):
    """
    Gera um gráfico de 12 derivações do ECG, plotando todas as ondas detectadas.
    """
    num_derivacoes = len(ecg_data)
    if num_derivacoes == 0: return None
    
    num_amostras = len(list(ecg_data.values())[0])
    eixo_tempo = np.arange(num_amostras) / taxa_amostragem

    fig, axes = plt.subplots(num_derivacoes, 1, figsize=(18, 1.5 * num_derivacoes), sharex=True)
    if num_derivacoes == 1: axes = [axes]
    
    plt.subplots_adjust(hspace=0.4)
    
    titulo = (f"ECG: {metrics.get('nome_paciente', 'N/A')} ({metrics.get('idade_paciente', 'N/A')} anos) | "
              f"Data: {metrics.get('data_exame', 'N/A')}\n"
              f"FC: {metrics.get('frequencia_cardiaca', 'N/A')} | "
              f"QRS: {metrics.get('duracao_qrs', 'N/A')} | "
              f"Eixo: {metrics.get('eixo_qrs', 'N/A')} | "
              f"QTc: {metrics.get('intervalo_qtc', 'N/A')} | "
              f"Onda P: {metrics.get('onda_p_presente', 'N/A')}")
    fig.suptitle(titulo, fontsize=12)

    all_samples = np.concatenate([amostras for amostras in ecg_data.values() if len(amostras) == num_amostras])
    if len(all_samples) == 0:
        return None
        
    ymin, ymax = np.floor(all_samples.min()), np.ceil(all_samples.max())

    picos_r_validos = clean_indices(artifacts.get("picos_r", []))
    picos_p_validos = clean_indices(artifacts.get("picos_p", []))
    picos_q_validos = clean_indices(artifacts.get("picos_q", []))
    picos_s_validos = clean_indices(artifacts.get("picos_s", []))
    picos_t_validos = clean_indices(artifacts.get("picos_t", []))

    lead_de_analise = artifacts.get("r_peak_lead_name", list(ecg_data.keys())[0])
    eixo_analisado = None

    for i, (nome, amostras) in enumerate(ecg_data.items()):
        if len(amostras) != num_amostras:
            continue
            
        ax = axes[i]
        if nome == lead_de_analise:
            eixo_analisado = ax

        ax.plot(eixo_tempo, amostras, linewidth=1.0, color='black')
        ax.set_ylabel(f"{nome}\n(mV)", fontsize=9, rotation=0, ha='right', va='center')
        ax.set_ylim(ymin, ymax)

        if nome == lead_de_analise:
            if picos_r_validos:
                ax.plot(eixo_tempo[picos_r_validos], amostras[picos_r_validos], "o", mfc='none', mec='red', markersize=7, label='Picos R')
            if picos_p_validos:
                ax.plot(eixo_tempo[picos_p_validos], amostras[picos_p_validos], "x", color='blue', markersize=5, label='Ondas P')
            # Você pode adicionar a plotagem dos outros picos (Q, S, T) aqui se desejar
        
        ax.grid(which='major', color='lightcoral', linestyle='-', linewidth=0.6)
        ax.grid(which='minor', color='lightpink', linestyle=':', linewidth=0.4)
        ax.set_xticks(np.arange(0, eixo_tempo[-1], 0.2), minor=False)
        ax.set_xticks(np.arange(0, eixo_tempo[-1], 0.04), minor=True)
        ax.tick_params(labelsize=8)
    
    axes[-1].set_xlabel("Tempo (s)", fontsize=10)
    
    if eixo_analisado:
        handles, labels = eixo_analisado.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _get_age_based_norms(age):
    """
    Retorna um dicionário com os valores de referência do ECG baseados na idade.
    Os valores são baseados em diretrizes comuns de cardiologia pediátrica.
    """
    # Tenta converter a idade para um inteiro. Se falhar, usa normas de adulto.
    try:
        age = int(age)
    except (ValueError, TypeError):
        age = 30 # Default para adulto se a idade for inválida

    if age <= 1/12: # Até 1 mês (Neonato)
        return {'hr_min': 90, 'hr_max': 180, 'qrs_max_ms': 80, 'qtc_max_ms': 450, 'eixo_min': 0, 'eixo_max': 180}
    elif age <= 1: # 1 mês a 1 ano (Lactente)
        return {'hr_min': 90, 'hr_max': 160, 'qrs_max_ms': 80, 'qtc_max_ms': 450, 'eixo_min': 0, 'eixo_max': 135}
    elif age <= 3: # 1 a 3 anos
        return {'hr_min': 80, 'hr_max': 150, 'qrs_max_ms': 90, 'qtc_max_ms': 450, 'eixo_min': 0, 'eixo_max': 110}
    elif age <= 5: # 3 a 5 anos
        return {'hr_min': 70, 'hr_max': 140, 'qrs_max_ms': 90, 'qtc_max_ms': 450, 'eixo_min': 0, 'eixo_max': 110}
    elif age <= 12: # 5 a 12 anos
        return {'hr_min': 60, 'hr_max': 120, 'qrs_max_ms': 100, 'qtc_max_ms': 450, 'eixo_min': 0, 'eixo_max': 110}
    elif age <= 16: # 12 a 16 anos
        return {'hr_min': 60, 'hr_max': 110, 'qrs_max_ms': 100, 'qtc_max_ms': 450, 'eixo_min': -15, 'eixo_max': 110}
    else: # Adulto (> 16 anos)
        return {'hr_min': 60, 'hr_max': 100, 'qrs_max_ms': 110, 'qtc_max_ms': 460, 'eixo_min': -30, 'eixo_max': 90}


# Agora, substitua a sua função generate_analysis_report por esta versão
def generate_analysis_report(metrics):
    """
    Gera um laudo textual completo e ESTRATIFICADO POR IDADE.
    """
    conclusions = {}
    is_overall_normal = True
    
    # PASSO 1: Obter as normas corretas para a idade do paciente
    norms = _get_age_based_norms(metrics.get("idade_paciente"))

    # 1. Análise de Frequência Cardíaca
    fc_str = metrics.get("frequencia_cardiaca", "N/A")
    if "bpm" in fc_str:
        try:
            fc_valor = float(re.search(r'[\d\.]+', fc_str).group())
            # PASSO 2: Usar as normas dinâmicas na comparação
            if norms['hr_min'] <= fc_valor <= norms['hr_max']:
                conclusions["frequencia_cardiaca_html"] = f"<li>Frequência Cardíaca: <span class='normal'>Normal para a idade</span> ({fc_valor:.1f} bpm).</li>"
            else:
                is_overall_normal = False
                status = "Taquicardia" if fc_valor > norms['hr_max'] else "Bradicardia"
                conclusions["frequencia_cardiaca_html"] = f"<li>Frequência Cardíaca: <span class='abnormal'>{status} para a idade</span> ({fc_valor:.1f} bpm).</li>"
        except (ValueError, TypeError, AttributeError):
            is_overall_normal = False
            conclusions["frequencia_cardiaca_html"] = "<li>Frequência Cardíaca: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["frequencia_cardiaca_html"] = "<li>Frequência Cardíaca: <span class='abnormal'>Não detectada.</span></li>"

    # 2. Análise da Duração do QRS
    qrs_str = metrics.get("duracao_qrs", "N/A")
    if "ms" in qrs_str:
        try:
            qrs_valor = float(re.search(r'[\d\.]+', qrs_str).group())
            if "estimado" in qrs_str:
                conclusions["duracao_qrs_html"] = f"<li>Duração do QRS: <span class='abnormal'>Não Medido</span> ({qrs_str}).</li>"
                is_overall_normal = False
            elif qrs_valor < norms['qrs_max_ms']:
                conclusions["duracao_qrs_html"] = f"<li>Duração do QRS: <span class='normal'>Normal para a idade</span> ({qrs_valor:.0f} ms).</li>"
            else:
                is_overall_normal = False
                conclusions["duracao_qrs_html"] = f"<li>Duração do QRS: <span class='abnormal'>Alargado para a idade</span> ({qrs_valor:.0f} ms).</li>"
        except (ValueError, TypeError, AttributeError):
             is_overall_normal = False
             conclusions["duracao_qrs_html"] = "<li>Duração do QRS: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["duracao_qrs_html"] = "<li>Duração do QRS: <span class='abnormal'>Não detectada.</span></li>"
    
    # 3. Análise do Eixo QRS
    eixo_str = metrics.get("eixo_qrs", "N/A")
    if "°" in eixo_str:
        try:
            eixo_valor = float(eixo_str.replace("°", ""))
            if norms['eixo_min'] <= eixo_valor <= norms['eixo_max']:
                conclusions["eixo_qrs_html"] = f"<li>Eixo Elétrico QRS: <span class='normal'>Normal para a idade</span> ({eixo_valor:.1f}°).</li>"
            else:
                is_overall_normal = False
                desvio = "para a Direita" if eixo_valor > norms['eixo_max'] else "para a Esquerda"
                conclusions["eixo_qrs_html"] = f"<li>Eixo Elétrico QRS: <span class='abnormal'>Desvio de eixo {desvio} para a idade</span> ({eixo_valor:.1f}°).</li>"
        except (ValueError, TypeError):
            is_overall_normal = False
            conclusions["eixo_qrs_html"] = "<li>Eixo Elétrico QRS: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["eixo_qrs_html"] = "<li>Eixo Elétrico QRS: <span class='abnormal'>Não detectado.</span></li>"
        
    # 4. Análise do Intervalo QTc
    qtc_str = metrics.get("intervalo_qtc", "N/A")
    if "ms" in qtc_str:
        try:
            qtc_valor = float(re.search(r'[\d\.]+', qtc_str).group())
            if "estimado" in qtc_str:
                conclusions["intervalo_qtc_html"] = f"<li>Intervalo QTc: <span class='abnormal'>Não Medido</span> ({qtc_str}).</li>"
                is_overall_normal = False
            elif qtc_valor < norms['qtc_max_ms']:
                conclusions["intervalo_qtc_html"] = f"<li>Intervalo QTc: <span class='normal'>Normal</span> ({qtc_valor:.0f} ms).</li>"
            else:
                is_overall_normal = False
                conclusions["intervalo_qtc_html"] = f"<li>Intervalo QTc: <span class='abnormal'>Alargado</span> ({qtc_valor:.0f} ms).</li>"
        except (ValueError, TypeError, AttributeError):
            is_overall_normal = False
            conclusions["intervalo_qtc_html"] = "<li>Intervalo QTc: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["intervalo_qtc_html"] = "<li>Intervalo QTc: <span class='abnormal'>Não detectado.</span></li>"
        
    # 5. Análise da Onda P
    onda_p_status = metrics.get("onda_p_presente", "Não Avaliado")
    if "Presente" in onda_p_status:
        conclusions["onda_p_html"] = f"<li>Onda P: <span class='normal'>{onda_p_status}</span>.</li>"
    else:
        is_overall_normal = False
        conclusions["onda_p_html"] = f"<li>Onda P: <span class='abnormal'>{onda_p_status}</span>.</li>"

    # Conclusão Geral
    if is_overall_normal:
        conclusions["overall_status_html"] = "<p><strong><span class='normal'>Resultado Geral: Dentro dos padrões de normalidade para a idade do paciente.</span></strong></p>"
    else:
        conclusions["overall_status_html"] = "<p><strong><span class='abnormal'>Resultado Geral: Anormalidades detectadas para a idade ou parâmetros não puderam ser medidos. Recomenda-se avaliação médica.</span></strong></p>"

    conclusions["attention_note"] = "<b>Atenção:</b> Esta análise é gerada por computador e serve apenas como demonstração. Não substitui, de forma alguma, a avaliação e o diagnóstico de um profissional de saúde qualificado."
    
    return conclusions

def gerar_pdf_ecg(metrics, analysis, image_b64):
    """Gera um relatório em PDF com os dados do ECG."""
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='Justify', alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='InfoTitle', fontSize=12, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Info', fontSize=10, fontName='Helvetica', leftIndent=10))

    story = []

    story.append(Paragraph("Relatório de Eletrocardiograma (ECG)", styles['h1']))
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph("Informações do Paciente", styles['InfoTitle']))
    story.append(Paragraph(f"<b>Nome:</b> {metrics.get('nome_paciente', 'N/A')}", styles['Info']))
    story.append(Paragraph(f"<b>Idade:</b> {metrics.get('idade_paciente', 'N/A')} anos", styles['Info']))
    story.append(Paragraph(f"<b>Sexo:</b> {metrics.get('sexo_paciente', 'N/A')}", styles['Info']))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Informações do Exame", styles['InfoTitle']))
    story.append(Paragraph(f"<b>Data:</b> {metrics.get('data_exame', 'N/A')} às {metrics.get('hora_exame', 'N/A')}", styles['Info']))
    story.append(Spacer(1, 10 * mm))

    image_data = base64.b64decode(image_b64)
    img = Image(io.BytesIO(image_data), width=180*mm, height=120*mm)
    story.append(img)
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph("Resultados da Análise Automática", styles['h2']))
    
    # Esta parte pode ser expandida para adicionar cada métrica individualmente no PDF.
    # Por simplicidade, estamos apenas adicionando a conclusão geral.
    
    story.append(Spacer(1, 10 * mm))
    
    # A função `generate_analysis_report` gera HTML, que não funciona bem no PDF diretamente.
    # Aqui, seria necessário criar parágrafos de texto puro.
    # Exemplo simples:
    story.append(Paragraph(f"<b>Frequência Cardíaca:</b> {metrics.get('frequencia_cardiaca', 'N/A')}", styles['Justify']))
    story.append(Paragraph(f"<b>Duração QRS:</b> {metrics.get('duracao_qrs', 'N/A')}", styles['Justify']))
    story.append(Paragraph(f"<b>Intervalo QTc:</b> {metrics.get('intervalo_qtc', 'N/A')}", styles['Justify']))
    story.append(Spacer(1, 5*mm))
    # story.append(Paragraph("<b>Atenção:</b> Esta é uma análise automática e demonstrativa. Não substitui o laudo de um cardiologista.", styles['Justify']))
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer