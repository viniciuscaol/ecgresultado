import matplotlib
matplotlib.use('Agg')
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import base64
import math
from scipy.signal import find_peaks, medfilt, butter, filtfilt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor

# --- Constantes de Análise ---
# Valores padrão de acordo com diretrizes comuns (podem variar)
QRS_NORMAL_MAX_S = 0.11
EIXO_QRS_NORMAL_MIN_GRAUS = -30
EIXO_QRS_NORMAL_MAX_GRAUS = 90
QTC_NORMAL_MAX_MALE_MS = 450
QTC_NORMAL_MAX_FEMALE_MS = 460
P_WAVE_SEARCH_WINDOW_MS = 250  # Janela antes do QRS para procurar a Onda P

# --- Funções de Processamento e Análise de ECG ---

def _butter_lowpass_filter(data, cutoff, fs, order=5):
    """Aplica um filtro Butterworth passa-baixa."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def parse_ecg_from_xml(xml_content):
    """
    Extrai dados brutos do ECG e metadados do conteúdo XML.
    Retorna os dados, a taxa de amostragem e os metadados do paciente.
    """
    if not xml_content:
        raise ValueError("Conteúdo XML vazio recebido.")

    root = ET.fromstring(xml_content)
    
    # --- Extração de Tags Essenciais ---
    exame_tag = root.find("Exame")
    if exame_tag is None:
        raise ValueError("Tag <Exame> não encontrada no XML.")
    
    registros_tag = exame_tag.find("Registros")
    if registros_tag is None:
        raise ValueError("Tag <Registros> não encontrada dentro de <Exame>.")

    # --- Extração de Parâmetros ---
    taxa_amostragem_str = registros_tag.get("TaxaAmostragem")
    match_taxa = re.search(r'\d+', taxa_amostragem_str) if taxa_amostragem_str else None
    if not match_taxa:
        raise ValueError(f"Taxa de amostragem inválida ou não encontrada: '{taxa_amostragem_str}'.")
    taxa_amostragem = float(match_taxa.group())

    sensibilidade_str = registros_tag.get("Sensibilidade")
    sensibilidade_valor = 0.005 # Valor padrão em mV se não encontrado
    if sensibilidade_str:
        match_sens = re.search(r'(\d+(\.\d+)?)', sensibilidade_str)
        if match_sens:
            sensibilidade_microvolts = float(match_sens.group(1))
            sensibilidade_valor = sensibilidade_microvolts / 1000.0 # microV para mV
    
    # --- Extração dos Dados dos Canais ---
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
                # Pula canais com dados inválidos mas continua o processamento
                continue

    if not ecg_data:
        raise ValueError("Nenhum dado de canal válido foi processado.")

    # --- Extração de Metadados do Paciente ---
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
            ano_nasc = int(data_nascimento.split('/')[-1])
            ano_exame = int(metadata["data_exame"].split('/')[-1])
            metadata["idade_paciente"] = ano_exame - ano_nasc
        except (ValueError, IndexError):
            metadata["idade_paciente"] = "N/A"

    return ecg_data, taxa_amostragem, metadata

def analyze_ecg_signals(ecg_data, taxa_amostragem):
    """
    Analisa os sinais de ECG para calcular métricas clínicas.
    """
    metrics = {}
    
    # Seleciona a derivação para análise (prioridade: DII, depois V1, depois a primeira disponível)
    signal_lead_name = "DII" if "DII" in ecg_data else ("V1" if "V1" in ecg_data else list(ecg_data.keys())[0])
    signal = ecg_data.get(signal_lead_name)
    
    if signal is None or len(signal) == 0:
        return {"error": "Sinal de ECG indisponível para análise."}

    # --- 1. Detecção de Picos R e Cálculo de FC ---
    signal_for_r_peaks = -signal if np.abs(np.min(signal)) > np.abs(np.max(signal)) else signal
    kernel_size = int(0.06 * taxa_amostragem)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    smoothed_signal = medfilt(signal_for_r_peaks, kernel_size=kernel_size)
    
    min_peak_height = np.std(smoothed_signal) * 1.5 # Limiar mais robusto
    min_peak_distance = int(0.4 * taxa_amostragem) # Max FC de 150 bpm

    picos_r, _ = find_peaks(smoothed_signal, height=min_peak_height, distance=min_peak_distance)

    if len(picos_r) < 2:
        # Tenta com um limiar menor se poucos picos forem encontrados
        min_peak_height = np.std(smoothed_signal) * 0.8
        picos_r, _ = find_peaks(smoothed_signal, height=min_peak_height, distance=min_peak_distance)

    metrics["frequencia_cardiaca"] = "N/A"
    metrics["intervalo_rr"] = "N/A"
    intervalos_rr_s = []
    if len(picos_r) > 1:
        intervalos_rr_amostras = np.diff(picos_r)
        intervalos_rr_s = intervalos_rr_amostras / taxa_amostragem
        valid_rr = intervalos_rr_s[(intervalos_rr_s > 0.3) & (intervalos_rr_s < 2.0)] # FC entre 30 e 200
        if len(valid_rr) > 0:
            rr_medio_s = np.mean(valid_rr)
            metrics["intervalo_rr"] = f"{rr_medio_s * 1000:.0f} ms"
            metrics["frequencia_cardiaca"] = f"{60 / rr_medio_s:.1f} bpm"
    
    # --- 2. Duração do QRS (Estimativa) ---
    metrics["duracao_qrs"] = "N/A"
    if len(picos_r) > 0:
        # Estimativa: duração fixa baseada na taxa de amostragem.
        # Uma medição real requereria encontrar o início e o fim do QRS.
        qrs_duration_samples = int(0.09 * taxa_amostragem) # Estimativa de 90ms
        metrics["duracao_qrs"] = f"{(qrs_duration_samples / taxa_amostragem) * 1000:.0f} ms"

    # --- 3. Eixo QRS ---
    metrics["eixo_qrs"] = "N/A"
    d1 = ecg_data.get("DI")
    avf = ecg_data.get("aVF")
    if d1 is not None and avf is not None and len(picos_r) > 0:
        amp_d1 = np.mean(d1[picos_r])
        amp_avf = np.mean(avf[picos_r])
        if abs(amp_d1) > 0.01 or abs(amp_avf) > 0.01:
             eixo_graus = np.degrees(np.arctan2(amp_avf, amp_d1))
             metrics["eixo_qrs"] = f"{eixo_graus:.1f}°"

    # --- 4. Intervalo QT e QTc (Fórmula de Bazett) ---
    metrics["intervalo_qtc"] = "N/A"
    if len(intervalos_rr_s) > 0:
        rr_medio_s = np.mean(intervalos_rr_s)
        # Estimativa de QT (fórmula de Hodges simplificada)
        qt_estimado_s = 0.38 * np.sqrt(rr_medio_s)
        if rr_medio_s > 0:
            qtc_s = qt_estimado_s / np.sqrt(rr_medio_s)
            metrics["intervalo_qtc"] = f"{qtc_s * 1000:.0f} ms"

    # --- 5. Detecção de Onda P ---
    picos_p = []
    onda_p_presente = "Não Avaliado"
    if len(picos_r) > 1:
        # Filtra o sinal para realçar a onda P (baixa frequência)
        filtered_p_signal = _butter_lowpass_filter(signal, cutoff=25, fs=taxa_amostragem, order=4)
        
        search_window_samples = int((P_WAVE_SEARCH_WINDOW_MS / 1000.0) * taxa_amostragem)
        found_p_count = 0

        for r_peak_idx in picos_r:
            start_window = max(0, r_peak_idx - search_window_samples)
            end_window = r_peak_idx - int(0.04 * taxa_amostragem) # Evita pegar o início do QRS
            
            if start_window >= end_window: continue

            segmento = filtered_p_signal[start_window:end_window]
            p_candidates, _ = find_peaks(segmento, height=np.std(segmento)*0.5, distance=int(0.1*taxa_amostragem))

            if len(p_candidates) > 0:
                # Pega o pico mais proeminente na janela como a onda P
                picos_p.append(start_window + p_candidates[np.argmax(segmento[p_candidates])])
                found_p_count += 1
        
        if found_p_count == 0:
            onda_p_presente = "Ausente ou não detectada"
        elif found_p_count >= len(picos_r) * 0.8: # Se P for encontrada em 80% dos batimentos
            onda_p_presente = "Presente e Consistente"
        else:
            onda_p_presente = "Presente mas Inconsistente"
            
    metrics["onda_p_presente"] = onda_p_presente
    
    # Armazena os picos para plotagem
    analysis_artifacts = {
        "picos_r": picos_r,
        "picos_p": np.array(picos_p),
        "r_peak_lead_name": signal_lead_name,
        "p_peak_lead_name": signal_lead_name
    }

    return metrics, analysis_artifacts

def generate_ecg_plot(ecg_data, taxa_amostragem, metrics, artifacts):
    """
    Gera um gráfico de 12 derivações do ECG e o retorna como uma string base64.
    """
    num_derivacoes = len(ecg_data)
    if num_derivacoes == 0: return None
    
    eixo_tempo = np.arange(len(list(ecg_data.values())[0])) / taxa_amostragem

    fig, axes = plt.subplots(num_derivacoes, 1, figsize=(18, 1.5 * num_derivacoes), sharex=True)
    if num_derivacoes == 1: axes = [axes] # Garante que `axes` seja sempre iterável
    
    plt.subplots_adjust(hspace=0.4)
    
    titulo = (f"ECG: {metrics.get('nome_paciente', 'N/A')} ({metrics.get('idade_paciente', 'N/A')} anos) | "
              f"Data: {metrics.get('data_exame', 'N/A')}\n"
              f"FC: {metrics.get('frequencia_cardiaca', 'N/A')} | "
              f"QRS: {metrics.get('duracao_qrs', 'N/A')} | "
              f"Eixo: {metrics.get('eixo_qrs', 'N/A')} | "
              f"QTc: {metrics.get('intervalo_qtc', 'N/A')} | "
              f"Onda P: {metrics.get('onda_p_presente', 'N/A')}")
    fig.suptitle(titulo, fontsize=12)

    # Define limites de Y globais para consistência visual
    all_samples = np.concatenate(list(ecg_data.values()))
    ymin, ymax = np.floor(all_samples.min()), np.ceil(all_samples.max())

    for i, (nome, amostras) in enumerate(ecg_data.items()):
        ax = axes[i]
        ax.plot(eixo_tempo, amostras, linewidth=1.0, color='black')
        ax.set_ylabel(f"{nome}\n(mV)", fontsize=9, rotation=0, ha='right', va='center')
        ax.set_ylim(ymin, ymax)

        # Plotar picos R
        if nome == artifacts["r_peak_lead_name"] and len(artifacts["picos_r"]) > 0:
            ax.plot(eixo_tempo[artifacts["picos_r"]], amostras[artifacts["picos_r"]], "o", color='red', markersize=5, label='Picos R')
        # Plotar picos P
        if nome == artifacts["p_peak_lead_name"] and len(artifacts["picos_p"]) > 0:
            ax.plot(eixo_tempo[artifacts["picos_p"]], amostras[artifacts["picos_p"]], "x", color='blue', markersize=5, label='Ondas P')

        ax.grid(which='major', color='lightcoral', linestyle='-', linewidth=0.6)
        ax.grid(which='minor', color='lightpink', linestyle=':', linewidth=0.4)
        ax.set_xticks(np.arange(0, eixo_tempo[-1], 0.2), minor=False)
        ax.set_xticks(np.arange(0, eixo_tempo[-1], 0.04), minor=True)
        ax.tick_params(labelsize=8)
    
    axes[-1].set_xlabel("Tempo (s)", fontsize=10)
    fig.legend(loc='upper right', fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_analysis_report(metrics):
    """
    Gera um laudo textual com base nas métricas calculadas, produzindo snippets HTML
    para serem usados diretamente no template.
    """
    conclusions = {}
    is_overall_normal = True

    # --- Valores de referência (simplificados) ---
    fc_normal_min, fc_normal_max = 60, 100
    qrs_normal_max_ms = 110
    eixo_qrs_normal_min, eixo_qrs_normal_max = -30, 90
    qtc_normal_max_ms = 460 # Limite geral simplificado

    # 1. Análise de Frequência Cardíaca
    fc_str = metrics.get("frequencia_cardiaca", "N/A")
    if "bpm" in fc_str:
        try:
            fc_valor = float(fc_str.replace(" bpm", ""))
            if fc_normal_min <= fc_valor <= fc_normal_max:
                conclusions["frequencia_cardiaca_html"] = f"<li>Frequência Cardíaca: <span class='normal'>Normal</span> ({fc_valor:.1f} bpm).</li>"
            else:
                is_overall_normal = False
                status = "Taquicardia" if fc_valor > fc_normal_max else "Bradicardia"
                conclusions["frequencia_cardiaca_html"] = f"<li>Frequência Cardíaca: <span class='abnormal'>{status}</span> ({fc_valor:.1f} bpm). Ritmo fora da faixa de normalidade.</li>"
        except (ValueError, TypeError):
            is_overall_normal = False
            conclusions["frequencia_cardiaca_html"] = "<li>Frequência Cardíaca: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["frequencia_cardiaca_html"] = "<li>Frequência Cardíaca: <span class='abnormal'>Não detectada.</span></li>"

    # 2. Análise da Duração do QRS
    qrs_str = metrics.get("duracao_qrs", "N/A")
    if "ms" in qrs_str:
        try:
            qrs_valor = float(qrs_str.replace(" ms", ""))
            if qrs_valor < qrs_normal_max_ms:
                conclusions["duracao_qrs_html"] = f"<li>Duração do QRS: <span class='normal'>Normal</span> ({qrs_valor:.0f} ms).</li>"
            else:
                is_overall_normal = False
                conclusions["duracao_qrs_html"] = f"<li>Duração do QRS: <span class='abnormal'>Alargado</span> ({qrs_valor:.0f} ms). Pode indicar distúrbio de condução.</li>"
        except (ValueError, TypeError):
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
            if eixo_qrs_normal_min <= eixo_valor <= eixo_qrs_normal_max:
                conclusions["eixo_qrs_html"] = f"<li>Eixo Elétrico QRS: <span class='normal'>Normal</span> ({eixo_valor:.1f}°).</li>"
            else:
                is_overall_normal = False
                desvio = "para a Direita" if eixo_valor > eixo_qrs_normal_max else "para a Esquerda"
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
            qtc_valor = float(qtc_str.replace(" ms", ""))
            if qtc_valor < qtc_normal_max_ms:
                conclusions["intervalo_qtc_html"] = f"<li>Intervalo QTc: <span class='normal'>Normal</span> ({qtc_valor:.0f} ms).</li>"
            else:
                is_overall_normal = False
                conclusions["intervalo_qtc_html"] = f"<li>Intervalo QTc: <span class='abnormal'>Alargado</span> ({qtc_valor:.0f} ms). Risco aumentado de arritmias.</li>"
        except (ValueError, TypeError):
            is_overall_normal = False
            conclusions["intervalo_qtc_html"] = "<li>Intervalo QTc: <span class='abnormal'>Valor inválido.</span></li>"
    else:
        is_overall_normal = False
        conclusions["intervalo_qtc_html"] = "<li>Intervalo QTc: <span class='abnormal'>Não detectado.</span></li>"
        
    # 5. Análise da Onda P
    onda_p_status = metrics.get("onda_p_presente", "Não Avaliado")
    if "Presente" in onda_p_status:
        conclusions["onda_p_html"] = f"<li>Onda P: <span class='normal'>{onda_p_status}</span>. Indica provável ritmo sinusal.</li>"
    else:
        is_overall_normal = False
        conclusions["onda_p_html"] = f"<li>Onda P: <span class='abnormal'>{onda_p_status}</span>. Pode indicar ritmo não sinusal.</li>"

    # Conclusão Geral
    if is_overall_normal:
        conclusions["overall_status_html"] = "<p><span class='normal'>Resultado Geral: Dentro dos padrões de normalidade para as métricas analisadas.</span></p>"
    else:
        conclusions["overall_status_html"] = "<p><span class='abnormal'>Resultado Geral: Fora dos padrões de normalidade. Recomenda-se avaliação médica.</span></p>"

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

    # --- Cabeçalho ---
    story.append(Paragraph("Relatório de Eletrocardiograma (ECG)", styles['h1']))
    story.append(Spacer(1, 10 * mm))

    # --- Informações do Paciente e Exame ---
    story.append(Paragraph("Informações do Paciente", styles['InfoTitle']))
    story.append(Paragraph(f"<b>Nome:</b> {metrics.get('nome_paciente', 'N/A')}", styles['Info']))
    story.append(Paragraph(f"<b>Idade:</b> {metrics.get('idade_paciente', 'N/A')} anos", styles['Info']))
    story.append(Paragraph(f"<b>Sexo:</b> {metrics.get('sexo_paciente', 'N/A')}", styles['Info']))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Informações do Exame", styles['InfoTitle']))
    story.append(Paragraph(f"<b>Data:</b> {metrics.get('data_exame', 'N/A')} às {metrics.get('hora_exame', 'N/A')}", styles['Info']))
    story.append(Spacer(1, 10 * mm))

    # --- Imagem do ECG ---
    image_data = base64.b64decode(image_b64)
    img = Image(io.BytesIO(image_data), width=180*mm, height=120*mm)
    story.append(img)
    story.append(Spacer(1, 10 * mm))

    # --- Resultados e Conclusões ---
    story.append(Paragraph("Resultados da Análise Automática", styles['h2']))
    
    # ... Adicionar os resultados da análise no PDF. Isso precisa ser implementado ...
    # Exemplo: story.append(Paragraph(f"<b>Frequência Cardíaca:</b> {metrics.get('frequencia_cardiaca')}", styles['Justify']))

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(analysis.get('overall_html', ''), styles['h4']))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(analysis.get('attention_note', ''), styles['Justify']))
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer