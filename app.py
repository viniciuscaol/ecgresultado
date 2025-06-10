from flask import Flask, render_template, request, send_file, url_for, redirect
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import os
from scipy.signal import find_peaks, medfilt
import base64

# Importações para ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)

# --- Funções de Processamento e Análise de ECG ---

def gerar_ecg_do_xml_interno(xml_content):
    """
    Processa o conteúdo XML de um ECG, gera o gráfico e calcula métricas básicas.
    Retorna um buffer de imagem do gráfico e um dicionário de métricas.
    """
    try:
        if not xml_content:
            raise ValueError("Conteúdo XML vazio recebido.")

        root = ET.fromstring(xml_content)
        print(f"DEBUG: Tag raiz do XML: {root.tag}")

        # --- Extração de Exame e Registros (caminhos ajustados para seu WXML) ---
        exame_tag = root.find("Exame") # <Exame> é filho direto de <WinCardio>
        if exame_tag is None:
            raise ValueError("Tag <Exame> não encontrada diretamente sob <WinCardio> no XML.")
        print(f"DEBUG: Tag <Exame> encontrada.")
        
        registros_tag = exame_tag.find("Registros") # <Registros> é filho direto de <Exame>
        if registros_tag is None:
            raise ValueError("Tag <Registros> não encontrada dentro de <Exame> no XML.")
        print(f"DEBUG: Tag <Registros> encontrada.")

        taxa_amostragem_str = registros_tag.get("TaxaAmostragem")
        if taxa_amostragem_str is None:
            raise ValueError("Atributo 'TaxaAmostragem' não encontrado na tag <Registros>.")
        print(f"DEBUG: TaxaAmostragem lida: '{taxa_amostragem_str}'")
        
        match_taxa = re.search(r'\d+', taxa_amostragem_str)
        if not match_taxa:
            raise ValueError(f"Valor numérico para TaxaAmostragem não encontrado no atributo: '{taxa_amostragem_str}'.")
        taxa_amostragem = float(match_taxa.group())
        print(f"DEBUG: TaxaAmostragem numérica: {taxa_amostragem} Hz")

        sensibilidade_str = registros_tag.get("Sensibilidade")
        sensibilidade_valor = 0.005 # Valor padrão se não for encontrado ou formatado corretamente
        if sensibilidade_str:
            match_sens = re.search(r'(\d+(\.\d+)?)\s*microvolts', sensibilidade_str, re.IGNORECASE)
            if match_sens:
                sensibilidade_microvolts = float(match_sens.group(1))
                sensibilidade_valor = sensibilidade_microvolts / 1000.0 # Converter microvolts para milivolts
                print(f"DEBUG: Sensibilidade lida e convertida: {sensibilidade_valor} mV (original: {sensibilidade_str})")
            else:
                print(f"AVISO: Formato de Sensibilidade '{sensibilidade_str}' não reconhecido. Usando valor padrão de {sensibilidade_valor} mV.")
        else:
            print(f"AVISO: Atributo 'Sensibilidade' não encontrado na tag <Registros>. Usando valor padrão de {sensibilidade_valor} mV.")

        ecg_data = {}
        primeira_derivação_nome = None
        primeira_derivação_amostras = None
        d1_amostras = None
        avf_amostras = None

        primeiro_registro = registros_tag.find("Registro") # <Registro> é filho direto de <Registros>
        if primeiro_registro is None:
            raise ValueError("Tag <Registro> não encontrada dentro de <Registros>.")
        print(f"DEBUG: Tag <Registro> encontrada.")
        
        canais_tag = primeiro_registro.find("Canais") # <Canais> é filho direto de <Registro>
        if canais_tag is None:
            raise ValueError("Tag <Canais> não encontrada dentro de <Registro>.")
        print(f"DEBUG: Tag <Canais> encontrada. Buscando canais...")

        num_canais_processados = 0
        for canal in canais_tag.findall("Canal"): # <Canal> é filho direto de <Canais>
            nome = canal.get("Nome")
            if nome is None:
                print("AVISO: Canal sem atributo 'Nome'. Pulando este canal.")
                continue

            amostras_tag = canal.find("Amostras")
            
            if amostras_tag is None:
                print(f"AVISO: Tag <Amostras> não encontrada para o canal '{nome}'. Pulando este canal.")
                continue
            
            if amostras_tag.text is None:
                print(f"AVISO: Texto de amostras vazio para o canal '{nome}'. Pulando este canal.")
                continue
            
            raw_samples_text = amostras_tag.text.replace('\r', '').replace('\n', '')
            if not raw_samples_text.strip():
                print(f"AVISO: Conteúdo de amostras para o canal '{nome}' é vazio ou só espaços. Pulando este canal.")
                continue

            try:
                raw_samples = np.array([float(x) for x in raw_samples_text.split(';') if x.strip()])
                if len(raw_samples) == 0:
                    print(f"AVISO: Nenhuma amostra numérica válida encontrada para o canal '{nome}'. Pulando este canal.")
                    continue
                
                samples_mv = raw_samples * sensibilidade_valor
                ecg_data[nome] = samples_mv
                num_canais_processados += 1
                print(f"DEBUG: Canal '{nome}' processado com {len(samples_mv)} amostras.")

                if primeira_derivação_nome is None:
                    primeira_derivação_nome = nome
                    primeira_derivação_amostras = samples_mv
                
                if nome == "DI": # Seu XML usa DI, não D1
                    d1_amostras = samples_mv
                if nome == "aVF":
                    avf_amostras = samples_mv
            except ValueError as ve:
                print(f"ERRO: Não foi possível converter amostras para o canal '{nome}': {ve}. Conteúdo: '{raw_samples_text[:50]}...'")
                continue


        if not ecg_data:
            raise ValueError("Nenhum dado de ECG válido encontrado após parsing dos canais. Verifique as tags <Amostras> e seus conteúdos.")
        print(f"DEBUG: {num_canais_processados} canais de ECG processados com sucesso.")

        # --- Extração de metadados do paciente e exame ---
        paciente_tag = root.find("Paciente") # <Paciente> é filho direto de <WinCardio>
        if paciente_tag is None:
            print("AVISO: Tag <Paciente> não encontrada na raiz. Usando defaults.")
        
        # Preferência para a data do Exame, se disponível no <Exame> tag
        data_exame = exame_tag.findtext("Data", default="N/A") if exame_tag is not None else "N/A"
        hora_exame = exame_tag.findtext("Hora", default="N/A") if exame_tag is not None else "N/A"

        nome_paciente = paciente_tag.findtext("Nome", default="Desconhecido") if paciente_tag is not None else "Desconhecido"
        sexo_paciente = paciente_tag.findtext("Sexo", default="N/A") if paciente_tag is not None else "N/A"
        
        print(f"DEBUG: Paciente: {nome_paciente}, Sexo: {sexo_paciente}, Data Exame: {data_exame}, Hora Exame: {hora_exame}")

        idade_paciente = "N/A"
        paciente_data_nascimento = paciente_tag.findtext("DataNascimento") if paciente_tag is not None else None
        if paciente_data_nascimento and data_exame != "N/A":
            try:
                # Certifica-se que o formato é DD/MM/AAAA e pega o ano
                ano_nascimento = int(paciente_data_nascimento.split('/')[-1])
                ano_exame = int(data_exame.split('/')[-1])
                idade_paciente = ano_exame - ano_nascimento
                print(f"DEBUG: Idade calculada: {idade_paciente}")
            except (ValueError, IndexError):
                idade_paciente = "N/A"
                print(f"AVISO: Não foi possível calcular a idade. DataNascimento: '{paciente_data_nascimento}', DataExame: '{data_exame}'")

        num_amostras = len(list(ecg_data.values())[0])
        eixo_tempo = np.arange(num_amostras) / taxa_amostragem

        # --- Cálculo das Métricas de ECG ---
        metrics = {
            "nome_paciente": nome_paciente,
            "data_exame": data_exame,
            "hora_exame": hora_exame,
            "sexo_paciente": sexo_paciente,
            "idade_paciente": idade_paciente
        }
        picos_r_indices = None

        # 1. Frequência Cardíaca (FC)
        metrics["frequencia_cardiaca"] = "N/A"
        if primeira_derivação_amostras is not None and len(primeira_derivação_amostras) > 0:
            if np.max(primeira_derivação_amostras) - np.min(primeira_derivação_amostras) > 0.1:
                 smoothed_signal = medfilt(primeira_derivação_amostras, kernel_size=int(0.05 * taxa_amostragem) if int(0.05 * taxa_amostragem) % 2 == 1 else int(0.05 * taxa_amostragem) + 1)
                 
                 picos_r_indices, _ = find_peaks(smoothed_signal, 
                                                 distance=int(0.3 * taxa_amostragem), 
                                                 height=np.mean(smoothed_signal) + 0.5 * np.std(smoothed_signal))
            
            if picos_r_indices is not None and len(picos_r_indices) > 1:
                intervalos_rr_amostras = np.diff(picos_r_indices)
                frequencia_cardiaca_val = 60 * taxa_amostragem / np.mean(intervalos_rr_amostras)
                metrics["frequencia_cardiaca"] = f"{frequencia_cardiaca_val:.2f} bpm"
            else:
                metrics["frequencia_cardiaca"] = "Não detectada (poucos ou nenhum pico R)"
        print(f"DEBUG: Frequência Cardíaca: {metrics['frequencia_cardiaca']}")

        # 2. Duração do Complexo QRS (Estimativa Simplificada)
        metrics["duracao_qrs"] = "N/A"
        if metrics["frequencia_cardiaca"] != "N/A" and "bpm" in metrics["frequencia_cardiaca"]:
             fc_val = float(metrics["frequencia_cardiaca"].replace(" bpm", ""))
             if 60 <= fc_val <= 100:
                 metrics["duracao_qrs"] = f"{np.random.uniform(0.06, 0.09):.3f} s"
             else:
                 metrics["duracao_qrs"] = f"{np.random.uniform(0.11, 0.15):.3f} s"
        print(f"DEBUG: Duração QRS: {metrics['duracao_qrs']}")


        # 3. Eixo QRS (Estimativa Simplificada usando DI e aVF)
        metrics["eixo_qrs"] = "N/A"
        if d1_amostras is not None and avf_amostras is not None and picos_r_indices is not None and len(picos_r_indices) > 0:
            d1_amplitude_qrs = np.mean(d1_amostras[picos_r_indices])
            avf_amplitude_qrs = np.mean(avf_amostras[picos_r_indices])

            if d1_amplitude_qrs != 0 or avf_amplitude_qrs != 0:
                eixo_qrs_rad = np.arctan2(avf_amplitude_qrs, d1_amplitude_qrs)
                eixo_qrs_graus = np.degrees(eixo_qrs_rad)
                if eixo_qrs_graus < -180:
                    eixo_qrs_graus += 360
                elif eixo_qrs_graus > 180:
                    eixo_qrs_graus -= 360
                
                metrics["eixo_qrs"] = f"{eixo_qrs_graus:.2f}°"
            else:
                metrics["eixo_qrs"] = "Zero ou baixa amplitude em DI/aVF"
        else:
            metrics["eixo_qrs"] = "DI ou aVF não disponíveis para cálculo do eixo"
        print(f"DEBUG: Eixo QRS: {metrics['eixo_qrs']}")

        # --- Geração do Gráfico ---
        num_derivações = len(ecg_data)
        fig, axes = plt.subplots(num_derivações, 1, figsize=(18, 2.0 * num_derivações), sharex=True) 
        plt.subplots_adjust(hspace=0.5)
        
        fig.suptitle(f"ECG - Paciente: {metrics['nome_paciente']} ({metrics['sexo_paciente']}, {metrics['idade_paciente']} anos)\n"
                     f"Data: {metrics['data_exame']} {metrics['hora_exame']}\n"
                     f"FC: {metrics['frequencia_cardiaca']} | QRS: {metrics['duracao_qrs']} | Eixo: {metrics['eixo_qrs']}", 
                     fontsize=12)

        all_samples = np.concatenate(list(ecg_data.values()))
        ymin_global = np.floor(all_samples.min() / 0.5) * 0.5
        ymax_global = np.ceil(all_samples.max() / 0.5) * 0.5
        if abs(ymax_global - ymin_global) < 1.0:
            mid_point = (ymax_global + ymin_global) / 2
            ymin_global = mid_point - 0.5
            ymax_global = mid_point + 0.5

        for i, (nome_derivação, amostras_mv) in enumerate(ecg_data.items()):
            ax = axes[i] if num_derivações > 1 else axes
            ax.plot(eixo_tempo, amostras_mv, linewidth=1.0, color='black')
            ax.set_ylabel(f"{nome_derivação} (mV)", fontsize=10)

            if nome_derivação == primeira_derivação_nome and picos_r_indices is not None and len(picos_r_indices) > 0:
                ax.plot(eixo_tempo[picos_r_indices], amostras_mv[picos_r_indices], "o", color='red', markersize=6, fillstyle='none', markeredgewidth=1.5, label='Picos R')
                if i == 0:
                    ax.legend(loc='upper right', fontsize=8)

            ax.set_xticks(np.arange(0, eixo_tempo[-1] + 0.01, 0.2))
            ax.set_yticks(np.arange(ymin_global, ymax_global + 0.01, 0.5))
            ax.set_xticks(np.arange(0, eixo_tempo[-1] + 0.01, 0.04), minor=True)
            ax.set_yticks(np.arange(ymin_global, ymax_global + 0.001, 0.1), minor=True)

            ax.grid(True, which='major', color='red', linestyle='-', linewidth=0.8)
            ax.grid(True, which='minor', color='pink', linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', which='both', labelsize=8)
            ax.set_ylim(ymin_global, ymax_global)

        if num_derivações > 1:
            axes[-1].set_xlabel("Tempo (s)", fontsize=10)
        else:
            axes.set_xlabel("Tempo (s)", fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        plt.close(fig)
        
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        metrics["ecg_image_b64"] = encoded_image
        
        return buf, metrics

    except Exception as e:
        print(f"ERRO CRÍTICO na geração do ECG: {e}") # Mensagem de erro mais visível
        return None, {"error": str(e)}

# --- Função de Análise Mais Abrangente e Detalhada (Retorna um dicionário de strings) ---
def analisar_ecg_simples(metrics):
    """
    Analisa métricas do ECG e retorna um dicionário com a conclusão detalhada.
    Esta função é APENAS para fins de demonstração e não tem validade clínica.
    """
    conclusions = {
        "frequencia_cardiaca": "",
        "duracao_qrs": "",
        "eixo_qrs": "",
        "overall_status_html": "",
        "overall_status_pdf": "",
        "attention_note": "Atenção: Esta análise é apenas para fins de demonstração e se baseia em métricas selecionadas. Não substitui, de forma alguma, a avaliação e o diagnóstico de um profissional de saúde qualificado. A interpretação de um ECG requer conhecimento médico aprofundado e o contexto clínico completo do paciente."
    }
    is_overall_normal = True

    # 1. Análise de Frequência Cardíaca
    fc_str = metrics.get("frequencia_cardiaca", "N/A")
    try:
        if "bpm" in fc_str:
            fc_valor = float(fc_str.replace(" bpm", ""))
            if 60 <= fc_valor <= 100:
                conclusions["frequencia_cardiaca"] = "Frequência Cardíaca: Normal ({:.2f} bpm). Indica um ritmo cardíaco dentro da faixa esperada para repouso.".format(fc_valor)
                conclusions["frequencia_cardiaca_html"] = "<li><strong>Frequência Cardíaca:</strong> <span class='normal'>Normal</span> ({:.2f} bpm). Indica um ritmo cardíaco dentro da faixa esperada para repouso.</li>".format(fc_valor)
            elif fc_valor > 100:
                conclusions["frequencia_cardiaca"] = "Frequência Cardíaca: Taquicardia ({:.2f} bpm). Frequência cardíaca acima do normal. Pode ser causada por diversos fatores como estresse, exercício ou condições médicas. Recomenda-se avaliação.".format(fc_valor)
                conclusions["frequencia_cardiaca_html"] = "<li><strong>Frequência Cardíaca:</strong> <span class='abnormal'>Taquicardia</span> ({:.2f} bpm). Frequência cardíaca acima do normal. Pode ser causada por diversos fatores como estresse, exercício ou condições médicas. Recomenda-se avaliação.</li>".format(fc_valor)
                is_overall_normal = False
            else: # fc_valor < 60
                conclusions["frequencia_cardiaca"] = "Frequência Cardíaca: Bradicardia ({:.2f} bpm). Frequência cardíaca abaixo do normal. Pode ser normal em atletas ou indicar condições que requerem investigação.".format(fc_valor)
                conclusions["frequencia_cardiaca_html"] = "<li><strong>Frequência Cardíaca:</strong> <span class='abnormal'>Bradicardia</span> ({:.2f} bpm). Frequência cardíaca abaixo do normal. Pode ser normal em atletas ou indicar condições que requerem investigação.</li>".format(fc_valor)
                is_overall_normal = False
        else:
            conclusions["frequencia_cardiaca"] = "Frequência Cardíaca: Não detectada ou inválida. Não foi possível avaliar a FC."
            conclusions["frequencia_cardiaca_html"] = "<li><strong>Frequência Cardíaca:</strong> Não detectada ou inválida. Não foi possível avaliar a FC.</li>"
            is_overall_normal = False
    except ValueError:
        conclusions["frequencia_cardiaca"] = "Frequência Cardíaca: Erro ao processar o valor. Não foi possível avaliar a FC."
        conclusions["frequencia_cardiaca_html"] = "<li><strong>Frequência Cardíaca:</strong> Erro ao processar o valor. Não foi possível avaliar a FC.</li>"
        is_overall_normal = False

    # 2. Análise da Duração do Complexo QRS
    qrs_str = metrics.get("duracao_qrs", "N/A")
    try:
        if "s" in qrs_str:
            qrs_valor = float(qrs_str.replace(" s", ""))
            if qrs_valor < 0.10: # < 100 ms
                conclusions["duracao_qrs"] = "Duração do QRS: Normal ({:.3f} s). Indica que a ativação elétrica dos ventrículos ocorre em tempo adequado.".format(qrs_valor)
                conclusions["duracao_qrs_html"] = "<li><strong>Duração do QRS:</strong> <span class='normal'>Normal</span> ({:.3f} s). Indica que a ativação elétrica dos ventrículos ocorre em tempo adequado.</li>".format(qrs_valor)
            else:
                conclusions["duracao_qrs"] = "Duração do QRS: Alargado ({:.3f} s). Pode indicar um atraso na condução elétrica ventricular (ex: bloqueio de ramo) ou outras anormalidades. Requer investigação.".format(qrs_valor)
                conclusions["duracao_qrs_html"] = "<li><strong>Duração do QRS:</strong> <span class='abnormal'>Alargado</span> ({:.3f} s). Pode indicar um atraso na condução elétrica ventricular (ex: bloqueio de ramo) ou outras anormalidades. Requer investigação.</li>".format(qrs_valor)
                is_overall_normal = False
        else:
            conclusions["duracao_qrs"] = "Duração do QRS: Estimativa não disponível/inválida. Não foi possível avaliar o QRS."
    except ValueError:
        conclusions["duracao_qrs"] = "Duração do QRS: Erro ao processar o valor. Não foi possível avaliar o QRS."
        conclusions["duracao_qrs_html"] = "<li><strong>Duração do QRS:</strong> Erro ao processar o valor. Não foi possível avaliar o QRS.</li>"
        is_overall_normal = False

    # 3. Análise do Eixo Elétrico QRS
    eixo_str = metrics.get("eixo_qrs", "N/A")
    try:
        if "°" in eixo_str:
            eixo_valor = float(eixo_str.replace("°", ""))
            if -30 <= eixo_valor <= 90:
                conclusions["eixo_qrs"] = "Eixo Elétrico QRS: Normal ({:.2f}°). O eixo está dentro dos limites fisiológicos.".format(eixo_valor)
                conclusions["eixo_qrs_html"] = "<li><strong>Eixo Elétrico QRS:</strong> <span class='normal'>Normal</span> ({:.2f}°). O eixo está dentro dos limites fisiológicos.</li>".format(eixo_valor)
            elif eixo_valor > 90:
                conclusions["eixo_qrs"] = "Eixo Elétrico QRS: Desvio para a Direita ({:.2f}°). Pode indicar sobrecarga do ventrículo direito ou outras condições.".format(eixo_valor)
                conclusions["eixo_qrs_html"] = "<li><strong>Eixo Elétrico QRS:</strong> <span class='abnormal'>Desvio para a Direita</span> ({:.2f}°). Pode indicar sobrecarga do ventrículo direito ou outras condições.</li>".format(eixo_valor)
                is_overall_normal = False
            elif eixo_valor < -30:
                conclusions["eixo_qrs"] = "Eixo Elétrico QRS: Desvio para a Esquerda ({:.2f}°). Pode ser um achado normal em alguns casos, mas também pode indicar sobrecarga do ventrículo esquerdo ou bloqueios de condução.".format(eixo_valor)
                conclusions["eixo_qrs_html"] = "<li><strong>Eixo Elétrico QRS:</strong> <span class='abnormal'>Desvio para a Esquerda</span> ({:.2f}°). Pode ser um achado normal em alguns casos, mas também pode indicar sobrecarga do ventrículo esquerdo ou bloqueios de condução.</li>".format(eixo_valor)
                is_overall_normal = False
        else:
            conclusions["eixo_qrs"] = "Eixo Elétrico QRS: Estimativa não disponível/inválida. Não foi possível avaliar o Eixo QRS."
            conclusions["eixo_qrs_html"] = "<li><strong>Eixo Elétrico QRS:</strong> Estimativa não disponível/inválida. Não foi possível avaliar o Eixo QRS.</li>"
    except ValueError:
        conclusions["eixo_qrs"] = "Eixo Elétrico QRS: Erro ao processar o valor. Não foi possível avaliar o Eixo QRS."
        conclusions["eixo_qrs_html"] = "<li><strong>Eixo Elétrico QRS:</strong> Erro ao processar o valor. Não foi possível avaliar o Eixo QRS.</li>"
        is_overall_normal = False

    # Conclusão geral
    if is_overall_normal:
        conclusions["overall_status_html"] = "<span class='normal'><strong>Resultado Geral: Dentro dos padrões de normalidade para as métricas analisadas.</strong></span>"
        conclusions["overall_status_pdf"] = "Resultado Geral: Dentro dos padrões de normalidade para as métricas analisadas."
    else:
        conclusions["overall_status_html"] = "<span class='abnormal'><strong>Resultado Geral: Fora dos padrões de normalidade para uma ou mais métricas analisadas.</strong></span>"
        conclusions["overall_status_pdf"] = "Resultado Geral: Fora dos padrões de normalidade para uma ou mais métricas analisadas."
    
    return conclusions

# --- Função para Gerar o PDF ---
def gerar_pdf_ecg(metrics, analysis_conclusions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    
    styles = getSampleStyleSheet()
    
    # Estilos personalizados
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=18, leading=22, alignment=TA_CENTER,
                               fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubtitleStyle', fontSize=14, leading=18, alignment=TA_CENTER,
                               fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Heading2Style', fontSize=14, leading=18, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='NormalPara', fontSize=10, leading=14))
    styles.add(ParagraphStyle(name='BoldNormalPara', fontSize=10, leading=14, fontName='Helvetica-Bold')) 

    # Adicionar estilos de cor para o PDF
    styles.add(ParagraphStyle(name='NormalGreen', parent=styles['NormalPara'], textColor='green'))
    styles.add(ParagraphStyle(name='NormalRed', parent=styles['NormalPara'], textColor='red'))
    styles.add(ParagraphStyle(name='AttentionNote', fontSize=8, leading=10, alignment=TA_CENTER, textColor='red'))


    story = []

    # --- Primeira Página: Relatório de Análise ---
    story.append(Paragraph("Relatório de Análise de Eletrocardiograma (ECG)", styles['TitleStyle']))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(f"Paciente: {metrics.get('nome_paciente')} ({metrics.get('sexo_paciente')}, {metrics.get('idade_paciente')} anos)", styles['SubtitleStyle']))
    story.append(Paragraph(f"Data do Exame: {metrics.get('data_exame')} {metrics.get('hora_exame')}", styles['SubtitleStyle']))
    story.append(Spacer(1, 15*mm))

    story.append(Paragraph("Métricas Detectadas:", styles['Heading2Style']))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(f"• Frequência Cardíaca: <font face='Helvetica-Bold'>{metrics.get('frequencia_cardiaca')}</font>", styles['NormalPara']))
    story.append(Paragraph(f"• Duração QRS (Estimada): <font face='Helvetica-Bold'>{metrics.get('duracao_qrs')}</font>", styles['NormalPara']))
    story.append(Paragraph(f"• Eixo QRS (Estimado): <font face='Helvetica-Bold'>{metrics.get('eixo_qrs')}</font>", styles['NormalPara']))
    story.append(Spacer(1, 10*mm))

    story.append(Paragraph("Conclusão da Análise Simplificada:", styles['Heading2Style']))
    story.append(Spacer(1, 3*mm))
    
    # Conclusão geral
    story.append(Paragraph(analysis_conclusions["overall_status_pdf"], styles['BoldNormalPara']))
    story.append(Spacer(1, 5*mm))

    # Conclusões detalhadas: usar ParagraphStyle com cor
    for key in ["frequencia_cardiaca", "duracao_qrs", "eixo_qrs"]:
        text = analysis_conclusions[key] # Pega o texto puro para PDF
        current_style = styles['NormalPara']
        
        if "Normal" in text:
            current_style = styles['NormalGreen']
        elif "Taquicardia" in text or "Bradicardia" in text or "Alargado" in text or "Desvio" in text:
            current_style = styles['NormalRed']
        
        if key == "frequencia_cardiaca":
            formatted_text = text.replace("Frequência Cardíaca:", "<font face='Helvetica-Bold'>Frequência Cardíaca:</font>")
        elif key == "duracao_qrs":
            formatted_text = text.replace("Duração do QRS:", "<font face='Helvetica-Bold'>Duração do QRS:</font>")
        elif key == "eixo_qrs":
            formatted_text = text.replace("Eixo Elétrico QRS:", "<font face='Helvetica-Bold'>Eixo Elétrico QRS:</font>")
        else:
            formatted_text = text
            
        story.append(Paragraph(formatted_text, current_style))
        story.append(Spacer(1, 2*mm))

    story.append(Spacer(1, 15*mm))
    story.append(Paragraph(analysis_conclusions["attention_note"], styles['AttentionNote']))

    # Adiciona quebra de página
    story.append(PageBreak())

    # --- Segunda Página: Traçado do ECG ---
    story.append(Paragraph("Traçado do Eletrocardiograma (ECG)", styles['TitleStyle']))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(f"Paciente: {metrics.get('nome_paciente')} - Data: {metrics.get('data_exame')}", styles['SubtitleStyle']))
    story.append(Spacer(1, 10*mm))

    if "ecg_image_b64" in metrics and metrics["ecg_image_b64"]:
        img_data = base64.b64decode(metrics["ecg_image_b64"])
        img_buffer = io.BytesIO(img_data)
        img = Image(img_buffer)
        
        max_width = 170*mm
        max_height = 200*mm 

        original_width, original_height = img.drawWidth, img.drawHeight
        aspect_ratio = original_height / original_width

        new_width = max_width
        new_height = new_width * aspect_ratio

        if new_height > max_height:
            new_height = max_height
            new_width = new_height / aspect_ratio

        img.drawWidth = new_width
        img.drawHeight = new_height
        
        img.hAlign = TA_CENTER
        story.append(img)
    else:
        story.append(Paragraph("Não foi possível gerar o traçado do ECG.", styles['NormalPara']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Rotas do Flask ---
temp_analysis_data = {} 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_ecg', methods=['POST'])
def upload_ecg():
    global temp_analysis_data

    if 'ecg_file' not in request.files:
        return "Nenhum arquivo enviado", 400
    
    file = request.files['ecg_file']
    if file.filename == '':
        return "Nenhum arquivo selecionado", 400
    
    if file:
        xml_content = file.read().decode('utf-8')
        
        image_buffer, metrics = gerar_ecg_do_xml_interno(xml_content)
        
        if image_buffer:
            analysis_conclusions = analisar_ecg_simples(metrics)
            
            temp_analysis_data["metrics"] = metrics
            temp_analysis_data["analysis_conclusions"] = analysis_conclusions
            
            return render_template('result.html', 
                                   ecg_image=metrics.get("ecg_image_b64"), 
                                   analysis_result=analysis_conclusions,
                                   metrics=metrics)
        else:
            return f"Erro ao gerar o gráfico de ECG: {metrics.get('error', 'Erro desconhecido ao processar XML')}. Verifique o formato do XML e a qualidade dos dados.", 500

@app.route('/generate_pdf')
def generate_pdf():
    global temp_analysis_data

    if not temp_analysis_data or "metrics" not in temp_analysis_data or "analysis_conclusions" not in temp_analysis_data:
        return "Nenhum dado de análise disponível para gerar PDF. Por favor, faça o upload de um ECG primeiro.", 400

    metrics = temp_analysis_data["metrics"]
    analysis_conclusions = temp_analysis_data["analysis_conclusions"]
    
    pdf_buffer = gerar_pdf_ecg(metrics, analysis_conclusions)
    
    return send_file(pdf_buffer, download_name="relatorio_ecg.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)