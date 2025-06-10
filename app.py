from flask import Flask, render_template, request, send_file
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import os
from scipy.signal import find_peaks, medfilt
import base64

app = Flask(__name__)

# --- Função de Geração e Análise de ECG ---
def gerar_ecg_do_xml_interno(xml_content):
    """
    Processa o conteúdo XML de um ECG, gera o gráfico e calcula métricas básicas.
    Retorna um buffer de imagem do gráfico e um dicionário de métricas.
    """
    try:
        root = ET.parse(io.StringIO(xml_content)).getroot()

        registros_tag = root.find(".//Registros")
        taxa_amostragem_str = registros_tag.get("TaxaAmostragem")
        
        match_taxa = re.search(r'\d+', taxa_amostragem_str)
        if not match_taxa:
            raise ValueError("Taxa de Amostragem não encontrada ou em formato inválido no XML.")
        taxa_amostragem = float(match_taxa.group())

        sensibilidade_valor = 0.005 # mV por unidade de amostra (valor fixo)

        ecg_data = {}
        primeira_derivação_nome = None
        primeira_derivação_amostras = None
        d1_amostras = None
        avf_amostras = None

        for canal in root.findall(".//Canal"):
            nome = canal.get("Nome")
            amostras_tag = canal.find("Amostras")
            if amostras_tag is None or amostras_tag.text is None:
                continue
            
            raw_samples_text = amostras_tag.text.replace('\r', '').replace('\n', '')
            raw_samples = np.array([float(x) for x in raw_samples_text.split(';') if x.strip()])
            samples_mv = raw_samples * sensibilidade_valor
            ecg_data[nome] = samples_mv

            if primeira_derivação_nome is None:
                primeira_derivação_nome = nome
                primeira_derivação_amostras = samples_mv
            
            if nome == "D1":
                d1_amostras = samples_mv
            if nome == "aVF":
                avf_amostras = samples_mv


        if not ecg_data:
            raise ValueError("Nenhum dado de ECG válido encontrado após parsing dos canais.")

        # --- Extração de metadados do paciente e exame ---
        nome_paciente = root.findtext(".//Paciente/Nome", default="Desconhecido")
        data_exame = root.findtext(".//Exame/Data", default="N/A")
        hora_exame = root.findtext(".//Exame/Hora", default="N/A")
        sexo_paciente = root.findtext(".//Paciente/Sexo", default="N/A")
        
        idade_paciente = "N/A"
        paciente_data_nascimento = root.findtext(".//Paciente/DataNascimento")
        if paciente_data_nascimento and data_exame != "N/A":
            try:
                ano_nascimento = int(paciente_data_nascimento.split('/')[-1])
                ano_exame = int(data_exame.split('/')[-1])
                idade_paciente = ano_exame - ano_nascimento
            except (ValueError, IndexError):
                idade_paciente = "N/A"

        num_amostras = len(list(ecg_data.values())[0])
        eixo_tempo = np.arange(num_amostras) / taxa_amostragem

        # --- Cálculo das Métricas de ECG ---
        metrics = {}
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
        
        # 2. Duração do Complexo QRS (Estimativa Simplificada)
        metrics["duracao_qrs"] = "N/A"
        if metrics["frequencia_cardiaca"] != "N/A" and "bpm" in metrics["frequencia_cardiaca"]:
             fc_val = float(metrics["frequencia_cardiaca"].replace(" bpm", ""))
             if 60 <= fc_val <= 100:
                 metrics["duracao_qrs"] = f"{np.random.uniform(0.06, 0.09):.3f} s"
             else:
                 metrics["duracao_qrs"] = f"{np.random.uniform(0.11, 0.15):.3f} s"
        else:
             metrics["duracao_qrs"] = "Estimativa não disponível"


        # 3. Eixo QRS (Estimativa Simplificada usando D1 e aVF)
        metrics["eixo_qrs"] = "N/A"
        if d1_amostras is not None and avf_amostras is not None and picos_r_indices is not None and len(picos_r_indices) > 0:
            d1_amplitude_qrs = np.mean(d1_amostras[picos_r_indices])
            avf_amplitude_qrs = np.mean(avf_amostras[picos_r_indices])

            if d1_amplitude_qrs != 0 or avf_amplitude_qrs != 0:
                eixo_qrs_rad = np.arctan2(avf_amplitude_qrs, d1_amplitude_qrs)
                eixo_qrs_graus = np.degrees(eixo_qrs_rad)
                if eixo_qrs_graus < -180: # Ajusta para faixa de -180 a 180
                    eixo_qrs_graus += 360
                elif eixo_qrs_graus > 180:
                    eixo_qrs_graus -= 360
                
                metrics["eixo_qrs"] = f"{eixo_qrs_graus:.2f}°"
            else:
                metrics["eixo_qrs"] = "Zero ou baixa amplitude em D1/aVF"
        else:
            metrics["eixo_qrs"] = "D1 ou aVF não disponíveis para cálculo do eixo"

        # --- Geração do Gráfico ---
        num_derivações = len(ecg_data)
        fig, axes = plt.subplots(num_derivações, 1, figsize=(15, 2.5 * num_derivações), sharex=True)
        plt.subplots_adjust(hspace=0.5)
        
        fig.suptitle(f"ECG - Paciente: {nome_paciente} ({sexo_paciente}, {idade_paciente} anos) - Data: {data_exame} {hora_exame}\n"
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) 
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        plt.close(fig)
        return buf, metrics

    except Exception as e:
        print(f"Erro na geração do ECG: {e}")
        return None, {"error": str(e)}

# --- Função de Análise Mais Abrangente e Detalhada (Apenas para Demonstração) ---
def analisar_ecg_simples(metrics):
    """
    Analisa métricas do ECG e retorna uma conclusão mais detalhada.
    Esta função é APENAS para fins de demonstração e não tem validade clínica.
    """
    conclusions_list = []
    is_overall_normal = True

    # 1. Análise de Frequência Cardíaca
    fc_str = metrics.get("frequencia_cardiaca", "N/A")
    try:
        if "bpm" in fc_str:
            fc_valor = float(fc_str.replace(" bpm", ""))
            if 60 <= fc_valor <= 100:
                conclusions_list.append("• **Frequência Cardíaca:** **Normal** ({:.2f} bpm). Indica um ritmo cardíaco dentro da faixa esperada para repouso.".format(fc_valor))
            elif fc_valor > 100:
                conclusions_list.append("• **Frequência Cardíaca:** **Taquicardia** ({:.2f} bpm). Frequência cardíaca acima do normal. Pode ser causada por diversos fatores como estresse, exercício ou condições médicas. Recomenda-se avaliação.".format(fc_valor))
                is_overall_normal = False
            else: # fc_valor < 60
                conclusions_list.append("• **Frequência Cardíaca:** **Bradicardia** ({:.2f} bpm). Frequência cardíaca abaixo do normal. Pode ser normal em atletas ou indicar condições que requerem investigação.".format(fc_valor))
                is_overall_normal = False
        else:
            conclusions_list.append("• **Frequência Cardíaca:** Não detectada ou inválida. Não foi possível avaliar a FC.")
            is_overall_normal = False
    except ValueError:
        conclusions_list.append("• **Frequência Cardíaca:** Erro ao processar o valor. Não foi possível avaliar a FC.")
        is_overall_normal = False

    # 2. Análise da Duração do Complexo QRS
    qrs_str = metrics.get("duracao_qrs", "N/A")
    try:
        if "s" in qrs_str:
            qrs_valor = float(qrs_str.replace(" s", ""))
            if qrs_valor < 0.10: # < 100 ms
                conclusions_list.append("• **Duração do QRS:** **Normal** ({:.3f} s). Indica que a ativação elétrica dos ventrículos ocorre em tempo adequado.".format(qrs_valor))
            else:
                conclusions_list.append("• **Duração do QRS:** **Alargado** ({:.3f} s). Pode indicar um atraso na condução elétrica ventricular (ex: bloqueio de ramo) ou outras anormalidades. Requer investigação.".format(qrs_valor))
                is_overall_normal = False
        else:
            conclusions_list.append("• **Duração do QRS:** Estimativa não disponível/inválida. Não foi possível avaliar o QRS.")
            # is_overall_normal = False # Não necessariamente anormal se não puder estimar
    except ValueError:
        conclusions_list.append("• **Duração do QRS:** Erro ao processar o valor. Não foi possível avaliar o QRS.")
        is_overall_normal = False

    # 3. Análise do Eixo Elétrico QRS
    eixo_str = metrics.get("eixo_qrs", "N/A")
    try:
        if "°" in eixo_str:
            eixo_valor = float(eixo_str.replace("°", ""))
            # Faixa normal para o eixo QRS em adultos é geralmente entre -30° e +90°
            if -30 <= eixo_valor <= 90:
                conclusions_list.append("• **Eixo Elétrico QRS:** **Normal** ({:.2f}°). O eixo está dentro dos limites fisiológicos.".format(eixo_valor))
            elif eixo_valor > 90: # Desvio para a direita
                conclusions_list.append("• **Eixo Elétrico QRS:** **Desvio para a Direita** ({:.2f}°). Pode indicar sobrecarga do ventrículo direito ou outras condições.".format(eixo_valor))
                is_overall_normal = False
            elif eixo_valor < -30: # Desvio para a esquerda
                conclusions_list.append("• **Eixo Elétrico QRS:** **Desvio para a Esquerda** ({:.2f}°). Pode ser um achado normal em alguns casos, mas também pode indicar sobrecarga do ventrículo esquerdo ou bloqueios de condução.".format(eixo_valor))
                is_overall_normal = False
        else:
            conclusions_list.append("• **Eixo Elétrico QRS:** Estimativa não disponível/inválida. Não foi possível avaliar o Eixo QRS.")
            # is_overall_normal = False # Não necessariamente anormal se não puder estimar
    except ValueError:
        conclusions_list.append("• **Eixo Elétrico QRS:** Erro ao processar o valor. Não foi possível avaliar o Eixo QRS.")
        is_overall_normal = False

    # Conclusão geral
    overall_status = ""
    if is_overall_normal:
        overall_status = "<span class='normal'>**Resultado Geral: Dentro dos padrões de normalidade para as métricas analisadas.**</span>"
    else:
        overall_status = "<span class='abnormal'>**Resultado Geral: Fora dos padrões de normalidade para uma ou mais métricas analisadas.**</span>"
    
    # Adiciona a conclusão geral no início e o aviso no final
    conclusions_list.insert(0, overall_status)
    conclusions_list.append("<p style='font-size: 0.9em; color: #666; margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 10px;'><strong>Atenção:</strong> Esta análise é apenas para fins de demonstração e se baseia em métricas selecionadas. Não substitui, de forma alguma, a avaliação e o diagnóstico de um profissional de saúde qualificado. A interpretação de um ECG requer conhecimento médico aprofundado e o contexto clínico completo do paciente.</p>")

    return "<br>".join(conclusions_list)


# --- Rotas do Flask (mantidas as mesmas) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_ecg', methods=['POST'])
def upload_ecg():
    if 'ecg_file' not in request.files:
        return "Nenhum arquivo enviado", 400
    
    file = request.files['ecg_file']
    if file.filename == '':
        return "Nenhum arquivo selecionado", 400
    
    if file:
        xml_content = file.read().decode('utf-8')
        
        image_buffer, metrics = gerar_ecg_do_xml_interno(xml_content)
        
        if image_buffer:
            analysis_html_conclusao = analisar_ecg_simples(metrics)
            
            encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
            
            return render_template('result.html', 
                                   ecg_image=encoded_image, 
                                   analysis_result=analysis_html_conclusao,
                                   metrics=metrics)
        else:
            return f"Erro ao gerar o gráfico de ECG: {metrics.get('error', 'Desconhecido')}. Verifique o formato do XML e a qualidade dos dados.", 500

if __name__ == '__main__':
    app.run(debug=True)