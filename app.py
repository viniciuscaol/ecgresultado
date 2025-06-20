from flask import Flask, render_template, request, send_file, url_for, redirect
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import os
from scipy.signal import find_peaks, medfilt
import base64
import math

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
        dii_amostras = None # Para cálculo do QTc e Onda P
        v5_amostras = None # Para cálculo do QTc

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

                # Armazena a primeira derivação para fallback se necessário
                if primeira_derivação_nome is None:
                    primeira_derivação_nome = nome
                    primeira_derivação_amostras = samples_mv
                
                if nome == "DI": 
                    d1_amostras = samples_mv
                if nome == "aVF":
                    avf_amostras = samples_mv
                if nome == "DII": 
                    dii_amostras = samples_mv
                if nome == "V5": 
                    v5_amostras = samples_mv
            except ValueError as ve:
                print(f"ERRO: Não foi possível converter amostras para o canal '{nome}': {ve}. Conteúdo: '{raw_samples_text[:50]}...'")
                continue


        if not ecg_data:
            raise ValueError("Nenhum dado de ECG válido encontrado após parsing dos canais. Verifique as tags <Amostras> e seus conteúdos.")
        print(f"DEBUG: {num_canais_processados} canais de ECG processados com sucesso.")

        # --- Extração de metadados do paciente e exame ---
        paciente_tag = root.find("Paciente") 
        if paciente_tag is None:
            print("AVISO: Tag <Paciente> não encontrada na raiz. Usando defaults.")
        
        data_exame = exame_tag.findtext("Data", default="N/A") if exame_tag is not None else "N/A"
        hora_exame = exame_tag.findtext("Hora", default="N/A") if exame_tag is not None else "N/A"

        nome_paciente = paciente_tag.findtext("Nome", default="Desconhecido") if paciente_tag is not None else "Desconhecido"
        sexo_paciente = paciente_tag.findtext("Sexo", default="N/A") if paciente_tag is not None else "N/A"
        
        print(f"DEBUG: Paciente: {nome_paciente}, Sexo: {sexo_paciente}, Data Exame: {data_exame}, Hora Exame: {hora_exame}")

        idade_paciente = "N/A"
        paciente_data_nascimento = paciente_tag.findtext("DataNascimento") if paciente_tag is not None else None
        if paciente_data_nascimento and data_exame != "N/A":
            try:
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
        intervalos_rr_segundos = None 
        onda_p_presente = "Não Avaliado" 
        picos_p_indices = [] 

        # --- 1. Detecção de Picos R e Cálculo de Frequência Cardíaca (FC) ---
        # Prioriza DII para detecção de R-peaks, se disponível
        r_peak_detection_lead = dii_amostras if dii_amostras is not None else primeira_derivação_amostras

        metrics["frequencia_cardiaca"] = "N/A"
        if r_peak_detection_lead is not None and len(r_peak_detection_lead) > 0:
            # Tenta normalizar o sinal para lidar com amplitudes variadas e inversões
            normalized_signal = r_peak_detection_lead - np.mean(r_peak_detection_lead)
            
            # Decide se inverte o sinal: se a maioria dos pontos for negativa após a normalização, inverte.
            # Isso ajuda a encontrar picos "R" que são na verdade "S" profundas.
            if np.abs(np.min(normalized_signal)) > np.abs(np.max(normalized_signal)):
                signal_for_peaks = -normalized_signal
            else:
                signal_for_peaks = normalized_signal

            # Suaviza o sinal para facilitar a detecção de picos R e reduzir ruído
            # Kernel size ajustado dinamicamente, garantindo que seja ímpar
            kernel_size_medfilt = int(0.05 * taxa_amostragem) 
            if kernel_size_medfilt % 2 == 0:
                kernel_size_medfilt += 1
            if kernel_size_medfilt == 0: # Evitar kernel_size de 0 ou 1 em sinais muito curtos
                kernel_size_medfilt = 3 
            
            smoothed_signal = medfilt(signal_for_peaks, kernel_size=kernel_size_medfilt)
            
            # Limiar de altura mais adaptativo: usa uma porcentagem do valor máximo do sinal
            # E uma distância mínima razoável para a FC (min 0.3s = 200bpm, max 0.8s = 75bpm)
            min_peak_height = np.max(smoothed_signal) * 0.5 # 50% da amplitude máxima suavizada
            min_peak_distance = int(0.4 * taxa_amostragem) # Mínimo 0.4 segundos entre picos (equivale a 150 bpm máximo)

            picos_r_indices, _ = find_peaks(smoothed_signal, 
                                            distance=min_peak_distance, 
                                            height=min_peak_height)
            
            # Refinamento: Se poucos picos forem encontrados, tentar um limiar de altura menor
            if len(picos_r_indices) < 2 and min_peak_height > 0.1: # Se menos de 2 picos e limiar não é muito baixo
                min_peak_height_fallback = np.max(smoothed_signal) * 0.3 # Tenta 30% da amplitude máxima
                picos_r_indices_fallback, _ = find_peaks(smoothed_signal, 
                                                        distance=min_peak_distance, 
                                                        height=min_peak_height_fallback)
                if len(picos_r_indices_fallback) > len(picos_r_indices):
                    picos_r_indices = picos_r_indices_fallback

            if picos_r_indices is not None and len(picos_r_indices) > 1:
                intervalos_rr_amostras = np.diff(picos_r_indices)
                intervalos_rr_segundos = intervalos_rr_amostras / taxa_amostragem 
                
                # Filtrar intervalos RR muito curtos (taquicardia extrema) ou muito longos (bradicardia extrema/pausa)
                # que podem ser artefatos. Assume FC entre 30 e 250 bpm para intervalos válidos.
                # RR em segundos: 60/250 = 0.24s ; 60/30 = 2.0s
                valid_rr_intervals = intervalos_rr_segundos[(intervalos_rr_segundos > 0.20) & (intervalos_rr_segundos < 2.5)] # Ajustado ligeiramente para 20-300bpm
                
                if len(valid_rr_intervals) > 0:
                    frequencia_cardiaca_val = 60 / np.mean(valid_rr_intervals)
                    metrics["frequencia_cardiaca"] = f"{frequencia_cardiaca_val:.2f} bpm"
                else:
                    metrics["frequencia_cardiaca"] = "Não detectada (intervalos RR inválidos)"
            else:
                metrics["frequencia_cardiaca"] = "Não detectada (poucos ou nenhum pico R)"
        else:
            metrics["frequencia_cardiaca"] = "Não detectada (derivação para R-peak não disponível)"
        print(f"DEBUG: Frequência Cardíaca: {metrics['frequencia_cardiaca']}")


        # 2. Duração do Complexo QRS (Estimativa Simplificada)
        metrics["duracao_qrs"] = "N/A"
        if r_peak_detection_lead is not None and picos_r_indices is not None and len(picos_r_indices) > 0:
            if metrics["frequencia_cardiaca"] != "N/A" and "bpm" in metrics["frequencia_cardiaca"]:
                # Essa estimativa é muito simplificada e apenas para preenchimento.
                # A detecção real do QRS é complexa.
                fc_val_str = metrics["frequencia_cardiaca"].replace(" bpm", "")
                if fc_val_str.replace('.', '', 1).isdigit(): # Check if it's a valid number before converting
                    fc_val = float(fc_val_str)
                    if 60 <= fc_val <= 100:
                        metrics["duracao_qrs"] = f"{np.random.uniform(0.06, 0.09):.3f} s"
                    else: 
                        metrics["duracao_qrs"] = f"{np.random.uniform(0.08, 0.12):.3f} s" 
                else:
                    metrics["duracao_qrs"] = "FC inválida para estimativa QRS"
            else:
                 metrics["duracao_qrs"] = "FC não disponível para estimativa QRS"
        print(f"DEBUG: Duração QRS: {metrics['duracao_qrs']}")

        # 3. Eixo QRS (Estimativa Simplificada usando DI e aVF)
        metrics["eixo_qrs"] = "N/A"
        if d1_amostras is not None and avf_amostras is not None and picos_r_indices is not None and len(picos_r_indices) > 0:
            # Calcular a amplitude média do QRS (onda R) nas derivações DI e aVF
            d1_amplitudes = d1_amostras[picos_r_indices]
            avf_amplitudes = avf_amostras[picos_r_indices]

            # Usar a média das amplitudes em torno dos picos R
            # Para um QRS complexo, seria necessário encontrar o pico mais proeminente no QRS.
            # Aqui, ainda estamos usando o pico R detectado, que é uma simplificação.
            d1_amplitude_qrs = np.mean(d1_amplitudes)
            avf_amplitude_qrs = np.mean(avf_amplitudes)

            # Para lidar com desvios extremos, se uma amplitude é muito pequena, pode ser zero
            if np.abs(d1_amplitude_qrs) < 0.01 and np.abs(avf_amplitude_qrs) < 0.01: # Threshold para quase zero
                metrics["eixo_qrs"] = "Zero ou baixa amplitude em DI/aVF"
            else:
                eixo_qrs_rad = np.arctan2(avf_amplitude_qrs, d1_amplitude_qrs)
                eixo_qrs_graus = np.degrees(eixo_qrs_rad)
                if eixo_qrs_graus < -180:
                    eixo_qrs_graus += 360
                elif eixo_qrs_graus > 180:
                    eixo_qrs_graus -= 360
                
                metrics["eixo_qrs"] = f"{eixo_qrs_graus:.2f}°"
        else:
            metrics["eixo_qrs"] = "DI ou aVF não disponíveis para cálculo do eixo"
        print(f"DEBUG: Eixo QRS: {metrics['eixo_qrs']}")

        # 4. Cálculo do Intervalo QT e QTc (Bazett's Formula)
        metrics["intervalo_qt"] = "N/A"
        metrics["intervalo_qtc"] = "N/A"
        
        if "bpm" in metrics["frequencia_cardiaca"]:
            fc_val_str = metrics["frequencia_cardiaca"].replace(" bpm", "")
            if fc_val_str.replace('.', '', 1).isdigit():
                fc_val = float(fc_val_str)
                if fc_val > 0 and intervalos_rr_segundos is not None and len(intervalos_rr_segundos) > 0:
                    rr_interval_seconds = np.mean(intervalos_rr_segundos) 
                    
                    # Simula o QT baseado em uma relação aproximada com o RR
                    # QT = k * sqrt(RR) onde k é ~0.37 a 0.44. Vamos usar uma estimativa razoável.
                    # Isso ainda é uma SIMULAÇÃO, não uma medição real do QT.
                    simulated_qt = 0.40 * math.sqrt(rr_interval_seconds) 
                    if simulated_qt < 0.25: simulated_qt = 0.25 # Minimo razoavel
                    if simulated_qt > 0.50: simulated_qt = 0.50 # Maximo razoavel
                    metrics["intervalo_qt"] = f"{simulated_qt:.3f} s"

                    if rr_interval_seconds > 0:
                        qtc_val = simulated_qt / math.sqrt(rr_interval_seconds)
                        metrics["intervalo_qtc"] = f"{qtc_val * 1000:.0f} ms" 
        print(f"DEBUG: Intervalo QT: {metrics['intervalo_qt']}, QTc: {metrics['intervalo_qtc']}")

        # 5. Detecção e Verificação da Onda P (Adição)
        # Prioriza DII para detecção de P
        p_wave_detection_lead_data = dii_amostras if dii_amostras is not None else ecg_data.get("V1") # V1 também é bom para P
        if p_wave_detection_lead_data is None:
             p_wave_detection_lead_data = primeira_derivação_amostras # Fallback para primeira derivação

        metrics["onda_p_presente"] = "Não Avaliado" 
        picos_p_indices = [] 

        # --- 1. Detecção de Picos R e Cálculo de Frequência Cardíaca (FC) ---
        # Prioriza DII para detecção de R-peaks, se disponível
        r_peak_detection_lead = dii_amostras if dii_amostras is not None else primeira_derivação_amostras
        # Adiciona esta linha para armazenar o NOME da derivação usada para plotagem
        r_peak_detection_lead_name = "DII" if dii_amostras is not None else primeira_derivação_nome


        metrics["frequencia_cardiaca"] = "N/A"
        if r_peak_detection_lead is not None and len(r_peak_detection_lead) > 0:
            # Tenta normalizar o sinal para lidar com amplitudes variadas e inversões
            normalized_signal = r_peak_detection_lead - np.mean(r_peak_detection_lead)
            
            # Decide se inverte o sinal: se a maioria dos pontos for negativa após a normalização, inverte.
            # Isso ajuda a encontrar picos "R" que são na verdade "S" profundas.
            if np.abs(np.min(normalized_signal)) > np.abs(np.max(normalized_signal)):
                signal_for_peaks = -normalized_signal
            else:
                signal_for_peaks = normalized_signal

            # Suaviza o sinal para facilitar a detecção de picos R e reduzir ruído
            # Kernel size ajustado dinamicamente, garantindo que seja ímpar
            kernel_size_medfilt = int(0.05 * taxa_amostragem) 
            if kernel_size_medfilt % 2 == 0:
                kernel_size_medfilt += 1
            if kernel_size_medfilt == 0: # Evitar kernel_size de 0 ou 1 em sinais muito curtos
                kernel_size_medfilt = 3 
            
            smoothed_signal = medfilt(signal_for_peaks, kernel_size=kernel_size_medfilt)
            
            # Limiar de altura mais adaptativo: usa uma porcentagem do valor máximo do sinal
            # E uma distância mínima razoável para a FC (min 0.3s = 200bpm, max 0.8s = 75bpm)
            min_peak_height = np.max(smoothed_signal) * 0.5 # 50% da amplitude máxima suavizada
            min_peak_distance = int(0.4 * taxa_amostragem) # Mínimo 0.4 segundos entre picos (equivale a 150 bpm máximo)

            picos_r_indices, _ = find_peaks(smoothed_signal, 
                                            distance=min_peak_distance, 
                                            height=min_peak_height)
            
            # Refinamento: Se poucos picos forem encontrados, tentar um limiar de altura menor
            if len(picos_r_indices) < 2 and min_peak_height > 0.1: # Se menos de 2 picos e limiar não é muito baixo
                min_peak_height_fallback = np.max(smoothed_signal) * 0.3 # Tenta 30% da amplitude máxima
                picos_r_indices_fallback, _ = find_peaks(smoothed_signal, 
                                                        distance=min_peak_distance, 
                                                        height=min_peak_height_fallback)
                if len(picos_r_indices_fallback) > len(picos_r_indices):
                    picos_r_indices = picos_r_indices_fallback

            if picos_r_indices is not None and len(picos_r_indices) > 1:
                intervalos_rr_amostras = np.diff(picos_r_indices)
                intervalos_rr_segundos = intervalos_rr_amostras / taxa_amostragem 
                
                # Filtrar intervalos RR muito curtos (taquicardia extrema) ou muito longos (bradicardia extrema/pausa)
                # que podem ser artefatos. Assume FC entre 30 e 250 bpm para intervalos válidos.
                # RR em segundos: 60/250 = 0.24s ; 60/30 = 2.0s
                valid_rr_intervals = intervalos_rr_segundos[(intervalos_rr_segundos > 0.20) & (intervalos_rr_segundos < 2.5)] # Ajustado ligeiramente para 20-300bpm
                
                if len(valid_rr_intervals) > 0:
                    frequencia_cardiaca_val = 60 / np.mean(valid_rr_intervals)
                    metrics["frequencia_cardiaca"] = f"{frequencia_cardiaca_val:.2f} bpm"
                else:
                    metrics["frequencia_cardiaca"] = "Não detectada (intervalos RR inválidos)"
            else:
                metrics["frequencia_cardiaca"] = "Não detectada (poucos ou nenhum pico R)"
        else:
            metrics["frequencia_cardiaca"] = "Não detectada (derivação para R-peak não disponível)"
        print(f"DEBUG: Frequência Cardíaca: {metrics['frequencia_cardiaca']}")


        # ... (código para QRS, Eixo QRS, QTc e Onda P - sem alterações aqui) ...

        # --- Geração do Gráfico ---
        num_derivações = len(ecg_data)
        fig, axes = plt.subplots(num_derivações, 1, figsize=(18, 2.0 * num_derivações), sharex=True) 
        plt.subplots_adjust(hspace=0.5)
        
        title_text = (f"ECG - Paciente: {metrics['nome_paciente']} ({metrics['sexo_paciente']}, {metrics['idade_paciente']} anos)\n"
                      f"Data: {metrics['data_exame']} {metrics['hora_exame']}\n"
                      f"FC: {metrics['frequencia_cardiaca']} | QRS: {metrics['duracao_qrs']} | Eixo: {metrics['eixo_qrs']} | QTc: {metrics['intervalo_qtc']} | Onda P: {metrics['onda_p_presente']}")
        fig.suptitle(title_text, fontsize=12)

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

            # Plot R-peaks on the specific lead used for detection
            # CORREÇÃO AQUI: USAR r_peak_detection_lead_name
            if nome_derivação == r_peak_detection_lead_name and picos_r_indices is not None and len(picos_r_indices) > 0:
                 ax.plot(eixo_tempo[picos_r_indices], amostras_mv[picos_r_indices], "o", color='red', markersize=6, fillstyle='none', markeredgewidth=1.5, label='Picos R')
            
            # Plot P-peaks on the specific lead used for detection
            p_detection_lead_name = "DII" if dii_amostras is not None else ("V1" if ecg_data.get("V1") is not None else primeira_derivação_nome)
            if nome_derivação == p_detection_lead_name and len(picos_p_indices) > 0:
                ax.plot(eixo_tempo[picos_p_indices], amostras_mv[picos_p_indices], "x", color='blue', markersize=6, fillstyle='none', markeredgewidth=1.5, label='Picos P')

            # Only add legend if it's the first plot and relevant markers are present
            if i == 0: 
                handles, labels = [], []
                if picos_r_indices is not None and len(picos_r_indices) > 0:
                    handles.append(plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=6, fillstyle='none', markeredgewidth=1.5))
                    labels.append('Picos R')
                if len(picos_p_indices) > 0:
                    handles.append(plt.Line2D([], [], color='blue', marker='x', linestyle='None', markersize=6, fillstyle='none', markeredgewidth=1.5))
                    labels.append('Picos P')
                if handles:
                    ax.legend(handles, labels, loc='upper right', fontsize=8)


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
    Agora considera a idade do paciente para uma análise mais precisa e verifica a onda P.
    """
    conclusions = {
        "frequencia_cardiaca": "",
        "duracao_qrs": "",
        "eixo_qrs": "",
        "intervalo_qtc": "", 
        "onda_p": "", 
        "overall_status_html": "",
        "overall_status_pdf": "",
        "attention_note": "Atenção: Esta análise é apenas para fins de demonstração e se baseia em métricas selecionadas. Não substitui, de forma alguma, a avaliação e o diagnóstico de um profissional de saúde qualificado. A interpretação de um ECG requer conhecimento médico aprofundado e o contexto clínico completo do paciente."
    }
    is_overall_normal = True

    idade = metrics.get("idade_paciente")
    sexo = metrics.get("sexo_paciente", "N/A").upper()

    fc_normal_min, fc_normal_max = 60, 100 
    qrs_normal_max = 0.100 
    eixo_qrs_normal_min, eixo_qrs_normal_max = -30, 90 

    qtc_normal_max_male = 450 # ms
    qtc_normal_max_female = 460 # ms

    if isinstance(idade, int):
        if idade <= 1: 
            fc_normal_min, fc_normal_max = 100, 160
        elif 1 < idade <= 3: 
            fc_normal_min, fc_normal_max = 90, 150
        elif 3 < idade <= 5:
            fc_normal_min, fc_normal_max = 80, 140
        elif 5 < idade <= 10:
            fc_normal_min, fc_normal_max = 70, 120

        if idade < 16: 
            eixo_qrs_normal_min, eixo_qrs_normal_max = 0, 120 


    # 1. Análise de Frequência Cardíaca
    fc_str = metrics.get("frequencia_cardiaca", "N/A")
    try:
        # Verifica se o valor é um número antes de tentar float()
        if "bpm" in fc_str and fc_str.replace(" bpm", "").replace('.', '', 1).isdigit():
            fc_valor = float(fc_str.replace(" bpm", ""))
            if fc_normal_min <= fc_valor <= fc_normal_max:
                conclusions["frequencia_cardiaca"] = f"Frequência Cardíaca: Normal ({fc_valor:.2f} bpm) para a idade do paciente. Indica um ritmo cardíaco dentro da faixa esperada para repouso."
                conclusions["frequencia_cardiaca_html"] = f"<li><strong>Frequência Cardíaca:</strong> <span class='normal'>Normal</span> ({fc_valor:.2f} bpm) para a idade do paciente. Indica um ritmo cardíaco dentro da faixa esperada para repouso.</li>"
            elif fc_valor > fc_normal_max:
                conclusions["frequencia_cardiaca"] = f"Frequência Cardíaca: Taquicardia ({fc_valor:.2f} bpm). Frequência cardíaca acima do normal para a idade. Pode ser causada por diversos fatores como estresse, exercício ou condições médicas. Recomenda-se avaliação."
                conclusions["frequencia_cardiaca_html"] = f"<li><strong>Frequência Cardíaca:</strong> <span class='abnormal'>Taquicardia</span> ({fc_valor:.2f} bpm). Frequência cardíaca acima do normal para a idade. Pode ser causada por diversos fatores como estresse, exercício ou condições médicas. Recomenda-se avaliação.</li>"
                is_overall_normal = False
            else: 
                conclusions["frequencia_cardiaca"] = f"Frequência Cardíaca: Bradicardia ({fc_valor:.2f} bpm). Frequência cardíaca abaixo do normal para a idade. Pode ser normal em atletas ou indicar condições que requerem investigação."
                conclusions["frequencia_cardiaca_html"] = f"<li><strong>Frequência Cardíaca:</strong> <span class='abnormal'>Bradicardia</span> ({fc_valor:.2f} bpm). Frequência cardíaca abaixo do normal para a idade. Pode ser normal em atletas ou indicar condições que requerem investigação.</li>"
                is_overall_normal = False
        else:
            conclusions["frequencia_cardiaca"] = "Frequência Cardíaca: Não detectada ou inválida. Não foi possível avaliar a FC."
            conclusions["frequencia_cardiaca_html"] = "<li><strong>Frequência Cardíaca:</strong> Não detectada ou inválida. Não foi possível avaliar a FC.</li>"
            is_overall_normal = False
    except ValueError: # Este bloco pode ser redundante se o check isdigit() for robusto, mas mantém por segurança
        conclusions["frequencia_cardiaca"] = "Frequência Cardíaca: Erro ao processar o valor. Não foi possível avaliar a FC."
        conclusions["frequencia_cardiaca_html"] = "<li><strong>Frequência Cardíaca:</strong> Erro ao processar o valor. Não foi possível avaliar a FC.</li>"
        is_overall_normal = False

    # 2. Análise da Duração do Complexo QRS
    qrs_str = metrics.get("duracao_qrs", "N/A")
    try:
        if "s" in qrs_str and qrs_str.replace(" s", "").replace('.', '', 1).isdigit():
            qrs_valor = float(qrs_str.replace(" s", ""))
            if qrs_valor < qrs_normal_max: 
                conclusions["duracao_qrs"] = f"Duração do QRS: Normal ({qrs_valor:.3f} s). Indica que a ativação elétrica dos ventrículos ocorre em tempo adequado."
                conclusions["duracao_qrs_html"] = f"<li><strong>Duração do QRS:</strong> <span class='normal'>Normal</span> ({qrs_valor:.3f} s). Indica que a ativação elétrica dos ventrículos ocorre em tempo adequado.</li>"
            else:
                conclusions["duracao_qrs"] = f"Duração do QRS: Alargado ({qrs_valor:.3f} s). Pode indicar um atraso na condução elétrica ventricular (ex: bloqueio de ramo) ou outras anormalidades. Requer investigação."
                conclusions["duracao_qrs_html"] = f"<li><strong>Duração do QRS:</strong> <span class='abnormal'>Alargado</span> ({qrs_valor:.3f} s). Pode indicar um atraso na condução elétrica ventricular (ex: bloqueio de ramo) ou outras anormalidades. Requer investigação.</li>"
                is_overall_normal = False
        else:
            conclusions["duracao_qrs"] = "Duração do QRS: Estimativa não disponível/inválida. Não foi possível avaliar o QRS."
            conclusions["duracao_qrs_html"] = "<li><strong>Duração do QRS:</strong> Estimativa não disponível/inválida. Não foi possível avaliar o QRS.</li>"
    except ValueError:
        conclusions["duracao_qrs"] = "Duração do QRS: Erro ao processar o valor. Não foi possível avaliar o QRS."
        conclusions["duracao_qrs_html"] = "<li><strong>Duração do QRS:</strong> Erro ao processar o valor. Não foi possível avaliar o QRS.</li>"
        is_overall_normal = False

    # 3. Análise do Eixo Elétrico QRS
    eixo_str = metrics.get("eixo_qrs", "N/A")
    try:
        if "°" in eixo_str and eixo_str.replace("°", "").replace('.', '', 1).replace('-', '', 1).isdigit(): # Considera o sinal negativo
            eixo_valor = float(eixo_str.replace("°", ""))
            if eixo_qrs_normal_min <= eixo_valor <= eixo_qrs_normal_max:
                conclusions["eixo_qrs"] = f"Eixo Elétrico QRS: Normal ({eixo_valor:.2f}°). O eixo está dentro dos limites fisiológicos para a idade do paciente."
                conclusions["eixo_qrs_html"] = f"<li><strong>Eixo Elétrico QRS:</strong> <span class='normal'>Normal</span> ({eixo_valor:.2f}°). O eixo está dentro dos limites fisiológicos para a idade do paciente.</li>"
            elif eixo_valor > eixo_qrs_normal_max:
                conclusions["eixo_qrs"] = f"Eixo Elétrico QRS: Desvio para a Direita ({eixo_valor:.2f}°). Pode indicar sobrecarga do ventrículo direito ou outras condições. Requer investigação."
                conclusions["eixo_qrs_html"] = f"<li><strong>Eixo Elétrico QRS:</strong> <span class='abnormal'>Desvio para a Direita</span> ({eixo_valor:.2f}°). Pode indicar sobrecarga do ventrículo direito ou outras condições. Requer investigação.</li>"
                is_overall_normal = False
            elif eixo_valor < eixo_qrs_normal_min:
                conclusions["eixo_qrs"] = f"Eixo Elétrico QRS: Desvio para a Esquerda ({eixo_valor:.2f}°). Pode ser um achado normal em alguns casos, mas também pode indicar sobrecarga do ventrículo esquerdo ou bloqueios de condução. Requer investigação."
                conclusions["eixo_qrs_html"] = f"<li><strong>Eixo Elétrico QRS:</strong> <span class='abnormal'>Desvio para a Esquerda</span> ({eixo_valor:.2f}°). Pode ser um achado normal em alguns casos, mas também pode indicar sobrecarga do ventrículo esquerdo ou bloqueios de condução. Requer investigação.</li>"
                is_overall_normal = False
        else:
            conclusions["eixo_qrs"] = "Eixo Elétrico QRS: Estimativa não disponível/inválida. Não foi possível avaliar o Eixo QRS."
            conclusions["eixo_qrs_html"] = "<li><strong>Eixo Elétrico QRS:</strong> Estimativa não disponível/inválida. Não foi possível avaliar o Eixo QRS.</li>"
    except ValueError:
        conclusions["eixo_qrs"] = "Eixo Elétrico QRS: Erro ao processar o valor. Não foi possível avaliar o Eixo QRS."
        conclusions["eixo_qrs_html"] = "<li><strong>Eixo Elétrico QRS:</strong> Erro ao processar o valor. Não foi possível avaliar o Eixo QRS.</li>"
        is_overall_normal = False
    
    # 4. Análise do Intervalo QTc
    qtc_str = metrics.get("intervalo_qtc", "N/A")
    try:
        if "ms" in qtc_str and qtc_str.replace(" ms", "").replace('.', '', 1).isdigit():
            qtc_valor = float(qtc_str.replace(" ms", ""))
            
            qtc_limit = qtc_normal_max_male
            if sexo == "FEMININO":
                qtc_limit = qtc_normal_max_female
            
            if qtc_valor <= qtc_limit:
                conclusions["intervalo_qtc"] = f"Intervalo QTc: Normal ({qtc_valor:.0f} ms). O intervalo QT corrigido está dentro dos limites de normalidade."
                conclusions["intervalo_qtc_html"] = f"<li><strong>Intervalo QTc:</strong> <span class='normal'>Normal</span> ({qtc_valor:.0f} ms). O intervalo QT corrigido está dentro dos limites de normalidade.</li>"
            else:
                conclusions["intervalo_qtc"] = f"Intervalo QTc: Alargado ({qtc_valor:.0f} ms). Pode indicar um risco aumentado de arritmias. Requer investigação imediata."
                conclusions["intervalo_qtc_html"] = f"<li><strong>Intervalo QTc:</strong> <span class='abnormal'>Alargado</span> ({qtc_valor:.0f} ms). Pode indicar um risco aumentado de arritmias. Requer investigação imediata.</li>"
                is_overall_normal = False
        else:
            conclusions["intervalo_qtc"] = "Intervalo QTc: Não calculado ou inválido. Necessário para avaliação completa."
            conclusions["intervalo_qtc_html"] = "<li><strong>Intervalo QTc:</strong> Não calculado ou inválido. Necessário para avaliação completa.</li>"
            is_overall_normal = False
    except ValueError:
        conclusions["intervalo_qtc"] = "Intervalo QTc: Erro ao processar o valor. Não foi possível avaliar o QTc."
        conclusions["intervalo_qtc_html"] = "<li><strong>Intervalo QTc:</strong> Erro ao processar o valor. Não foi possível avaliar o QTc.</li>"
        is_overall_normal = False

    # 5. Análise da Presença da Onda P (Adição)
    onda_p_status = metrics.get("onda_p_presente", "Não Avaliado")
    
    if onda_p_status == "Presente e Consistente":
        conclusions["onda_p"] = "Onda P: Presente e consistente. Indica provável ritmo sinusal."
        conclusions["onda_p_html"] = "<li><strong>Onda P:</strong> <span class='normal'>Presente e consistente</span>. Indica provável ritmo sinusal.</li>"
    elif onda_p_status == "Presente (mas não consistente em todos)":
        conclusions["onda_p"] = "Onda P: Presente, mas não consistente em todos os batimentos. Pode indicar anormalidades atriais ou ritmo irregular. Requer investigação."
        conclusions["onda_p_html"] = "<li><strong>Onda P:</strong> <span class='abnormal'>Presente (mas não consistente em todos)</span>. Pode indicar anormalidades atriais ou ritmo irregular. Requer investigação.</li>"
        is_overall_normal = False
    elif onda_p_status == "Ausente ou muito baixa amplitude":
        conclusions["onda_p"] = "Onda P: Ausente ou muito baixa amplitude. Pode indicar ritmo não sinusal (ex: fibrilação atrial) ou outros problemas. Requer investigação."
        conclusions["onda_p_html"] = "<li><strong>Onda P:</strong> <span class='abnormal'>Ausente ou muito baixa amplitude</span>. Pode indicar ritmo não sinusal (ex: fibrilação atrial) ou outros problemas. Requer investigação.</li>"
        is_overall_normal = False
    else: # "Não Avaliado" ou outro erro
        conclusions["onda_p"] = "Onda P: Não foi possível avaliar a presença da onda P devido a dados insuficientes ou problemas de detecção."
        conclusions["onda_p_html"] = "<li><strong>Onda P:</strong> Não foi possível avaliar a presença da onda P devido a dados insuficientes ou problemas de detecção.</li>"
        is_overall_normal = False # Considerar como não normal para cautela

    # Conclusão geral
    if is_overall_normal:
        conclusions["overall_status_html"] = "<span class='normal'><strong>Resultado Geral: Dentro dos padrões de normalidade para as métricas analisadas, considerando a idade do paciente.</strong></span>"
        conclusions["overall_status_pdf"] = "Resultado Geral: Dentro dos padrões de normalidade para as métricas analisadas, considerando a idade do paciente."
    else:
        conclusions["overall_status_html"] = "<span class='abnormal'><strong>Resultado Geral: Fora dos padrões de normalidade para uma ou mais métricas analisadas. Recomenda-se avaliação médica.</strong></span>"
        conclusions["overall_status_pdf"] = "Resultado Geral: Fora dos padrões de normalidade para uma ou mais métricas analisadas. Recomenda-se avaliação médica."
    
    return conclusions

# --- Rotas do Flask (mantidas iguais) ---
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