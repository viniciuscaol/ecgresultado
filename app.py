from flask import Flask, render_template, request, send_file, url_for, redirect
import os
import uuid
import json
import logging
from ecg_analysis import (
    parse_ecg_from_xml,
    analyze_ecg_signals,
    generate_ecg_plot,
    generate_analysis_report,
    gerar_pdf_ecg
)

app = Flask(__name__)

# --- Configuração ---
# MELHORIA: Cria pastas temporárias para uploads e resultados
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'temp_results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# MELHORIA: Configura o logging para depuração
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    """Renderiza a página inicial de upload."""
    return render_template('index.html')

@app.route('/upload_ecg', methods=['POST'])
def upload_ecg():
    """Processa o upload do arquivo ECG, analisa e mostra os resultados."""
    if 'ecg_file' not in request.files:
        return "Nenhum arquivo enviado", 400
    
    file = request.files['ecg_file']
    if file.filename == '':
        return "Nenhum arquivo selecionado", 400
    
    if file:
        try:
            xml_content = file.read().decode('utf-8')
            
            # --- Etapa 1: Parse do XML ---
            ecg_data, taxa_amostragem, metadata = parse_ecg_from_xml(xml_content)
            app.logger.info(f"XML do paciente {metadata['nome_paciente']} parseado com sucesso.")

            # --- Etapa 2: Análise do Sinal ---
            metrics, artifacts = analyze_ecg_signals(ecg_data, taxa_amostragem)
            metrics.update(metadata) # Junta os metadados do paciente com as métricas
            app.logger.info(f"Análise de sinal concluída. FC: {metrics['frequencia_cardiaca']}")
            
            # --- Etapa 3: Geração da Imagem ---
            ecg_image_b64 = generate_ecg_plot(ecg_data, taxa_amostragem, metrics, artifacts)
            app.logger.info("Gráfico do ECG gerado.")

            # --- Etapa 4: Geração do Laudo ---
            analysis_report = generate_analysis_report(metrics)
            
            # MELHORIA: Salva os resultados em um arquivo temporário com ID único
            analysis_id = str(uuid.uuid4())
            result_data = {
                "metrics": metrics,
                "analysis_report": analysis_report,
                "ecg_image_b64": ecg_image_b64
            }
            result_filepath = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}.json")
            with open(result_filepath, 'w') as f:
                json.dump(result_data, f)
            app.logger.info(f"Resultados salvos em arquivo temporário: {analysis_id}.json")

            return redirect(url_for('show_result', analysis_id=analysis_id))

        except ValueError as e:
            app.logger.error(f"Erro de Validação no processamento do XML: {e}")
            return f"Erro ao processar o arquivo: {e}. Verifique se o formato do XML está correto e se contém as tags necessárias.", 500
        except Exception as e:
            app.logger.error(f"Erro inesperado: {e}", exc_info=True)
            return "Ocorreu um erro inesperado no servidor.", 500

@app.route('/result/<analysis_id>')
def show_result(analysis_id):
    """Exibe a página de resultados carregando os dados do arquivo JSON."""
    result_filepath = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}.json")
    try:
        with open(result_filepath, 'r') as f:
            data = json.load(f)
        return render_template('result.html', 
                               ecg_image=data["ecg_image_b64"], 
                               analysis_result=data["analysis_report"],
                               metrics=data["metrics"],
                               analysis_id=analysis_id)
    except FileNotFoundError:
        return "Resultado da análise não encontrado. Por favor, faça o upload novamente.", 404

@app.route('/generate_pdf/<analysis_id>')
def generate_pdf(analysis_id):
    """Gera um PDF a partir dos dados de análise salvos."""
    result_filepath = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}.json")
    try:
        with open(result_filepath, 'r') as f:
            data = json.load(f)
        
        pdf_buffer = gerar_pdf_ecg(data["metrics"], data["analysis_report"], data["ecg_image_b64"])
        
        # Opcional: remover o arquivo JSON após o uso
        # os.remove(result_filepath)
        
        return send_file(pdf_buffer, download_name="relatorio_ecg.pdf", mimetype='application/pdf')
        
    except FileNotFoundError:
        return "Dados da análise não encontrados para gerar o PDF.", 404

if __name__ == '__main__':
    app.run(debug=True)