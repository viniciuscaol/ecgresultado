async function uploadEcgFile() {
    const fileInput = document.getElementById('ecgFileInput');
    const errorMessage = document.getElementById('errorMessage');
    const loadingMessage = document.getElementById('loadingMessage');
    const ecgResultDiv = document.getElementById('ecgResult');
    
    errorMessage.style.display = 'none';
    ecgResultDiv.innerHTML = ''; // Limpa resultados anteriores

    if (fileInput.files.length === 0) {
        errorMessage.textContent = 'Por favor, selecione um arquivo XML.';
        errorMessage.style.display = 'block';
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('ecg_file', file);

    loadingMessage.style.display = 'block'; // Mostra mensagem de carregamento

    try {
        const response = await fetch('/upload_ecg', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);
            const imgElement = document.createElement('img');
            imgElement.src = imageUrl;
            imgElement.alt = 'Gráfico de ECG';
            ecgResultDiv.appendChild(imgElement);
        } else {
            const errorText = await response.text();
            errorMessage.textContent = `Erro do servidor: ${response.status} - ${errorText}`;
            errorMessage.style.display = 'block';
        }
    } catch (error) {
        errorMessage.textContent = `Erro de rede ou cliente: ${error.message}`;
        errorMessage.style.display = 'block';
    } finally {
        loadingMessage.style.display = 'none'; // Esconde mensagem de carregamento
    }
}