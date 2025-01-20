import cv2
import requests
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Configurações do Custom Vision
load_dotenv()

# Defina os parâmetros do seu Custom Vision
ENDPOINT = os.environ.get('ENDPOINT')
PREDICTION_KEY = os.environ.get('PREDICTION_KEY')
PROJECT_ID = os.environ.get('PROJECT_ID')
ITERATION_NAME = os.environ.get('ITERATION_NAME')
ENDPOINT_RESOURCE = f"{ENDPOINT}customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/{ITERATION_NAME}/image"

# Função para fazer predições no Custom Vision
def predict_objects(frame):
    headers = {
        'Content-Type': 'application/octet-stream',
        'Prediction-Key': PREDICTION_KEY,
    }

    # Converter o frame para o formato necessário (base64 ou binário)
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Enviar a imagem para a API de predição do Custom Vision
    response = requests.post(
        ENDPOINT_RESOURCE,
        headers=headers,
        data=img_bytes
    )

    if response.status_code == 200:
        result = response.json()
        return result['predictions']
    else:
        print(f"Erro ao fazer a predição: {response.status_code}")
        print(response.text)
        return []

# Função para processar o vídeo e identificar facas e tesouras
def detect_objects_in_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop para processar cada frame do vídeo
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Obter as predições para o frame
        predictions = predict_objects(frame)

        # Desenhar caixas delimitadoras para os objetos detectados
        for prediction in predictions:
            if prediction['probability'] > 0.5:  # Filtrar com base na probabilidade (ajuste o limiar conforme necessário)
                box = prediction['boundingBox']
                x, y, w, h = int(box['left'] * width), int(box['top'] * height), int(box['width'] * width), int(box['height'] * height)

                # Desenhar a caixa delimitadora no frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, prediction['tagName'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Escrever o frame processado no arquivo de vídeo de saída   
        out.write(frame)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, './video/video_hackthon_cut3.mp4')
output_video_path = os.path.join(script_dir, './video/output_video_with_objects.mp4')  # Nome do vídeo de saída

# Chamar a função para detectar objetos no vídeo e salvar o vídeo processado
detect_objects_in_video(input_video_path, output_video_path)