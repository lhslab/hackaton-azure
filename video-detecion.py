import cv2
import requests
import json
from dotenv import load_dotenv
import os

# Configurações do Custom Vision
load_dotenv()

# Configurações do Custom Vision
ENDPOINT = os.environ.get('ENDPOINT')
PREDICTION_KEY = os.environ.get('PREDICTION_KEY')
PROJECT_ID = os.environ.get('PROJECT_ID')
ITERATION_NAME = os.environ.get('ITERATION_NAME')
ENDPOINT_RESOURCE = f"{ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/{ITERATION_NAME}/image"

HEADERS = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream"
}

# Função para enviar um frame ao Custom Vision e obter o resultado
def analyze_frame(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(ENDPOINT_RESOURCE, headers=HEADERS, data=img_encoded.tobytes())
    return response.json()

# Função principal para processar vídeo
def process_video(video_path):
    # Carrega o vídeo
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Fim do vídeo

        # Redimensiona o frame (opcional, dependendo do modelo)
        resized_frame = cv2.resize(frame, (640, 480))  # Ajuste conforme necessário

        # Envia para o Custom Vision
        result = analyze_frame(resized_frame)

        # Processa o resultado
        predictions = result.get("predictions", [])
        for prediction in predictions:
            tag = prediction["tagName"]
            probability = prediction["probability"]
            if probability > 0.6:  # Threshold de confiança
                print(f"Objeto detectado: {tag} com confiança de {probability:.2f}")

        # Opcional: Mostrar frame ao vivo
        cv2.imshow("Frame", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
            break

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()

# Caminho do vídeo
video_path = "./video/video_hackthon_cut3.mp4"
process_video(video_path)
