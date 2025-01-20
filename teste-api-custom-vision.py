import requests
from dotenv import load_dotenv
import os

# Configurações do Custom Vision
load_dotenv()

ENDPOINT = os.environ.get('ENDPOINT')
PREDICTION_KEY = os.environ.get('PREDICTION_KEY')
PROJECT_ID = os.environ.get('PROJECT_ID')
ITERATION_NAME = os.environ.get('ITERATION_NAME')

headers = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream",
}

# Carregar a imagem para testar
with open("./image/teste.jpg", "rb") as image:
    data = image.read()

response = requests.post(
    f"{ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/{ITERATION_NAME}/image",
    headers=headers,
    data=data,
)
print(response.json())
