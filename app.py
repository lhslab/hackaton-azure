import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv
import os

# Configurações do Custom Vision
load_dotenv()

ENDPOINT = os.environ.get('ENDPOINT')
PREDICTION_KEY = os.environ.get('PREDICTION_KEY')
PROJECT_ID = os.environ.get('PROJECT_ID')
ITERATION_NAME = os.environ.get('ITERATION_NAME')
PREDICTION_URL = f"{ENDPOINT}customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/{ITERATION_NAME}/image"

# Função para enviar a imagem ao Custom Vision
def get_prediction(image_data):
    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/octet-stream"
    }
    response = requests.post(PREDICTION_URL, headers=headers, data=image_data)
    return response.json()

# Interface Streamlit
st.title("Detecção de objetos cortantes")
st.write("Envie uma imagem para análise:")

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Exibir a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem Carregada", use_column_width=True)

    # Obter predição do Custom Vision
    with st.spinner("Analisando a imagem..."):
        image_data = uploaded_file.getvalue()
        prediction_result = get_prediction(image_data)
    
    # Exibir resultados
    st.write("### Resultado da Predição:")
    for prediction in prediction_result["predictions"]:
        st.write(f"**{prediction['tagName']}**: {prediction['probability']:.2%}")
