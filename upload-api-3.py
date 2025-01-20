import json
import os
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Configurações do Custom Vision
load_dotenv()

TRAINING_KEY = os.environ.get('TRAINING_KEY')
PROJECT_ID = os.environ.get('PROJECT_ID')
ENDPOINT = f"https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Training/projects/{PROJECT_ID}/images/regions"

# Função para carregar e normalizar as anotações do JSON COCO
def upload_images_with_annotations(json_file_path, image_folder):
    # Carregar o arquivo COCO JSON
    with open(json_file_path, 'r') as file:
        coco_data = json.load(file)

    # Mapear categorias para suas tags
    categories = {category['id']: category['name'] for category in coco_data['categories']}

    # Iterar sobre as imagens e fazer o upload
    for image in tqdm(coco_data['images'], desc="Fazendo upload das imagens"):
        image_path = os.path.join(image_folder, image['file_name'])

        if not os.path.exists(image_path):
            print(f"Imagem {image['file_name']} não encontrada no diretório.")
            continue

        # Preparar as anotações para essa imagem
        annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image['id']]

        # Preparar o payload de regiões
        regions = []
        for anno in annotations:
            category_id = anno['category_id']
            tag_name = categories[category_id]
            bbox = anno['bbox']  # [x, y, largura, altura]

            # Normalizar a bounding box em relação ao tamanho da imagem
            left = bbox[0] / image['width']
            top = bbox[1] / image['height']
            width = bbox[2] / image['width']
            height = bbox[3] / image['height']

            # Garantir que as dimensões normalizadas estão no intervalo [0, 1]
            if 0 <= left <= 1 and 0 <= top <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                regions.append({
                    "tagName": tag_name,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height
                })

        if not regions:
            print(f"Sem regiões válidas para a imagem {image['file_name']}. Pulando...")
            continue

        # Abrir a imagem em binário
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()

        # Preparar os cabeçalhos e o payload
        headers = {
            'Training-Key': TRAINING_KEY
        }
        payload = {
            "regions": regions
        }

        # Realizar o upload da imagem com as regiões
        response = requests.post(
            ENDPOINT,
            headers=headers,
            files={"imageData": image_data},
            data={"regions": json.dumps(payload["regions"])}
        )

        # Verificar a resposta
        if response.status_code == 200:
            print(f"Imagem {image['file_name']} carregada com sucesso.")
        else:
            print(f"Erro ao carregar a imagem {image['file_name']}: {response.status_code}")
            print(response.text)

# Caminho para o arquivo COCO JSON e a pasta das imagens
json_file_path = './train-teste/_annotations.coco.json'
image_folder = './train-teste'  # Ou o diretório onde suas imagens estão armazenadas

# Fazer o upload das imagens com as anotações
upload_images_with_annotations(json_file_path, image_folder)
