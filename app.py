import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# -------------------------
# CONFIGURACIÓN DE LA APP
# -------------------------
MODELO_PATH = "mejor_modelo.pth"
NUM_CLASSES = 38
IMG_SIZE = 224

# nombres de las 38 clases en el mismo orden que ImageFolder las cargó
NOMBRES_CLASES = [
    "Apple - Apple scab", "Apple - Black rot", "Apple - Cedar apple rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery mildew", "Cherry - Healthy",
    "Corn - Cercospora leaf spot", "Corn - Common rust", "Corn - Northern leaf blight", "Corn - Healthy",
    "Grape - Black rot", "Grape - Esca (Black Measles)", "Grape - Leaf blight", "Grape - Healthy",
    "Orange - Haunglongbing", "Peach - Bacterial spot", "Peach - Healthy",
    "Pepper - Bacterial spot", "Pepper - Healthy",
    "Potato - Early blight", "Potato - Late blight", "Potato - Healthy",
    "Raspberry - Healthy", "Soybean - Healthy", "Squash - Powdery mildew",
    "Strawberry - Leaf scorch", "Strawberry - Healthy",
    "Tomato - Bacterial spot", "Tomato - Early blight", "Tomato - Late blight",
    "Tomato - Leaf mold", "Tomato - Septoria leaf spot",
    "Tomato - Spider mites", "Tomato - Target spot",
    "Tomato - Yellow leaf curl virus", "Tomato - Mosaic virus", "Tomato - Healthy"
]

# -------------------------
# CARGA DEL MODELO
# -------------------------
@st.cache_resource
def cargar_modelo():
    modelo = models.efficientnet_b0(weights=None)
    num_features = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, NUM_CLASSES)
    )
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location="cpu", weights_only=True))
    modelo.eval()
    return modelo

# -------------------------
# TRANSFORMACIÓN DE IMAGEN
# -------------------------
def preparar_imagen(imagen):
    transformacion = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # añade dimensión de batch: (3, 224, 224) → (1, 3, 224, 224)
    return transformacion(imagen).unsqueeze(0)

# -------------------------
# INTERFAZ DE LA APP
# -------------------------
st.title("Detector de enfermedades en plantas")
st.write("Sube una foto de una hoja y el modelo identificará si está sana o enferma.")

imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(imagen, caption="Imagen subida", use_container_width=True)

    with col2:
        with st.spinner("Analizando..."):
            modelo = cargar_modelo()
            tensor = preparar_imagen(imagen)

            with torch.no_grad():
                salidas = modelo(tensor)
                probabilidades = torch.softmax(salidas, dim=1)[0]
                indice_predicho = probabilidades.argmax().item()
                confianza = probabilidades[indice_predicho].item()

            clase_predicha = NOMBRES_CLASES[indice_predicho]
            es_sana = "Healthy" in clase_predicha

        if es_sana:
            st.success(f"Planta sana")
        else:
            st.error(f"Enfermedad detectada")

        st.markdown(f"**{clase_predicha}**")
        st.metric("Confianza", f"{confianza*100:.1f}%")

        st.write("Top 3 predicciones:")
        top3 = probabilidades.topk(3)
        for prob, idx in zip(top3.values, top3.indices):
            st.write(f"- {NOMBRES_CLASES[idx.item()]}: {prob.item()*100:.1f}%")