from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import uuid

# Inicializar FastAPI
app = FastAPI()

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Ruta del modelo y archivo de clases
MODEL_PATH = "hand_sign_model.h5"
CLASSES_FILE = "clases.txt"

# Cargar el modelo entrenado
model = load_model(MODEL_PATH)

# Leer las etiquetas desde el archivo de texto
with open(CLASSES_FILE, "r") as file:
    class_labels = [line.strip() for line in file.readlines()]

# Función para extraer puntos articulares
def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        return [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]
    else:
        raise HTTPException(status_code=400, detail="No se detectaron manos en la imagen.")

# Ruta principal de la API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Guardar el archivo temporalmente
        file_content = await file.read()
        temp_file_name = f"temp_{uuid.uuid4().hex}.jpg"
        with open(temp_file_name, "wb") as temp_file:
            temp_file.write(file_content)

        # Leer la imagen con OpenCV
        image = cv2.imread(temp_file_name)
        if image is None:
            raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")

        # Extraer puntos articulares
        landmarks = extract_hand_landmarks(image)
        if landmarks:
            # Convertir a array numpy y agregar dimensión para el modelo
            input_data = np.array([landmarks])  # Dimensión (1, N)
            
            # Realizar la predicción
            prediction = model.predict(input_data)
            predicted_class = int(np.argmax(prediction, axis=1)[0])  # Convertir a entero nativo de Python
            probability = float(prediction[0][predicted_class] * 100)  # Probabilidad en porcentaje como flotante

            # Obtener la etiqueta de la clase
            predicted_label = class_labels[predicted_class]

            # Responder con un JSON
            return {
                "predicted_label": predicted_label,
                "probability": f"{probability:.2f}%",
                "predicted_class": predicted_class
            }
        else:
            raise HTTPException(status_code=400, detail="No se pudo procesar la imagen para obtener puntos articulares.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)
