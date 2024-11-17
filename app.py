from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import torch
from io import BytesIO
import cv2

app = Flask(__name__)

# Cargar el modelo YOLOv8 desde la carpeta models
model = YOLO('models/runs/detect/train/weights/best.pt') 



@app.route('/classify_image', methods=['POST'])
def classify_image():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    # Decodificar la imagen base64
    image_data = base64.b64decode(data['image'].split(',')[1])  # Eliminar "data:image/jpeg;base64,"
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Realizar la predicción con el modelo YOLOv8
    results = model(image)

    # Obtener la imagen anotada como un arreglo de numpy
    annotated_array = results[0].plot()

    # Convertir de BGR a RGB si es necesario
    annotated_array_rgb = annotated_array[..., ::-1]  # Cambia de BGR a RGB

    # Convertir el array numpy a una imagen de PIL
    annotated_image = Image.fromarray(annotated_array_rgb)

    # Crear una miniatura de tamaño específico
    desired_size = (700, 600)  # Tamaño de la miniatura (ancho, alto)
    annotated_image.thumbnail(desired_size)

    # Convertir la imagen a formato base64 para enviar al cliente
    img_io = io.BytesIO()
    annotated_image.save(img_io, 'JPEG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return jsonify({'img_data': img_base64})






@app.route('/')
def index():
    return render_template('index.html')




@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Leer y convertir la imagen
    image = Image.open(file.stream).convert('RGB')

    # Realizar la predicción con el modelo YOLOv8
    results = model(conf = 0.7, source=image)

    # Obtener la imagen anotada como un arreglo de numpy
    annotated_array = results[0].plot()

    # Convertir de BGR a RGB si es necesario
    annotated_array_rgb = annotated_array[..., ::-1]  # Cambia de BGR a RGB

    # Convertir el array numpy a una imagen de PIL
    annotated_image = Image.fromarray(annotated_array_rgb)

    # Crear una miniatura de tamaño específico
    desired_size = (700, 600)  # Tamaño de la miniatura (ancho, alto)
    annotated_image.thumbnail(desired_size)

    # Convertir la imagen a formato base64 para enviar al cliente
    img_io = io.BytesIO()
    annotated_image.save(img_io, 'JPEG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return render_template('result.html', img_data=img_base64)

if __name__ == '__main__':
    app.run(debug=True)




