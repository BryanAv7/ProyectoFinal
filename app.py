# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Practica Final - Detección de Rostros con LBP
# Integrantes: Bryan Avila
# Carrera: Ingenieria en Ciencias de Computación
# Fecha: 25/07/2023
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# app.py - Servidor Flask para detección de rostros con LBP y WebSocket
# Librerias por utilizar
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Cargar clasificador LBP para detección de rostros
cascade_path = "/home/bryan/Documentos/proyectoFinal/modelo/cascade.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Rutas de las vistas
@app.route('/mobile')
def mobile():
    return render_template('mobile.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Función handle_frame que procesa los frames recibidos
@socketio.on('frame')
def handle_frame(data):
    print("🟢 Frame recibido en servidor")

    # Decodificar imagen base64
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convertir a escala de grises para detección
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros con LBP(parametros ajustados)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(38, 46),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Pixelear rostros detectados
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        small = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = pixelated

    # Codificar imagen procesada a base64 para enviar al cliente
    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    output_image = f'data:image/jpeg;base64,{frame_encoded}'

    # Enviar el frame procesado a todos los clientes conectados
    emit('processed_frame', {'image': output_image}, broadcast=True)

# Manejar la conexión del cliente
if __name__ == '__main__':
    print("Servidor Flask-SocketIO corriendo en http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
