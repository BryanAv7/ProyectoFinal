from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Configuración general
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# LBP
cascade_path = "modelo/cascade.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# SIFT
logo_path = "static/logoCatedra2025.png"
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
sift = cv2.SIFT_create()
kp_logo, des_logo = sift.detectAndCompute(logo, None)

# Frame compartido
last_frame = None
frame_lock = threading.Lock()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Rutas de la aplicación
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

@app.route('/')
def home():
    return render_template('dashboard.html')  # detección de rostros (LBP)

@app.route('/sift')
def sift_page():
    return render_template('sift.html')  # stream de detección con SIFT

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global last_frame
    data = request.get_data()

    with frame_lock:
        last_frame = data

    # Decodificar
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Procesar con LBP y emitir
    process_with_lbp_and_emit(frame)

    # Procesar con SIFT y emitir
    processed_sift = process_frame_sift(frame)
    _, buffer_sift = cv2.imencode('.jpg', processed_sift)
    sift_encoded = base64.b64encode(buffer_sift).decode('utf-8')
    output_sift = f'data:image/jpeg;base64,{sift_encoded}'
    socketio.emit('sift_frame', {'image': output_sift})

    return ('', 204)


@app.route('/stream')
def stream():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Funciones auxiliares
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def process_with_lbp_and_emit(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(38, 46))

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        small = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = pixelated

    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    output_image = f'data:image/jpeg;base64,{frame_encoded}'

    socketio.emit('processed_frame', {'image': output_image})


def process_frame_sift(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

    if des_logo is None or des_frame is None:
        # No se pueden hacer matches, devolver el frame sin matches
        return frame

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_logo, des_frame)

    frame_with_matches = cv2.drawMatches(logo, kp_logo, frame, kp_frame, matches, None, flags=2)

    # Superponer logo
    logo_resized = cv2.resize(logo, (100, 100))
    h, w, _ = frame.shape
    x_offset = (w - 100) // 2
    y_offset = 10

    if logo_resized.shape[2] == 4:
        alpha_channel = logo_resized[:, :, 3] / 255.0
        for c in range(3):
            frame_with_matches[y_offset:y_offset+100, x_offset:x_offset+100, c] = (
                alpha_channel * logo_resized[:, :, c] +
                (1 - alpha_channel) * frame_with_matches[y_offset:y_offset+100, x_offset:x_offset+100, c]
            )
    else:
        frame_with_matches[y_offset:y_offset+100, x_offset:x_offset+100] = logo_resized

    return frame_with_matches


def mjpeg_generator():
    global last_frame
    boundary = b'--frame'
    while True:
        with frame_lock:
            frame = last_frame
        if frame:
            np_arr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            processed = process_frame_sift(img)
            ret, jpeg = cv2.imencode('.jpg', processed)
            if ret:
                yield boundary + b'\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        else:
            time.sleep(0.1)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Ejecutar servidor unificado
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if __name__ == '__main__':
    print("Servidor Flask unificado corriendo en http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
