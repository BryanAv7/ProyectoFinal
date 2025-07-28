from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import threading
import time
import psutil

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

# Medición de FPS y memoria
last_time = time.time()
fps = 0
process = psutil.Process()

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
    global last_time, fps
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

    # Medición de FPS y memoria
    current_time = time.time()
    fps = 1.0 / (current_time - last_time)
    last_time = current_time

    mem_usage = process.memory_info().rss / (1024 * 1024)  # en MB
    socketio.emit('stats', {
        'fps': round(fps, 2),
        'mem': round(mem_usage, 2)
    })

def process_frame_sift(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
    
    if des_logo is None or des_frame is None or len(kp_frame) < 10:
        # Si no hay suficientes puntos, mostrar solo el logo al lado del frame
        h, w = frame.shape[:2]
        result = np.zeros((max(h, logo.shape[0]), w + logo.shape[1], 3), dtype=np.uint8)
        result[:logo.shape[0], :logo.shape[1]] = logo[:, :, :3]  # Logo (sin canal alpha si existe)
        result[:h, logo.shape[1]:logo.shape[1]+w] = frame
        return result

    # Configurar FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des_logo, des_frame, k=2)

    # Filtrar matches pero mantener más coincidencias
    good_matches = []
    colors = []
    
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance: 
            good_matches.append(m)
            colors.append((np.random.randint(100, 255), 
                         np.random.randint(100, 255),
                         np.random.randint(100, 255)))

    # Preparar imagen resultado (logo + frame actual)
    h, w = frame.shape[:2]
    result = np.zeros((max(h, logo.shape[0]), w + logo.shape[1], 3), dtype=np.uint8)
    result[:logo.shape[0], :logo.shape[1]] = logo[:, :, :3]  # Logo a la izquierda
    result[:h, logo.shape[1]:logo.shape[1]+w] = frame  # Frame a la derecha

    # Dibujar líneas de conexión entre matches
    if len(good_matches) > 0:
        for i, match in enumerate(good_matches):
            pt1 = tuple(np.round(kp_logo[match.queryIdx].pt).astype(int))
            pt2 = tuple(np.round(kp_frame[match.trainIdx].pt).astype(int) + np.array([logo.shape[1], 0]))
            
            cv2.line(result, pt1, pt2, colors[i], 2)
            cv2.circle(result, pt1, 5, colors[i], -1)
            cv2.circle(result, pt2, 5, colors[i], -1)

    return result

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
