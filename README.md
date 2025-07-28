# 🧠 Face and Object Detector with LBP and SIFT

This web application allows you to detect and anonymize faces using the **Local Binary Patterns (LBP)** technique, and also performs object detection using the **Scale-Invariant Feature Transform (SIFT)** technique. The system consists of a **backend in Flask with Flask-SocketIO** for real-time processing and streaming, and a frontend that displays the processed images along with performance statistics.

---

## 📁 Project Structure

proyectoFinal/
│
├── app.py                # Main Flask server file
├── modelo/               # Cascade.xml file for LBP
├── static/               # Static resources (CSS, images, logos)
├── templates/            # HTML files for frontend (dashboard.html, sift.html)
└── README.md             # Project documentation

---

## ⚙️ Requirements

- Python 3.8 o superior
- Flask
- Flask-SocketIO
- OpenCV
- NumPy
- eventlet
- psutil

Install the dependencies with:

```bash
pip install flask flask-socketio opencv-python numpy eventlet psutil
```

---

## 💻 Server execution

To start the unified server that serves the dashboard and real-time streaming, run:
```bash
python3 main.py
```


This will start the Flask server at:
```bash
http://0.0.0.0:5000
```
---

## 🖥️ What Does the Application Do?

1. Receives frames captured by the mobile server (which accesses the camera) and sends them via WebSocket.

2. Processes the frames using LBP to detect and pixelate faces, thereby protecting privacy.

3. Processes the frames using SIFT to detect and locate objects by matching them with a reference logo.

4. Streams the processed frames to the frontend for real-time visualization.

5. Sends real-time statistics of FPS and server memory usage for performance monitoring.

---

📄 Frontend Visualization

The frontend displays:

1. Live video with detected and anonymized faces (LBP).

2. Live video with object detection using SIFT.

3. Performance indicators: FPS and memory usage in MB.

---
