from flask import Flask, render_template
from flask_socketio import SocketIO
from pyngrok import ngrok, conf  # Menggunakan pyngrok secara langsung
import threading
import base64
import numpy as np
from ultralytics import YOLO
from stream_processor import process_stream

app = Flask(__name__)
# Hapus run_with_ngrok(app)
socketio = SocketIO(app)

# --- Variabel Global ---
output_frames = {}
lock = threading.Lock()

# --- Konfigurasi Stream ---
STREAMS_CONFIG = [
    {
        "id": "cctv_1",
        "name": "CCTV Jl. Ahmad Jazuli",
        "url": "https://www.youtube.com/live/ZRnoQXRWboo?si=DJTDHWO4uDHQmed1",
        "poly_right": np.array([[3, 419], [2, 609], [1243, 167], [999, 164], [932, 208], [4, 418]], np.int32),
        "poly_left": np.array([[0, 717], [1097, 713], [619, 419], [0, 671], [1, 712]], np.int32)
    },
    {
        "id": "cctv_2",
        "name": "CCTV Jl. Yos Sudarso",
        "url": "https://www.youtube.com/live/_8Sru9lBzXI?si=6Mf3QUiZRjY6Ud81",
        "poly_right": np.array([[1270, 253], [3, 708], [1271, 706], [1271, 254]], np.int32),
        "poly_left": np.array([[0, 698], [714, 425], [649, 131], [0, 242], [0, 700]], np.int32)
    }
]

# --- Rute Flask ---
@app.route('/')
def index():
    streams_for_template = []
    for stream in STREAMS_CONFIG:
        stream_copy = stream.copy()
        stream_copy['poly_right'] = stream_copy['poly_right'].tolist()
        stream_copy['poly_left'] = stream_copy['poly_left'].tolist()
        streams_for_template.append(stream_copy)
    return render_template('index.html', streams=streams_for_template)

# --- Logika SocketIO ---
@socketio.on('connect')
def handle_connect():
    print('Client terhubung!')

def frame_generator():
    """Mengirim frame dari semua stream ke client secara terus-menerus."""
    while True:
        with lock:
            for stream_id, frame_bytes in output_frames.items():
                if frame_bytes:
                    b64_frame = base64.b64encode(frame_bytes).decode('utf-8')
                    socketio.emit('update_frame', {'id': stream_id, 'image': b64_frame})
        socketio.sleep(0.1)

if __name__ == '__main__':
    # Konfigurasi dan mulai Ngrok secara eksplisit
    conf.get_default().region = "ap"  # Set region Asia Pasifik
    NGROK_AUTHTOKEN = "34BWilIxDYdhkyAO3Mer5f57AHN_5FWUuoPb71jL8s1wWcec6" # <-- GANTI DENGAN TOKEN ANDA
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    public_url = ngrok.connect(5000)
    print(f"✅ Buka dashboard Anda di: {public_url}")

    # Lanjutkan dengan memuat model dan memulai thread
    print("Memuat model YOLOv8...")
    model = YOLO("yolov8n.pt")
    model.fuse()
    print("Model berhasil dimuat.")

    for config in STREAMS_CONFIG:
        thread = threading.Thread(
            target=process_stream,
            args=(config, model, output_frames, lock),
            daemon=True
        )
        thread.start()

    socketio.start_background_task(target=frame_generator)
    
    print("Menjalankan server Flask-SocketIO...")
    socketio.run(app, port=5000, log_output=False) # log_output=False agar tidak terlalu ramai