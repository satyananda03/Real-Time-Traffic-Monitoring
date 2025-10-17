import cv2
import numpy as np
import time
from ultralytics import YOLO
from utils import get_youtube_live_url

# --- Konfigurasi Firestore (Akan digunakan nanti) ---
# from google.cloud import firestore
# db = firestore.Client()
# collection_ref = db.collection('vehicle_logs')
# --------------------------------------------------

# Fungsi ini akan dijalankan di setiap thread
def process_stream(stream_config, model, output_frames, lock):
    """
    Memproses satu stream video, melakukan deteksi, dan menyimpan hasilnya.
    """
    stream_id = stream_config['id']
    youtube_url = stream_config['url']
    polygon_left = stream_config['poly_left']
    polygon_right = stream_config['poly_right']
    
    print(f"[{stream_id}] Mencoba mendapatkan URL stream...")
    stream_url = get_youtube_live_url(youtube_url)
    if not stream_url:
        print(f"[{stream_id}] Gagal mendapatkan URL stream. Thread berhenti.")
        return
        
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"[{stream_id}] Gagal membuka stream video. Thread berhenti.")
        return
    print(f"[{stream_id}] Stream berhasil dibuka.")

    class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    
    # --- LOGIKA BARU: Melacak ID yang sedang di dalam poligon ---
    # Ini penting untuk mencatat peristiwa 'masuk' ke Firestore
    ids_inside_right = set()
    ids_inside_left = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{stream_id}] Frame kosong, mencoba menyambung ulang...")
            cap.release()
            time.sleep(5) # Jeda sebelum mencoba lagi
            cap = cv2.VideoCapture(stream_url)
            continue
            
        annotated_frame = frame.copy()
        
        # Gambar poligon
        cv2.polylines(annotated_frame, [polygon_right], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(annotated_frame, [polygon_left], isClosed=True, color=(0, 255, 255), thickness=2)

        # Lakukan deteksi
        results = model.track(frame, classes=[2, 3, 5, 7], conf=0.4, persist=True, verbose=False, tracker="botsort.yaml")

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.cpu().numpy()
            track_ids = boxes.id.astype(int)
            clss = boxes.cls.astype(int)
            xyxy = boxes.xyxy.astype(int)

            current_ids_in_frame = set(track_ids)

            for track_id, cls_id, bbox in zip(track_ids, clss, xyxy):
                bottom_center_point = (int((bbox[0] + bbox[2]) // 2), int(bbox[3]))
                
                # Cek Poligon Kanan
                is_in_right = cv2.pointPolygonTest(polygon_right, bottom_center_point, False) >= 0
                if is_in_right:
                    if track_id not in ids_inside_right:
                        ids_inside_right.add(track_id)
                        # --- DI SINI ANDA AKAN MENGIRIM DATA KE FIRESTORE ---
                        print(f"[{stream_id}] MASUK KANAN: ID {track_id}, Tipe: {class_names.get(cls_id)}")
                        # log_data = {
                        #     "timestamp": firestore.SERVER_TIMESTAMP,
                        #     "location_id": stream_id,
                        #     "vehicle_type": class_names.get(cls_id, "Unknown"),
                        #     "direction": "right"
                        # }
                        # collection_ref.add(log_data)
                        # ----------------------------------------------------
                
                # Cek Poligon Kiri
                is_in_left = cv2.pointPolygonTest(polygon_left, bottom_center_point, False) >= 0
                if is_in_left:
                    if track_id not in ids_inside_left:
                        ids_inside_left.add(track_id)
                        # --- DI SINI ANDA AKAN MENGIRIM DATA KE FIRESTORE ---
                        print(f"[{stream_id}] MASUK KIRI: ID {track_id}, Tipe: {class_names.get(cls_id)}")
                        # log_data = { ... }
                        # collection_ref.add(log_data)
                        # ----------------------------------------------------

                # Gambar bbox dan label
                box_color = (0, 255, 0) if is_in_right or is_in_left else (0, 0, 255)
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
                label = f"{class_names.get(cls_id)} ID:{track_id}"
                cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            # Bersihkan ID yang sudah tidak ada di poligon
            ids_inside_right.intersection_update(current_ids_in_frame)
            ids_inside_left.intersection_update(current_ids_in_frame)

        # Simpan frame yang sudah di-encode untuk dikirim ke web
        with lock:
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            output_frames[stream_id] = buffer.tobytes()