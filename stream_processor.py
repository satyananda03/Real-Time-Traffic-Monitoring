import cv2
import numpy as np
import time
from ultralytics import YOLO
from utils import get_youtube_live_url

# Fungsi sekarang menerima 'vehicle_counts' sebagai argumen
def process_stream(stream_config, model, output_frames, vehicle_counts, lock):
    stream_id = stream_config['id']
    youtube_url = stream_config['url']
    polygon_left = stream_config['poly_left']
    polygon_right = stream_config['poly_right']
    
    # ... (tidak ada perubahan pada bagian inisialisasi stream) ...
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
    
    # Set ini sekarang hanya untuk melacak apa yang ada di dalam frame,
    # bukan untuk total hitungan.
    ids_inside_right = set()
    ids_inside_left = set()

    while True:
        # ... (tidak ada perubahan pada pembacaan frame) ...
        ret, frame = cap.read()
        if not ret:
            print(f"[{stream_id}] Frame kosong, mencoba menyambung ulang...")
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            continue
            
        annotated_frame = frame.copy()
        
        cv2.polylines(annotated_frame, [polygon_right], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(annotated_frame, [polygon_left], isClosed=True, color=(0, 255, 255), thickness=2)

        results = model.track(frame, classes=[2, 3, 5, 7], conf=0.4, persist=True, verbose=False, tracker="botsort.yaml")

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.cpu().numpy()
            track_ids = boxes.id.astype(int)
            clss = boxes.cls.astype(int)
            xyxy = boxes.xyxy.astype(int)
            current_ids_in_frame = set(track_ids)

            for track_id, cls_id, bbox in zip(track_ids, clss, xyxy):
                bottom_center_point = (int((bbox[0] + bbox[2]) // 2), int(bbox[3]))
                class_name = class_names.get(cls_id, "Unknown")
                
                # Cek Poligon Kanan
                is_in_right = cv2.pointPolygonTest(polygon_right, bottom_center_point, False) >= 0
                if is_in_right and track_id not in ids_inside_right:
                    ids_inside_right.add(track_id)
                    # --- PERBARUI DATA HITUNGAN ---
                    with lock:
                        if class_name != "Unknown":
                            vehicle_counts[stream_id]['right'][class_name] += 1
                        vehicle_counts[stream_id]['right']['total'] += 1
                
                # Cek Poligon Kiri
                is_in_left = cv2.pointPolygonTest(polygon_left, bottom_center_point, False) >= 0
                if is_in_left and track_id not in ids_inside_left:
                    ids_inside_left.add(track_id)
                    # --- PERBARUI DATA HITUNGAN ---
                    with lock:
                        if class_name != "Unknown":
                            vehicle_counts[stream_id]['left'][class_name] += 1
                        vehicle_counts[stream_id]['left']['total'] += 1

                # Gambar bbox dan label (tidak ada perubahan di sini)
                box_color = (0, 255, 0) if is_in_right or is_in_left else (0, 0, 255)
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
                label = f"{class_name} ID:{track_id}"
                cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            # Bersihkan ID yang sudah keluar dari frame
            ids_inside_right.intersection_update(current_ids_in_frame)
            ids_inside_left.intersection_update(current_ids_in_frame)
        
        # ... (tidak ada perubahan pada blok optimasi) ...
        target_width = 854
        h, w, _ = annotated_frame.shape
        if w > target_width:
            ratio = target_width / w
            target_height = int(h * ratio)
            frame_to_send = cv2.resize(annotated_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            frame_to_send = annotated_frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]

        with lock:
            _, buffer = cv2.imencode('.jpg', frame_to_send, encode_param)
            output_frames[stream_id] = buffer.tobytes()