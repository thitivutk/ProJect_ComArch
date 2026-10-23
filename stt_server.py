from flask import Flask, request, Response, send_from_directory, redirect, abort, jsonify
import logging
import time
import datetime
import os
import re
import json
import queue
import shutil
import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO

HOST = "0.0.0.0"
PORT = 5000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# โฟลเดอร์บันทึกภาพกล้องฉุกเฉิน
ALERT_SAVE_DIR = os.path.join(BASE_DIR, "alert_captures")
os.makedirs(ALERT_SAVE_DIR, exist_ok=True)

# =========================================================
# ตั้งค่า Human Detection (YOLO11)
# =========================================================
# n / s / m / l / x : ยิ่งใหญ่ยิ่งแม่นแต่ช้าลง (ครั้งแรกจะดาวน์โหลดไฟล์โมเดลลงโฟลเดอร์ models/ ให้อัตโนมัติ)
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11l.pt")
PERSON_IMGSZ = 1024        # ใกล้ความกว้างภาพจริง (1024x600) ช่วยให้เจอคนที่อยู่ไกล
PERSON_CONF = 0.40         # ความมั่นใจขั้นต่ำ ลดลง (เช่น 0.25) ถ้าอยากให้นับคนที่เห็นไม่ครบตัว แต่จะนับผิดง่ายขึ้น
PERSON_CLASS_ID = 0        # class "person" ของ COCO
PREDICT_QUANTIZE = 16 if torch.cuda.is_available() else None   # FP16 ใช้ได้เฉพาะบน GPU (แทน half= ที่ Ultralytics เลิกใช้แล้ว)

DETECTED_SUBDIR = "detected"          # ภาพที่วาดกรอบแล้ว แยกจากภาพต้นฉบับ
DETECTIONS_FILE = "detections.json"   # ผลตรวจของแต่ละเฟรมในเหตุการณ์

# ตรวจชื่อจาก header / URL กันการเขียนหรือลบไฟล์นอกโฟลเดอร์ alert_captures (path traversal)
ALERT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
EVENT_NAME_RE = re.compile(r"alert_[A-Za-z0-9_-]{1,64}")
FRAME_TYPES = {"pre", "trigger", "post", "frame"}

person_model = None
detect_queue = queue.Queue()
detections_lock = threading.Lock()   # กันอ่าน/เขียน detections.json ชนกันระหว่าง gallery กับ worker

app = Flask(__name__)

# ปิด log กวนใจของ Werkzeug
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


def load_person_model():
    global person_model
    os.makedirs(os.path.dirname(PERSON_MODEL_PATH), exist_ok=True)
    person_model = YOLO(PERSON_MODEL_PATH)
    # รันเปล่า 1 ครั้งให้ CUDA พร้อมก่อน เฟรมแรกของเหตุการณ์จริงจะได้ไม่ช้า
    person_model.predict(np.zeros((600, 1024, 3), dtype=np.uint8), imgsz=PERSON_IMGSZ,
                         classes=[PERSON_CLASS_ID], quantize=PREDICT_QUANTIZE, verbose=False)


def load_detections(event_dir):
    path = os.path.join(event_dir, DETECTIONS_FILE)
    with detections_lock:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}


def save_detection(event_dir, filename, result):
    """เพิ่มผลตรวจของ 1 เฟรมลง detections.json แล้วคืนผลทั้งหมดของเหตุการณ์"""
    path = os.path.join(event_dir, DETECTIONS_FILE)
    with detections_lock:
        dets = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    dets = json.load(f)
            except (OSError, ValueError):
                dets = {}
        dets[filename] = result
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(dets, f, ensure_ascii=False, indent=1)
        os.replace(tmp_path, path)
        return dets


def draw_person_boxes(img, boxes):
    """วาดกรอบรอบคนแต่ละคน + ป้ายจำนวนคนมุมซ้ายบน"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"person {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        ty = max(y1, th + 8)
        cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 8, ty), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 4, ty - 5), font, 0.6, (0, 0, 0), 2)

    summary = f"People: {len(boxes)}"
    color = (0, 255, 0) if boxes else (180, 180, 180)
    (tw, th), _ = cv2.getTextSize(summary, font, 0.9, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(img, summary, (10, th + 10), font, 0.9, color, 2)


def detect_frame(event_dir, filename, frame_total):
    src = os.path.join(event_dir, filename)
    if not os.path.exists(src):   # เหตุการณ์ถูกลบไประหว่างรอคิว
        return

    # อ่านผ่าน np.fromfile เพราะ cv2.imread เปิด path ที่มีภาษาไทยบน Windows ไม่ได้
    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("อ่านไฟล์ JPEG ไม่ได้ (ไฟล์อาจเสีย)")

    t0 = time.time()
    result = person_model.predict(img, imgsz=PERSON_IMGSZ, conf=PERSON_CONF,
                                  classes=[PERSON_CLASS_ID], quantize=PREDICT_QUANTIZE, verbose=False)[0]
    boxes = [
        (int(x1), int(y1), int(x2), int(y2), float(conf))
        for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
    ]
    elapsed_ms = (time.time() - t0) * 1000

    if not os.path.isdir(event_dir):
        return
    draw_person_boxes(img, boxes)
    det_dir = os.path.join(event_dir, DETECTED_SUBDIR)
    os.makedirs(det_dir, exist_ok=True)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        jpg.tofile(os.path.join(det_dir, filename))

    dets = save_detection(event_dir, filename, {
        "count": len(boxes),
        "boxes": [[x1, y1, x2, y2, round(conf, 3)] for x1, y1, x2, y2, conf in boxes],
    })

    alert_name = os.path.basename(event_dir)
    print(f"[DETECT {alert_name}] {filename}: {len(boxes)} person ({elapsed_ms:.0f} ms)")
    if len(dets) >= frame_total:
        max_people = max(d["count"] for d in dets.values())
        print("================================================================")
        print(f"👤 [DETECT COMPLETE] {alert_name}: พบคนสูงสุด {max_people} คน (จาก {len(dets)} เฟรม)")
        print("================================================================")


def detect_worker():
    # worker ตัวเดียว เพราะ YOLO.predict ไม่ปลอดภัยถ้าหลาย thread เรียกพร้อมกัน
    while True:
        event_dir, filename, frame_total = detect_queue.get()
        try:
            detect_frame(event_dir, filename, frame_total)
        except Exception as e:
            print(f"[DETECT ERROR] {event_dir} / {filename}: {e}")
        finally:
            detect_queue.task_done()


def queue_undetected_frames():
    """ตอนเปิด server: ส่งเฟรมเก่าที่ยังไม่เคยตรวจเข้าคิว"""
    queued = 0
    for ev in sorted(os.listdir(ALERT_SAVE_DIR)):
        ev_dir = os.path.join(ALERT_SAVE_DIR, ev)
        if not EVENT_NAME_RE.fullmatch(ev) or not os.path.isdir(ev_dir):
            continue
        frames = sorted(f for f in os.listdir(ev_dir) if f.endswith(".jpg"))
        dets = load_detections(ev_dir)
        for f in frames:
            if f not in dets:
                detect_queue.put((ev_dir, f, len(frames)))
                queued += 1
    return queued


@app.route("/", methods=["GET"])
def index():
    return redirect("/gallery")


# =====================================================================
# โค้ดส่วนภาพฉุกเฉินและ Gallery (รับภาพ JPEG Full Resolution 1024x600 จาก ESP32-P4)
# =====================================================================

@app.route("/upload_alert_frame", methods=["POST"])
def upload_alert_frame():
    try:
        alert_id = request.headers.get("X-Alert-ID", str(int(time.time())))
        frame_idx = int(request.headers.get("X-Frame-Index", "0"))
        frame_total = int(request.headers.get("X-Frame-Total", "21"))
        frame_type = request.headers.get("X-Frame-Type", "frame")  # "pre", "trigger", "post"
        width = int(request.headers.get("X-Frame-Width", "1024"))
        height = int(request.headers.get("X-Frame-Height", "600"))

        if not ALERT_ID_RE.fullmatch(alert_id):
            return Response("Invalid X-Alert-ID", status=400)
        if frame_type not in FRAME_TYPES:
            return Response("Invalid X-Frame-Type", status=400)
        if frame_idx < 0 or frame_idx >= 1000:
            return Response("Invalid X-Frame-Index", status=400)

        raw_data = request.get_data()
        if not raw_data:
            return Response("Empty frame data", status=400)
        if raw_data[:2] != b"\xff\xd8":   # JPEG ต้องขึ้นต้นด้วย FF D8
            return Response("Frame data is not a JPEG", status=400)

        event_dir = os.path.join(ALERT_SAVE_DIR, f"alert_{alert_id}")
        os.makedirs(event_dir, exist_ok=True)

        meta_file = os.path.join(event_dir, "meta.txt")
        if not os.path.exists(meta_file):
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(meta_file, "w", encoding="utf-8") as f:
                f.write(now_str)

        filename = f"frame_{frame_idx:02d}_{frame_type}.jpg"
        file_path = os.path.join(event_dir, filename)
        # เขียนลงไฟล์ชั่วคราวก่อนแล้วค่อยเปลี่ยนชื่อ หน้า Gallery ที่อัปเดตเรียลไทม์จะได้ไม่ดึงภาพที่เขียนยังไม่เสร็จ
        part_path = file_path + ".part"
        with open(part_path, "wb") as f:
            f.write(raw_data)
        os.replace(part_path, file_path)

        # ส่งเข้าคิวตรวจหาคน (ทำใน background ไม่ให้บอร์ดต้องรอ)
        detect_queue.put((event_dir, filename, frame_total))

        print(f"[ALERT {alert_id}] Saved JPEG frame {frame_idx + 1:02d}/{frame_total} ({frame_type:7s}, {len(raw_data)} bytes) -> {file_path}")
        if frame_idx + 1 == frame_total:
            print("================================================================")
            print(f"🎉 [ALERT COMPLETE] All {frame_total} JPEG frames (1024x600) saved to: {event_dir}/")
            print("================================================================")
        return Response("OK", status=200)
    except Exception as e:
        print(f"[ALERT FRAME ERROR]: {e}")
        return Response(str(e), status=500)


@app.route("/view/<alert_name>/<filename>")
def view_image(alert_name, filename):
    if not EVENT_NAME_RE.fullmatch(alert_name):
        abort(404)
    folder = os.path.join(ALERT_SAVE_DIR, alert_name)
    return send_from_directory(folder, filename, mimetype="image/jpeg")


@app.route("/view/<alert_name>/detected/<filename>")
def view_detected_image(alert_name, filename):
    if not EVENT_NAME_RE.fullmatch(alert_name):
        abort(404)
    folder = os.path.join(ALERT_SAVE_DIR, alert_name, DETECTED_SUBDIR)
    return send_from_directory(folder, filename, mimetype="image/jpeg")


@app.route("/delete_event/<alert_name>", methods=["POST", "GET"])
def delete_event(alert_name):
    if EVENT_NAME_RE.fullmatch(alert_name):
        folder = os.path.join(ALERT_SAVE_DIR, alert_name)
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
            print(f"[ALERT DELETE] Deleted event folder: {folder}")
    # หน้า Gallery เรียกผ่าน fetch (POST) ส่วน GET ยังพากลับหน้า Gallery เหมือนเดิม
    if request.method == "POST":
        return Response("OK", status=200)
    return redirect("/gallery")


@app.route("/delete_all", methods=["POST", "GET"])
def delete_all():
    if os.path.exists(ALERT_SAVE_DIR):
        for item in os.listdir(ALERT_SAVE_DIR):
            item_path = os.path.join(ALERT_SAVE_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
        print("[ALERT DELETE] Deleted all alert events")
    if request.method == "POST":
        return Response("OK", status=200)
    return redirect("/gallery")


def list_events():
    """รายการเหตุการณ์ทั้งหมด (ใหม่สุดก่อน) พร้อมผลตรวจหาคนของแต่ละเฟรม"""
    events = []
    if not os.path.exists(ALERT_SAVE_DIR):
        return events
    for ev in os.listdir(ALERT_SAVE_DIR):
        ev_dir = os.path.join(ALERT_SAVE_DIR, ev)
        if not EVENT_NAME_RE.fullmatch(ev) or not os.path.isdir(ev_dir):
            continue
        frames = sorted([f for f in os.listdir(ev_dir) if f.endswith(".jpg")])
        if not frames:
            continue

        meta_file = os.path.join(ev_dir, "meta.txt")
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                time_str = f.read().strip()
        else:
            mtime = os.path.getmtime(ev_dir)
            time_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

        # ผลตรวจหาคนของแต่ละเฟรม (None = ยังตรวจไม่เสร็จ)
        dets = load_detections(ev_dir)
        people = [dets[f]["count"] if f in dets else None for f in frames]

        events.append({
            "id": ev,
            "time": time_str,
            "frames": frames,
            "people": people,
        })
    # เรียงตามเวลาที่บันทึก (ชื่อ alert_xxx เป็นเลขจากบอร์ด เรียงตามชื่อแล้วไม่ใช่ลำดับเวลา)
    events.sort(key=lambda e: (e["time"], e["id"]), reverse=True)
    return events


@app.route("/api/events")
def api_events():
    return jsonify(list_events())


# หน้า Gallery เป็น HTML คงที่ ข้อมูลทั้งหมดดึงจาก /api/events ทุก 1.5 วินาที แล้วอัปเดตเฉพาะส่วนที่เปลี่ยน
GALLERY_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚨 ESP32-P4 Emergency Alert Gallery (1024x600 HD)</title>
    <style>
        body { font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; background: #0b132b; color: #f8fafc; margin: 0; padding: 25px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #1c2541; padding-bottom: 15px; margin-bottom: 25px; }
        h1 { color: #ff4d4d; margin: 0; font-size: 1.75rem; display: flex; align-items: center; gap: 10px; }
        .btn-group { display: flex; gap: 10px; align-items: center; }
        .btn { padding: 8px 16px; border-radius: 8px; border: none; font-weight: bold; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; gap: 6px; transition: all 0.2s; font-size: 0.9rem; }
        .btn-danger { background: #ef4444; color: white; }
        .btn-danger:hover { background: #dc2626; }
        .btn-play { background: #10b981; color: white; }
        .btn-play:hover { background: #059669; }
        .btn-stop { background: #64748b; color: white; }
        .btn-stop:hover { background: #475569; }
        .btn-box { background: #f59e0b; color: #0b132b; }
        .btn-box:hover { background: #d97706; }

        .live { font-size: 0.85rem; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
        .live-on { color: #4ade80; background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.4); }
        .live-off { color: #f87171; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); }

        .event-card { background: #1c2541; border-radius: 14px; padding: 22px; margin-bottom: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.4); border: 1px solid #3a506b; }
        .event-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; border-bottom: 1px solid #3a506b; padding-bottom: 10px; }
        .event-title { font-size: 1.2rem; font-weight: 700; color: #48cae4; display: flex; align-items: center; gap: 8px; }
        .event-time { font-size: 0.95rem; color: #ffb703; font-weight: 600; background: rgba(255, 183, 3, 0.15); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(255, 183, 3, 0.3); }

        .people-summary { font-size: 0.95rem; font-weight: 700; padding: 4px 12px; border-radius: 20px; }
        .people-found { background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.5); }
        .people-none { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); }
        .people-pending { background: rgba(56, 189, 248, 0.15); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.4); }

        .player-container { text-align: center; background: #0b132b; border-radius: 12px; padding: 15px; border: 1px solid #3a506b; }
        /* แสดงขนาดจริง 1024x600 (ย่อเฉพาะตอนหน้าต่างแคบกว่าภาพ) คลิกเพื่อเปิดภาพในแท็บใหม่ */
        .player-img { display: block; margin: 0 auto; width: 1024px; max-width: 100%; height: auto; aspect-ratio: 1024 / 600; object-fit: contain; border-radius: 8px; background: #000; box-shadow: 0 4px 12px rgba(0,0,0,0.5); cursor: zoom-in; }
        .controls { margin-top: 15px; display: flex; justify-content: center; align-items: center; gap: 15px; flex-wrap: wrap; }
        .slider { width: 260px; max-width: 100%; accent-color: #3a86ff; height: 8px; cursor: pointer; }
        /* ความกว้างคงที่ ข้อความเปลี่ยนระหว่างเล่นแล้วแถวปุ่มจะไม่ขยับไปมา (ไม่กระพริบ) */
        .controls .btn-play { width: 230px; justify-content: center; }
        .controls .btn-box { width: 130px; justify-content: center; }
        .frame-label { font-weight: bold; width: 210px; text-align: left; font-variant-numeric: tabular-nums; white-space: nowrap; }
        .controls .badge { width: 72px; text-align: center; }

        .badge { padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
        .badge-pre { background: #3a86ff; color: white; }
        .badge-trigger { background: #ef4444; color: white; animation: pulse 1s infinite; box-shadow: 0 0 10px rgba(239, 68, 68, 0.8); }
        .badge-post { background: #10b981; color: white; }

        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; margin-top: 15px; }
        .grid-item { background: #0b132b; border-radius: 8px; padding: 6px; text-align: center; font-size: 0.75rem; border: 1px solid #3a506b; cursor: pointer; transition: transform 0.15s; }
        .grid-item:hover { transform: scale(1.05); border-color: #3a86ff; }
        .grid-item img { width: 100%; height: 75px; object-fit: cover; border-radius: 4px; }

        @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.8; transform: scale(1.05); } 100% { opacity: 1; transform: scale(1); } }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🚨 ESP32-P4 Emergency Alert Gallery (1024x600 HD)</h1>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">บันทึกภาพ 10 เฟรมก่อนหน้า + 1 เฟรมขณะตรวจจับ + 10 เฟรมหลังจากได้ยินเสียง <b>"ช่วยด้วย"</b> (Hardware JPEG 1024x600) · ตรวจจับคนด้วย YOLO11</div>
        </div>
        <div class="btn-group">
            <span id="live" class="live live-on">● Live</span>
            <button class="btn btn-danger" onclick="deleteAll()">🗑️ ลบทั้งหมด</button>
        </div>
    </div>

    <div id="empty" style="display: none; text-align: center; padding: 60px 20px; background: #1c2541; border-radius: 12px; color: #94a3b8;">
        <h2>📭 ยังไม่มีภาพบันทึกเหตุการณ์</h2>
        <p>เมื่อพูดคำว่า <b>"ช่วยด้วย"</b> ใส่ไมโครโฟน บอร์ดจะส่งภาพ 21 เฟรม (JPEG 1024x600) มาแสดงที่นี่โดยอัตโนมัติ</p>
    </div>
    <div id="events"></div>

    <script>
        const POLL_MS = 1500;
        const eventData = {};
        const container = document.getElementById('events');
        const emptyEl = document.getElementById('empty');
        const liveEl = document.getElementById('live');

        function esc(s) {
            return String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
        }
        function frameType(f) {
            return f.split('_').pop().replace('.jpg', '');
        }

        // จำนวนคนของทั้งวิดีโอ = จำนวนคนในเฟรมที่เห็นคนมากที่สุด
        function summaryHtml(people) {
            const done = people.filter(p => p !== null);
            const maxPeople = Math.max(0, ...done);
            if (done.length < people.length)
                return `<span class="people-summary people-pending">⏳ กำลังตรวจหาคน ${done.length}/${people.length}</span>`;
            if (maxPeople > 0)
                return `<span class="people-summary people-found" title="นับจากเฟรมที่เห็นคนมากที่สุด">👤 พบคน ${maxPeople} คน</span>`;
            return '<span class="people-summary people-none">ไม่พบคนในภาพ</span>';
        }

        function createCard(ev) {
            const id = ev.id;
            const card = document.createElement('div');
            card.className = 'event-card';
            card.id = 'card_' + id;
            card.innerHTML = `
                <div class="event-header">
                    <div class="event-title">
                        <span>🚨 เหตุการณ์: ${id}</span>
                        <span id="count_${id}" style="font-size: 0.9rem; color: #94a3b8;"></span>
                        <span id="summary_${id}"></span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span class="event-time">⏰ ${esc(ev.time)}</span>
                        <button class="btn btn-danger" style="padding: 4px 10px; font-size: 0.8rem;" onclick="deleteEvent('${id}')">🗑️ ลบ</button>
                    </div>
                </div>
                <div class="player-container">
                    <img id="view_${id}" class="player-img" alt="Loading Frame..." title="คลิกเพื่อเปิดภาพขนาดจริง" onclick="window.open(this.src, '_blank')">
                    <div class="controls">
                        <button class="btn btn-play" id="playbtn_${id}" onclick="playAnim('${id}')">▶ เล่นภาพเคลื่อนไหว (Play)</button>
                        <button class="btn btn-stop" onclick="stopAnim('${id}')">⏹ หยุด (Stop)</button>
                        <button class="btn btn-box" id="boxbtn_${id}" onclick="toggleBoxes('${id}')">🔲 ซ่อนกรอบ</button>
                        <input type="range" min="0" max="0" value="0" class="slider" id="slider_${id}" oninput="updateFrame('${id}', this.value)">
                        <span id="label_${id}" class="frame-label"></span>
                        <span id="badge_${id}" class="badge badge-pre">pre</span>
                    </div>
                </div>
                <details style="margin-top: 15px;">
                    <summary id="gridsum_${id}" style="cursor: pointer; color: #38bdf8; font-weight: 600;"></summary>
                    <div class="grid" id="grid_${id}"></div>
                </details>`;
            eventData[id] = { sig: '', urls: [], detUrls: [], people: [], types: [], showBoxes: true, timer: null, idx: 0, preloaded: {}, playToken: 0 };
            return card;
        }

        // อัปเดตการ์ดเฉพาะเมื่อมีเฟรมใหม่หรือผลตรวจใหม่ ภาพที่ดูอยู่ / การเล่น / กล่องภาพย่อยที่เปิดไว้ยังอยู่เหมือนเดิม
        function updateCard(ev) {
            const id = ev.id;
            const data = eventData[id];
            const sig = JSON.stringify([ev.frames, ev.people]);
            if (data.sig === sig) return;
            data.sig = sig;

            const n = ev.frames.length;
            data.urls = ev.frames.map(f => `/view/${id}/${f}`);
            data.detUrls = ev.frames.map((f, i) => ev.people[i] === null ? null : `/view/${id}/detected/${f}`);
            data.people = ev.people;
            data.types = ev.frames.map(frameType);

            document.getElementById('count_' + id).innerText = `(${n} ภาพ HD)`;
            document.getElementById('summary_' + id).innerHTML = summaryHtml(ev.people);
            document.getElementById('slider_' + id).max = n - 1;
            document.getElementById('gridsum_' + id).innerText = `🔍 ดูภาพย่อยทั้งหมด (${n} ภาพ)`;
            document.getElementById('grid_' + id).innerHTML = ev.frames.map((f, i) => {
                const type = data.types[i];
                const p = ev.people[i];
                const src = data.detUrls[i] || data.urls[i];
                return `
                    <div class="grid-item" onclick="updateFrame('${id}', ${i})">
                        <img src="${src}" loading="lazy">
                        <div style="margin-top: 4px;">#${i + 1} <span class="badge badge-${type}" style="padding: 2px 6px; font-size: 0.65rem;">${type}</span> ${p === null ? '⏳' : '👤 ' + p}</div>
                    </div>`;
            }).join('');
            updateFrame(id, data.idx);
        }

        function render(events) {
            emptyEl.style.display = events.length ? 'none' : '';
            const seen = new Set();
            events.forEach((ev, i) => {
                seen.add(ev.id);
                let card = document.getElementById('card_' + ev.id);
                if (!card) card = createCard(ev);
                if (container.children[i] !== card) container.insertBefore(card, container.children[i] || null);
                updateCard(ev);
            });
            Object.keys(eventData).forEach(id => {
                if (seen.has(id)) return;
                stopAnim(id);
                const card = document.getElementById('card_' + id);
                if (card) card.remove();
                delete eventData[id];
            });
        }

        async function refresh() {
            try {
                const res = await fetch('/api/events', { cache: 'no-store' });
                if (!res.ok) throw new Error('HTTP ' + res.status);
                render(await res.json());
                liveEl.className = 'live live-on';
                liveEl.innerText = '● Live';
            } catch (e) {
                liveEl.className = 'live live-off';
                liveEl.innerText = '● ขาดการเชื่อมต่อ server';
            }
        }
        async function pollLoop() {
            await refresh();
            setTimeout(pollLoop, POLL_MS);
        }

        async function deleteEvent(id) {
            if (!confirm('ยืนยันการลบเหตุการณ์นี้?')) return;
            await fetch('/delete_event/' + id, { method: 'POST' });
            refresh();
        }
        async function deleteAll() {
            if (!confirm('คุณต้องการลบเหตุการณ์ทั้งหมดใช่หรือไม่?')) return;
            await fetch('/delete_all', { method: 'POST' });
            refresh();
        }

        function updateFrame(ev, idx) {
            const data = eventData[ev];
            if (!data || data.urls.length === 0) return;
            data.idx = parseInt(idx);
            if (data.idx < 0) data.idx = 0;
            if (data.idx >= data.urls.length) data.idx = data.urls.length - 1;
            const detUrl = data.detUrls[data.idx];
            const imgEl = document.getElementById('view_' + ev);
            const src = (data.showBoxes && detUrl) ? detUrl : data.urls[data.idx];
            if (imgEl && imgEl.getAttribute('src') !== src) imgEl.src = src;
            const sliderEl = document.getElementById('slider_' + ev);
            if (sliderEl) sliderEl.value = data.idx;
            const labelEl = document.getElementById('label_' + ev);
            if (labelEl) {
                const p = data.people[data.idx];
                const peopleText = (p === null) ? '⏳ กำลังตรวจ' : `👤 ${p} คน`;
                labelEl.innerText = `ภาพที่ ${data.idx + 1} / ${data.urls.length} · ${peopleText}`;
            }
            const badgeEl = document.getElementById('badge_' + ev);
            if (badgeEl) {
                const curType = data.types[data.idx];
                badgeEl.innerText = curType;
                badgeEl.className = `badge badge-${curType}`;
            }
        }
        function toggleBoxes(ev) {
            const data = eventData[ev];
            if (!data) return;
            data.showBoxes = !data.showBoxes;
            const btnEl = document.getElementById('boxbtn_' + ev);
            if (btnEl) btnEl.innerText = data.showBoxes ? '🔲 ซ่อนกรอบ' : '🔲 แสดงกรอบ';
            updateFrame(ev, data.idx);
            if (data.timer) preloadFrames(data);   // กำลังเล่นอยู่ โหลดภาพชุดที่สลับไปให้พร้อม
        }

        // โหลดภาพทุกเฟรมเข้าเบราว์เซอร์ก่อนเล่น ไม่งั้นรอบแรกภาพจะโหลดไม่ทัน 120 ms แล้วถูกข้ามไป
        function preloadFrames(data) {
            const urls = data.urls.map((u, i) => (data.showBoxes && data.detUrls[i]) ? data.detUrls[i] : u);
            return Promise.all(urls.map(url => {
                let im = data.preloaded[url];
                if (!im) {
                    im = new Image();
                    im.src = url;
                    data.preloaded[url] = im;
                }
                return im.decode().catch(() => {});
            }));
        }
        async function playAnim(ev) {
            const data = eventData[ev];
            if (!data || data.urls.length <= 1) return;
            if (data.timer) return;   // กำลังเล่นอยู่แล้ว กดซ้ำไม่ต้องเริ่มใหม่
            stopAnim(ev);
            const token = data.playToken;
            const btnEl = document.getElementById('playbtn_' + ev);
            // ขึ้นข้อความ "กำลังโหลด" เฉพาะตอนโหลดนานจริง ๆ ถ้าภาพอยู่ใน cache แล้วจะได้ไม่กระพริบ
            const loadingTimer = setTimeout(() => {
                if (btnEl && data.playToken === token) btnEl.innerText = '⏳ กำลังโหลดภาพ...';
            }, 200);
            await preloadFrames(data);
            clearTimeout(loadingTimer);
            // ระหว่างรอโหลด ถ้ากด Stop / กด Play ซ้ำ / เหตุการณ์ถูกลบ ก็ไม่ต้องเริ่มเล่น
            if (eventData[ev] !== data || data.playToken !== token) return;
            if (btnEl) btnEl.innerText = '▶ เล่นภาพเคลื่อนไหว (Play)';
            data.timer = setInterval(() => {
                data.idx = (data.idx + 1) % data.urls.length;
                updateFrame(ev, data.idx);
            }, 120);
        }
        function stopAnim(ev) {
            const data = eventData[ev];
            if (!data) return;
            data.playToken++;
            const btnEl = document.getElementById('playbtn_' + ev);
            if (btnEl) btnEl.innerText = '▶ เล่นภาพเคลื่อนไหว (Play)';
            if (data.timer) {
                clearInterval(data.timer);
                data.timer = null;
            }
        }

        pollLoop();
    </script>
</body>
</html>
"""


@app.route("/gallery")
def gallery():
    return GALLERY_HTML


if __name__ == "__main__":
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"⏳ Loading person detection model: {os.path.basename(PERSON_MODEL_PATH)} on {device} ...")
    load_person_model()
    threading.Thread(target=detect_worker, daemon=True).start()
    pending = queue_undetected_frames()

    print("================================================================")
    print(f"  🚨 ESP32-P4 EMERGENCY ALERT & CAMERA SERVER")
    print(f"  Server listening on http://{HOST}:{PORT}")
    print(f"  Gallery URL      : http://localhost:{PORT}/gallery")
    print(f"  Alert Save Dir   : {ALERT_SAVE_DIR}")
    print(f"  Person Detection : {os.path.basename(PERSON_MODEL_PATH)} on {device} (conf >= {PERSON_CONF})")
    if pending:
        print(f"  Old frames queued for detection: {pending}")
    print("================================================================")
    app.run(host=HOST, port=PORT, threaded=True)
