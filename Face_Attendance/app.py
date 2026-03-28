from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify, session
import cv2
import face_recognition
import numpy as np
from face_recognizer import FaceRecognizer
from face_recognizer_pt import FaceRecognizerPT
from card_recognizer import CardRecognizer
from database import mark_attendance, init_db, get_attendance_records, generate_session_report
import os
import time
from datetime import datetime
from face_encoder import generate_encodings
from twilio.rest import Client
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import base64
import io

app = Flask(__name__)
app.secret_key = "secret_key_here"

# Initialize DB
init_db()

# Initialize Recognizer
recognizer = FaceRecognizer()
recognizer_pt = FaceRecognizerPT()
card_recognizer = CardRecognizer()

# Global state for UI hold logic: {student_id: {'until': timestamp, 'name': name}}
display_state = {}

# Constants
HOLD_DURATION = 3  # Seconds to hold the green frame
COOLDOWN_DURATION = 5 # Seconds before re-detecting same person (starts after hold)
FACE_JITTERS = int(os.getenv("FACE_JITTERS", "2"))
FACE_SCALES = os.getenv("FACE_SCALES", "0.5,0.75,1.0")
USE_PT = os.getenv("USE_PT", "0") == "1"
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))

# --- Configuration ---
# Twilio WhatsApp (Replace with your SID and Token from Twilio Console)
TWILIO_SID =  "ACefad1359c2910c3d38d72185d5567496"
TWILIO_AUTH_TOKEN = "1cfabf9fc5cceb13d3afe6cf2d39ad23"
TWILIO_FROM = "whatsapp:+14155238886"
TWILIO_TO = "whatsapp:+91919791005567" # Staff number

# Email Configuration (Gmail)
# NOTE: For Gmail, use an App Password if 2FA is enabled, or allow "Less Secure Apps"
EMAIL_SENDER = "akshayprabhu19012005@gmail.com"
EMAIL_PASSWORD = "qixo ixhb txqf qvtq"  # App Password
EMAIL_RECEIVER = "akshayprabhu19012005@gmail.com" # Sending to self by default

import json

# Global Camera Management
video_capture = None
is_registering = False
is_attendance_active = False
camera_paused = False
recognition_mode = "both"  # "face", "card", "both"
session_start_time = None
current_session_name = "Session"
marked_students = set()
ALLOWED_IDS = {s.strip() for s in os.getenv("ALLOWED_IDS", "100,103").split(",") if s.strip()}
SESSION_FILE = "session_state.json"
last_camera_probe = 0
last_card_check = 0
card_message = {"until": 0, "text": "", "ok": False}
card_hold_until = 0
card_confirm = {"id": None, "count": 0, "last_time": 0}
CARD_CHECK_INTERVAL = 0.5
CARD_CONFIRM_HITS = 2
CARD_CONFIRM_WINDOW = 2.0

# Payment / QR scanning
QR_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qr_id_map.json")
payment_camera_active = False
payment_last_qr = {"id": None, "name": None, "raw": None, "time": 0}
payment_message = ""
payment_qr_detector = cv2.QRCodeDetector()
BARCODE_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "barcode_id_map.json")
barcode_map = {}
barcode_message = {"until": 0, "text": "", "ok": False}
barcode_hold_until = 0
BARCODE_CONFIRM = {"val": None, "count": 0, "last_time": 0}
BARCODE_CONFIRM_HITS = 2
BARCODE_CONFIRM_WINDOW = 2.0

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except Exception:
    HAS_PYZBAR = False

def decode_barcodes(frame_bgr):
    if not HAS_PYZBAR:
        return []
    symbols = None
    try:
        symbols = [pyzbar.ZBarSymbol.CODE128, pyzbar.ZBarSymbol.CODE39, pyzbar.ZBarSymbol.CODE93]
    except Exception:
        symbols = None

    # Try OpenCV barcode detector first (from opencv-contrib)
    try:
        if hasattr(cv2, "barcode"):
            det = cv2.barcode_BarcodeDetector()
            ok, decoded, _, _ = det.detectAndDecode(frame_bgr)
            if ok and decoded:
                return [type("B", (), {"data": d.encode("utf-8"), "rect": (0, 0, 0, 0)}) for d in decoded]
    except Exception:
        pass

    # Try raw frame (pyzbar)
    out = pyzbar.decode(frame_bgr, symbols=symbols) if symbols else pyzbar.decode(frame_bgr)
    if out:
        return out

    # Preprocess for better detection
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    x0 = int(w * 0.1); x1 = int(w * 0.9)
    y0 = int(h * 0.2); y1 = int(h * 0.8)
    roi = gray[y0:y1, x0:x1]

    variants = []
    for img in (gray, roi):
        img = cv2.resize(img, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        variants.append(img)
        variants.append(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5))
        variants.append(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        variants.append(cv2.bitwise_not(variants[-1]))

    for v in variants:
        out = pyzbar.decode(v, symbols=symbols) if symbols else pyzbar.decode(v)
        if out:
            return out
    return []

def get_camera_indices():
    # Allow override: CAMERA_INDEX="0" or "0,1,2"
    env = os.getenv("CAMERA_INDEX")
    if env:
        indices = []
        for part in env.split(","):
            part = part.strip()
            if part.isdigit():
                indices.append(int(part))
        if indices:
            return indices
    return [0, 1, 2, 3]

def get_camera_backends():
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("DSHOW", cv2.CAP_DSHOW))
    return backends

def save_session_state():
    state = {
        "is_active": is_attendance_active,
        "start_time": session_start_time.isoformat() if session_start_time else None,
        "session_name": current_session_name
    }
    with open(SESSION_FILE, 'w') as f:
        json.dump(state, f)

def load_session_state():
    global is_attendance_active, session_start_time, current_session_name
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                state = json.load(f)
                is_attendance_active = state.get("is_active", False)
                start_time_str = state.get("start_time")
                if start_time_str:
                    session_start_time = datetime.fromisoformat(start_time_str)
                else:
                    session_start_time = None
                current_session_name = state.get("session_name", "Session")
        except Exception as e:
            print(f"Error loading session state: {e}")

def load_qr_map():
    data = {}
    try:
        if os.path.exists(QR_MAP_FILE):
            with open(QR_MAP_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                key = str(k).strip()
                if isinstance(v, dict):
                    data[key] = {
                        "id": str(v.get("id", key)),
                        "name": str(v.get("name", key)),
                        "password": str(v.get("password", ""))
                    }
    except Exception as e:
        print(f"Error loading QR map: {e}")
    return data

qr_map = load_qr_map()

def load_barcode_map():
    data = {}
    try:
        if os.path.exists(BARCODE_MAP_FILE):
            with open(BARCODE_MAP_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                key = str(k).strip().upper()
                if isinstance(v, dict):
                    data[key] = {
                        "id": str(v.get("id", key)),
                        "name": str(v.get("name", key))
                    }
    except Exception as e:
        print(f"Error loading barcode map: {e}")
    return data

barcode_map = load_barcode_map()

# Attendance success event (for auto-advance)
attendance_event = {"id": None, "name": None, "source": None, "time": 0}

def set_attendance_event(student_id, name, source):
    global attendance_event
    if not student_id or str(student_id).lower() == "none":
        return
    attendance_event = {
        "id": str(student_id),
        "name": str(name),
        "source": source,
        "time": time.time()
    }

# Load state on startup
load_session_state()

def send_email_report(file_path, summary):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"Attendance Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        body = f"Please find the attached attendance report for the recent session.\n\nSummary:\n{summary}"
        msg.attach(MIMEText(body, 'plain'))
        
        # Attachment
        if file_path and os.path.exists(file_path):
            attachment = open(file_path, "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= " + os.path.basename(file_path))
            msg.attach(part)
            attachment.close()
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, text)
        server.quit()
        return True, "Email sent successfully."
    except Exception as e:
        print(f"Email Error: {e}")
        return False, str(e)

def get_camera():
    global video_capture
    global last_camera_probe
    if video_capture is None or not video_capture.isOpened():
        now = time.time()
        if now - last_camera_probe < 0.2:
            return None
        last_camera_probe = now

        indices = get_camera_indices()
        backends = get_camera_backends()
        for idx in indices:
            for label, backend in backends:
                if backend is None:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, backend)

                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        video_capture = cap
                        try:
                            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        except Exception:
                            pass
                        print(f"Camera initialized successfully (Index {idx}, {label}).")
                        return video_capture
                cap.release()

        print("CRITICAL: No working camera found on any index.")
        video_capture = None
            
    return video_capture

def release_camera():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    cv2.destroyAllWindows()

def generate_frames():
    global is_registering, is_attendance_active, last_card_check, card_message, card_hold_until, card_confirm, camera_paused, recognition_mode
    global barcode_message, barcode_hold_until, barcode_map, BARCODE_CONFIRM
    
    frame_idx = 0
    while True:
        if camera_paused:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Paused", (170, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
        if is_registering:
            time.sleep(0.5)
            continue
            
        if not is_attendance_active:
            # KEEP CAMERA OPEN but show "Attendance Stopped"
            # This prevents the DSHOW/MSMF crash when toggling on/off repeatedly
            
            camera = get_camera()
            if camera and camera.isOpened():
                success, frame = camera.read()
                if success:
                    # Darken the frame to indicate inactivity
                    frame = cv2.addWeighted(frame, 0.3, np.zeros(frame.shape, frame.dtype), 0, 0)
                    cv2.putText(frame, "Attendance Stopped", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, "Click Start to Resume", (170, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Camera failed to read, try resetting
                    release_camera()
            else:
                # Fallback if camera completely dead
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Disconnected", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.05) # Low FPS when inactive to save CPU
            continue
            
        camera = get_camera()
        if camera is None or not camera.isOpened():
            print("Waiting for camera...")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Not Available", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        try:
            success, frame = camera.read()
        except cv2.error as e:
            print(f"Camera read error: {e}. Resetting...")
            release_camera()
            time.sleep(0.2)
            continue
        if not success:
            print("Failed to read frame from camera. Resetting...")
            release_camera()
            time.sleep(0.2)
            continue
            
        current_time = time.time()
        
        # Check if we are in a "HOLD" state for any student
        active_hold = None
        for sid, state in list(display_state.items()):
            if current_time < state['until']:
                active_hold = state
                break
            else:
                # Hold expired
                del display_state[sid]

        if active_hold:
            # === HOLD STATE ===
            # Use 0.5 scale for better long-range detection (more pixels = better detection)
            scale_factor = 0.5
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # If faces found, assume the largest one is our guy
            if face_locations:
                 # Just use the first face for simplicity in hold mode
                top, right, bottom, left = face_locations[0]
                
                # Scale up
                inv_scale = int(1/scale_factor)
                top *= inv_scale
                right *= inv_scale
                bottom *= inv_scale
                left *= inv_scale
                
                color = (0, 255, 0) # Green
                name = active_hold['name']
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, f"{name} - Mark Success", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                
                # Add "Verified" text on screen center
                cv2.putText(frame, "VERIFIED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            # === NORMAL RECOGNITION STATE ===
            frame_idx += 1
            do_heavy = (frame_idx % FRAME_SKIP == 0)
            scales = []
            try:
                scales = [float(s.strip()) for s in FACE_SCALES.split(",") if s.strip()]
            except Exception:
                scales = [0.5, 0.75, 1.0]

            face_locations = []
            face_encodings = []
            face_names = []
            scale_factor = None

            if do_heavy:
                for s in scales:
                    scale_factor = s
                    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=FACE_JITTERS)
                        break
            
            if recognition_mode in ("face", "both") and face_locations:
                for i, face_encoding in enumerate(face_encodings):
                    # 1. Dlib Recognition
                    name_dlib, id_dlib = recognizer.recognize_face(face_encoding)

                    # 2. PyTorch FaceNet Recognition (optional for speed)
                    name_pt, id_pt, dist_pt = "Unknown", None, 999.0
                    if USE_PT:
                        top, right, bottom, left = face_locations[i]
                        face_img = rgb_small_frame[top:bottom, left:right]
                        name_pt, id_pt, dist_pt = recognizer_pt.recognize_face(face_img)

                    # 3. Double Verification Logic
                    name = "Unknown"
                    student_id = None

                    if name_dlib != "Unknown" and name_pt != "Unknown":
                        if id_dlib == id_pt:
                            # Both models agree -> HIGH CONFIDENCE
                            name = name_dlib
                            student_id = id_dlib
                            print(f"Verified: {name} (Dlib & PT agree)")
                        else:
                            # Models disagree -> Conflict -> Treat as Unknown
                            print(f"Conflict: Dlib said {name_dlib}, PT said {name_pt}")
                    elif name_dlib != "Unknown":
                        # Accept Dlib-only match
                        name = name_dlib
                        student_id = id_dlib
                        print(f"Dlib Only: {name_dlib} (PT Unsure: {dist_pt:.2f})")
                    elif name_pt != "Unknown":
                        # Accept PT-only match
                        name = name_pt
                        student_id = id_pt
                        print(f"PT Only: {name_pt} (Dlib Unsure)")

                    if name != "Unknown" and student_id and str(student_id) in ALLOWED_IDS:
                        display_name = f"{name} ({student_id})"

                        # Mark attendance
                        if student_id not in marked_students:
                            mark_attendance(student_id, name)
                            marked_students.add(student_id)
                            set_attendance_event(student_id, name, "face")

                            # Trigger HOLD logic
                            display_state[student_id] = {
                                'until': current_time + HOLD_DURATION,
                                'name': name
                            }
                        else:
                            pass
                    else:
                        display_name = "Unknown"

                    face_names.append(display_name)

            # --- ID Card Recognition (either face OR card can mark attendance) ---
            if recognition_mode in ("card", "both") and card_recognizer is not None:
                if current_time - last_card_check >= CARD_CHECK_INTERVAL:
                    last_card_check = current_time
                    card_match = card_recognizer.recognize_card(frame)
                    if card_match:
                        name_card, id_card, score_card = card_match

                        # Confirm same card seen multiple times to reduce false positives
                        if card_confirm["id"] == id_card and (current_time - card_confirm["last_time"]) <= CARD_CONFIRM_WINDOW:
                            card_confirm["count"] += 1
                        else:
                            card_confirm = {"id": id_card, "count": 1, "last_time": current_time}

                        card_confirm["last_time"] = current_time

                        if card_confirm["count"] >= CARD_CONFIRM_HITS:
                            if str(id_card) in ALLOWED_IDS and id_card not in marked_students:
                                mark_attendance(id_card, name_card)
                                marked_students.add(id_card)
                                set_attendance_event(id_card, name_card, "card")

                            card_message = {
                                "until": current_time + HOLD_DURATION,
                                "text": f"CARD VERIFIED: {name_card} ({id_card})",
                                "ok": True
                            }
                            card_hold_until = current_time + HOLD_DURATION

            # --- Barcode Recognition (attendance) ---
            if recognition_mode in ("face", "card", "both") and HAS_PYZBAR and barcode_map:
                barcodes = decode_barcodes(frame)
                for b in barcodes:
                    try:
                        val = b.data.decode("utf-8").strip().upper()
                    except Exception:
                        continue
                    # draw box for debug
                    try:
                        (x, y, w, h) = b.rect
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                        cv2.putText(frame, val, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    except Exception:
                        pass
                    if val in barcode_map:
                        # confirm seen twice
                        if BARCODE_CONFIRM["val"] == val and (current_time - BARCODE_CONFIRM["last_time"]) <= BARCODE_CONFIRM_WINDOW:
                            BARCODE_CONFIRM["count"] += 1
                        else:
                            BARCODE_CONFIRM = {"val": val, "count": 1, "last_time": current_time}

                        BARCODE_CONFIRM["last_time"] = current_time
                        if BARCODE_CONFIRM["count"] >= BARCODE_CONFIRM_HITS:
                            name_bar = barcode_map[val]["name"]
                            id_bar = barcode_map[val]["id"]
                            if str(id_bar) in ALLOWED_IDS and id_bar not in marked_students:
                                mark_attendance(id_bar, name_bar)
                                marked_students.add(id_bar)
                                set_attendance_event(id_bar, name_bar, "barcode")
                            barcode_message = {
                                "until": current_time + HOLD_DURATION,
                                "text": f"BARCODE VERIFIED: {name_bar} ({id_bar})",
                                "ok": True
                            }
                            barcode_hold_until = current_time + HOLD_DURATION
                            break
                
            # Draw results
            if scale_factor:
                inv_scale = int(1/scale_factor)
            else:
                inv_scale = 1
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up
                top *= inv_scale
                right *= inv_scale
                bottom *= inv_scale
                left *= inv_scale

                color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Show card message banner if active
        if card_message and current_time < card_message.get("until", 0):
            color = (0, 255, 0) if card_message.get("ok") else (0, 200, 255)
            cv2.putText(frame, card_message["text"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if current_time < card_hold_until:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 0), 3)

        if barcode_message and current_time < barcode_message.get("until", 0):
            color = (0, 255, 0) if barcode_message.get("ok") else (0, 200, 255)
            cv2.putText(frame, barcode_message["text"], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if current_time < barcode_hold_until:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (0, 255, 0), 2)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html', is_active=is_attendance_active, session_name=current_session_name, camera_paused=camera_paused, recognition_mode=recognition_mode)

@app.route('/start_attendance', methods=['GET', 'POST'])
def start_attendance():
    global is_attendance_active, session_start_time, current_session_name, marked_students, camera_paused
    
    if request.method == 'POST':
        current_session_name = request.form.get('session_name', 'Session')
    
    is_attendance_active = True
    session_start_time = datetime.now()
    marked_students.clear() # Reset for new session
    save_session_state() # Save state
    
    flash(f"{current_session_name} Started at {session_start_time.strftime('%H:%M:%S')}")
    return redirect(url_for('attendance', reset=0))

@app.route('/stop_attendance')
def stop_attendance():
    global is_attendance_active, session_start_time, current_session_name
    is_attendance_active = False
    save_session_state() # Update state to inactive
    
    msg = f"{current_session_name} Stopped."
    
    # Generate Report if session was active
    if session_start_time:
        # Pass marked_students to filter only unique attendees for this session
        # Or generate_session_report handles it. Let's inspect generate_session_report.
        # Actually, if marked_students prevented duplicates in DB, we are good.
        # But marked_students only prevents re-marking in THIS runtime.
        # If generate_session_report queries DB by time, it might pick up multiple if app restarted?
        # But we want ONE entry per person per session.
        
        file_path, summary, attendees = generate_session_report(session_start_time, current_session_name)
        if file_path:
            msg += f" {summary}"
            
            # Auto-send Email with Excel Attachment
            email_success, email_status = send_email_report(file_path, summary)
            
            if email_success:
                msg += " [Email Sent]"
            else:
                print(f"Email Failed: {email_status}") # Log full error to console
                msg += " [Email Failed]" # Show simple message to user

            # Auto-send via Twilio (Text Summary)
            try:
                client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
                
                # Construct message
                whatsapp_msg = f"?? {current_session_name} Report\n"
                whatsapp_msg += f"?? {datetime.now().strftime('%Y-%m-%d')}\n"
                whatsapp_msg += f"? {session_start_time.strftime('%H:%M')} - {datetime.now().strftime('%H:%M')}\n\n"
                
                if attendees:
                    whatsapp_msg += "? Present:\n"
                    for student in attendees:
                        # student = (name, id, time)
                        whatsapp_msg += f"? {student[0]} ({student[1]}) - {student[2]}\n"
                else:
                    whatsapp_msg += "No students detected.\n"
                
                whatsapp_msg += "\n(Excel file saved locally)"

                message = client.messages.create(
                    from_=TWILIO_FROM,
                    body=whatsapp_msg,
                    to=TWILIO_TO
                )
                msg += f" (WhatsApp sent: {message.sid})"
            except Exception as e:
                print(f"Twilio Auto-Send Error: {e}")
                msg += " (WhatsApp failed - Check console)"
        else:
            msg += " No records found in this session."
            
    session_start_time = None # Reset
    flash(msg)
    return redirect(url_for('index'))

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global recognition_mode
    if mode in ("face", "card", "both"):
        recognition_mode = mode
        flash(f"Mode set to: {mode.upper()}")
    else:
        flash("Invalid mode.")
    return redirect(url_for('index'))

@app.route('/stop_camera')
def stop_camera():
    global camera_paused
    camera_paused = True
    release_camera()
    flash("Camera stopped.")
    return redirect(request.referrer or url_for('attendance'))

@app.route('/start_camera')
def start_camera():
    global camera_paused
    camera_paused = False
    get_camera()
    flash("Camera started.")
    return redirect(request.referrer or url_for('attendance'))

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_payment_frames():
    global payment_camera_active, payment_last_qr, payment_message, qr_map

    while True:
        if not payment_camera_active:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Payment Camera Stopped", (90, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        camera = get_camera()
        if camera is None or not camera.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Not Found", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        success, frame = camera.read()
        if not success:
            release_camera()
            time.sleep(0.1)
            continue

        data, points, _ = payment_qr_detector.detectAndDecode(frame)
        if data:
            key = data.strip()
            if key in qr_map:
                payment_last_qr = {
                    "id": qr_map[key]["id"],
                    "name": qr_map[key]["name"],
                    "raw": key,
                    "time": time.time()
                }
                payment_message = f"QR detected: {qr_map[key]['name']} ({qr_map[key]['id']})"
            else:
                payment_message = f"Unknown QR: {key}"

        if payment_message:
            cv2.putText(frame, payment_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/payment')
def payment():
    session.pop("payment_amount", None)
    session.pop("payment_id", None)
    session.pop("payment_name", None)
    global payment_camera_active, payment_message, payment_last_qr
    keep_camera = session.pop("payment_keep_camera", False)
    if not keep_camera:
        payment_camera_active = False
        payment_message = ""
        payment_last_qr = {"id": None, "name": None, "raw": None, "time": 0}
        release_camera()
    return render_template('payment_scan.html', camera_active=payment_camera_active)

@app.route('/payment_video')
def payment_video():
    return Response(generate_payment_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/payment_start_camera')
def payment_start_camera():
    global payment_camera_active, payment_message, payment_last_qr, qr_map
    payment_camera_active = True
    payment_message = ""
    payment_last_qr = {"id": None, "name": None, "raw": None, "time": 0}
    qr_map = load_qr_map()
    session["payment_keep_camera"] = True
    flash("Payment camera started.")
    return redirect(url_for('payment'))

@app.route('/payment_stop_camera')
def payment_stop_camera():
    global payment_camera_active
    payment_camera_active = False
    session["payment_keep_camera"] = False
    global payment_message, payment_last_qr
    payment_message = ""
    payment_last_qr = {"id": None, "name": None, "raw": None, "time": 0}
    release_camera()
    flash("Payment camera stopped.")
    return redirect(url_for('payment'))

@app.route('/payment_reset')
def payment_reset():
    session.pop("payment_amount", None)
    session.pop("payment_id", None)
    session.pop("payment_name", None)
    global payment_camera_active, payment_message, payment_last_qr
    payment_camera_active = False
    payment_message = ""
    payment_last_qr = {"id": None, "name": None, "raw": None, "time": 0}
    release_camera()
    return redirect(url_for('payment'))

@app.route('/qr_status')
def qr_status():
    return jsonify(payment_last_qr)

@app.route('/payment_amount')
def payment_amount():
    sid = request.args.get("sid", "").strip()
    name = request.args.get("name", "").strip()
    if not sid:
        # fallback to last qr
        sid = payment_last_qr.get("id") or ""
        name = payment_last_qr.get("name") or ""
    session["payment_id"] = sid
    session["payment_name"] = name
    return render_template('payment_amount.html', student_id=sid, name=name)

@app.route('/payment_password', methods=['POST'])
def payment_password():
    amount = request.form.get("amount", "").strip()
    if not amount:
        flash("Please select an amount.")
        return redirect(url_for('payment_amount'))
    session["payment_amount"] = amount
    return render_template('payment_password.html',
                           student_id=session.get("payment_id", ""),
                           name=session.get("payment_name", ""),
                           amount=amount)

@app.route('/payment_confirm', methods=['POST'])
def payment_confirm():
    password = request.form.get('password', '').strip()
    sid = session.get("payment_id", "")
    name = session.get("payment_name", "")
    amount = session.get("payment_amount", "")

    if not sid or not amount:
        flash("Payment failed: missing ID or amount.")
        return redirect(url_for('payment'))

    user = qr_map.get(sid)
    required_password = user.get("password") if user else ""
    if password != required_password:
        flash("Payment failed: invalid password.")
        return redirect(url_for('payment_password'))

    return redirect(url_for('payment_success'))

@app.route('/payment_success')
def payment_success():
    student_id = session.get("payment_id", "")
    name = session.get("payment_name", "")
    amount = session.get("payment_amount", "")
    session.pop("payment_amount", None)
    session.pop("payment_id", None)
    session.pop("payment_name", None)
    return render_template('payment_success.html',
                           student_id=student_id,
                           name=name,
                           amount=amount)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        global is_registering
        student_id = request.form['student_id']
        name = request.form['name']
        
        folder_name = f"{student_id}_{name}"
        # Use absolute path for dataset to avoid CWD issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, "dataset")
        folder_path = os.path.join(dataset_dir, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # Capture images using global camera
        is_registering = True
        time.sleep(1) # Wait for generator to pause
        
        cap = get_camera()
        count = 0
        while count < 20:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            cv2.imwrite(os.path.join(folder_path, f"{count}.jpg"), frame)
            count += 1
            time.sleep(0.2)
        
        is_registering = False
        
        flash(f"Registered {name} successfully! Captured {count} images.")
        return redirect(url_for('index'))
        
    return render_template('register.html')

@app.route('/send_report')
def send_report():
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        
        # Get today's attendance summary
        records = get_attendance_records()
        today_str = time.strftime("%Y-%m-%d")
        today_records = [r for r in records if r[3] == today_str]
        
        msg_body = f"Attendance Report for {today_str}:\n"
        if not today_records:
            msg_body += "No attendance marked today."
        else:
            for r in today_records:
                msg_body += f"- {r[2]} ({r[1]}) at {r[4]}\n"
        
        message = client.messages.create(
            from_=TWILIO_FROM,
            body=msg_body,
            to=TWILIO_TO
        )
        flash(f"Report sent via WhatsApp! SID: {message.sid}")
    except Exception as e:
        flash("Failed to send report. Check console for details.")
        print(f"Twilio Error: {e}")
        
    return redirect(url_for('attendance'))

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        generate_encodings()
        # Reload recognizer
        global recognizer
        recognizer = FaceRecognizer()
        flash("Model trained successfully!")
        return redirect(url_for('index'))
    return render_template('train.html')

@app.route('/attendance')
def attendance():
    global is_attendance_active, camera_paused, marked_students, attendance_event
    # Reset to fresh state when entering attendance page (unless reset=0)
    if request.args.get("reset", "1") == "1":
        is_attendance_active = False
        camera_paused = False
        marked_students.clear()
        attendance_event = {"id": None, "name": None, "source": None, "time": 0}
    records = get_attendance_records()
    return render_template('attendance.html',
                           records=records,
                           is_active=is_attendance_active,
                           camera_paused=camera_paused,
                           recognition_mode=recognition_mode,
                           session_name=current_session_name)

@app.route('/attendance_event')
def attendance_event_api():
    # Return and do not clear; success page will clear
    if not is_attendance_active:
        return jsonify({"id": None, "name": None, "source": None, "time": 0})
    if attendance_event.get("id") in (None, "", "None"):
        return jsonify({"id": None, "name": None, "source": None, "time": 0})
    return jsonify(attendance_event)

@app.route('/attendance_success')
def attendance_success():
    global attendance_event
    data = attendance_event.copy()
    attendance_event = {"id": None, "name": None, "source": None, "time": 0}
    return render_template('attendance_success.html',
                           student_id=data.get("id"),
                           name=data.get("name"),
                           source=data.get("source"))

if __name__ == "__main__":
    release_camera() # Ensure clean state on startup
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
