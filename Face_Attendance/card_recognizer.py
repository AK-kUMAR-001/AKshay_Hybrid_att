import os
import json
import re
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CARD_DIR = os.path.join(DATASET_DIR, "card_images")
CARD_MAP_FILE = os.getenv("CARD_ID_MAP", os.path.join(BASE_DIR, "card_id_map.json"))

DEFAULT_MIN_MATCHES = 80
DEFAULT_ORB_FEATURES = 1500
DEFAULT_MIN_MATCH_RATIO = 0.08
DEFAULT_OCR_INTERVAL = 0.0
DEFAULT_OCR_ROI = "0.0,0.45,1.0,0.98"
DEFAULT_OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

try:
    import pytesseract
    HAS_TESS = True
except Exception:
    HAS_TESS = False

if HAS_TESS:
    tess_cmd = os.getenv("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd


def _parse_label(label):
    # Supports: 103_akshay, akshay_103, id_001, 001
    parts = label.split("_")
    if len(parts) == 1:
        if parts[0].isdigit():
            return parts[0], parts[0]
        return parts[0], parts[0]

    if parts[0].lower() == "id" and len(parts) > 1:
        return parts[1], parts[1]

    if parts[0].isdigit():
        return parts[0], "_".join(parts[1:])

    if parts[-1].isdigit():
        return parts[-1], "_".join(parts[:-1])

    return label, label


class CardRecognizer:
    def __init__(self):
        self.min_matches = int(os.getenv("CARD_MIN_MATCHES", DEFAULT_MIN_MATCHES))
        self.min_match_ratio = float(os.getenv("CARD_MIN_MATCH_RATIO", DEFAULT_MIN_MATCH_RATIO))
        self.n_features = int(os.getenv("CARD_ORB_FEATURES", DEFAULT_ORB_FEATURES))
        self.ocr_interval = float(os.getenv("CARD_OCR_INTERVAL", DEFAULT_OCR_INTERVAL))
        self.ocr_roi = os.getenv("CARD_OCR_ROI", DEFAULT_OCR_ROI)
        self.ocr_whitelist = os.getenv("CARD_OCR_WHITELIST", DEFAULT_OCR_WHITELIST)
        self.orb = cv2.ORB_create(nfeatures=self.n_features)
        self.templates = []
        self.card_map = {}
        self._last_ocr_time = 0.0
        self._load_templates()
        self._load_card_map()

    def _load_templates(self):
        if not os.path.exists(CARD_DIR):
            print(f"Card images folder not found: {CARD_DIR}")
            return

        for filename in os.listdir(CARD_DIR):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(CARD_DIR, filename)
            image = cv2.imread(path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            if descriptors is None:
                continue

            label = os.path.splitext(filename)[0]
            student_id, name = _parse_label(label)

            self.templates.append({
                "id": student_id,
                "name": name,
                "label": label,
                "kp": keypoints,
                "des": descriptors,
            })

        print(f"Loaded {len(self.templates)} ID card templates.")

    def _load_card_map(self):
        if not os.path.exists(CARD_MAP_FILE):
            print(f"Card map not found: {CARD_MAP_FILE}")
            return
        try:
            with open(CARD_MAP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                key = re.sub(r"\s+", "", str(k)).upper()
                if isinstance(v, dict):
                    self.card_map[key] = {
                        "id": str(v.get("id", key)),
                        "name": str(v.get("name", key))
                    }
            print(f"Loaded {len(self.card_map)} card OCR mappings.")
        except Exception as e:
            print(f"Failed to load card map: {e}")

    def _ocr_match(self, frame_bgr):
        if not HAS_TESS or not self.card_map:
            return None

        now = cv2.getTickCount() / cv2.getTickFrequency()
        if self.ocr_interval > 0 and (now - self._last_ocr_time) < self.ocr_interval:
            return None
        self._last_ocr_time = now

        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            h0, w0 = gray.shape[:2]
            try:
                x0, y0, x1, y1 = [float(x) for x in self.ocr_roi.split(",")]
            except Exception:
                x0, y0, x1, y1 = 0.0, 0.45, 1.0, 0.98
            x0 = max(0, min(1, x0))
            y0 = max(0, min(1, y0))
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            rx0, ry0 = int(x0 * w0), int(y0 * h0)
            rx1, ry1 = int(x1 * w0), int(y1 * h0)
            if rx1 > rx0 and ry1 > ry0:
                gray = gray[ry0:ry1, rx0:rx1]
            h, w = gray.shape[:2]
            scale = 1000 / max(1, w)
            if scale > 1:
                gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

            gray = cv2.bilateralFilter(gray, 5, 75, 75)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 5)

            config = f"--oem 3 --psm 6 -c tessedit_char_whitelist={self.ocr_whitelist}"
            text = pytesseract.image_to_string(th, config=config)
            text = re.sub(r"\s+", "", text).upper()

            for key, info in self.card_map.items():
                if key in text:
                    return info["name"], info["id"], key

            tokens = re.findall(r"[A-Z0-9]{6,}", text)
            for t in tokens:
                if t in self.card_map:
                    info = self.card_map[t]
                    return info["name"], info["id"], t
        except Exception:
            return None

        return None

    def recognize_card(self, frame_bgr):
        ocr_hit = self._ocr_match(frame_bgr)
        if ocr_hit:
            name, sid, key = ocr_hit
            return name, sid, 999

        if not self.templates:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        best = None
        best_score = 0
        best_ratio = 0.0

        for tpl in self.templates:
            try:
                matches = bf.knnMatch(tpl["des"], des, k=2)
            except cv2.error:
                continue

            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            score = len(good)
            ratio = score / max(1, len(tpl["kp"]))

            if score > best_score:
                best_score = score
                best_ratio = ratio
                best = tpl

        if best and best_score >= self.min_matches and best_ratio >= self.min_match_ratio:
            return best["name"], best["id"], best_score

        return None
