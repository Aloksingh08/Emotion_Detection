
import cv2
import numpy as np
import time
import os
from datetime import datetime
from deepface import DeepFace

# ──────────── CONFIG ────────────
CAMERA_INDEX    = 0      # 0 = default webcam
ANALYZE_EVERY   = 5      # har 5 frames pe AI chalao (speed ke liye)
SCREENSHOT_DIR  = "screenshots"
MIN_FACE_SIZE   = 80     # pixels — chhote faces ignore karo
# ────────────────────────────────

# Har emotion ka rang (BGR) aur display naam
EMOTION_STYLE = {
    "happy":    {"color": (50,  205,  50),  "label": "Happy    😊"},
    "sad":      {"color": (255, 165,   0),  "label": "Sad    😢"},
    "angry":    {"color": (0,     0, 255),  "label": "Angry    😠"},
    "surprise": {"color": (255,   0, 255),  "label": "Surprise   😮"},
    "fear":     {"color": (128,   0, 128),  "label": "Fear     😨"},
    "disgust":  {"color": (0,   128, 128),  "label": "Disgust   🤢"},
    "neutral":  {"color": (180, 180, 180),  "label": "Neutral 😐"},
}

def get_style(emotion):
    return EMOTION_STYLE.get(emotion.lower(),
           {"color": (255, 255, 255), "label": emotion.upper()})

def draw_face(frame, x, y, w, h, emotion, confidence):
    """Face ke upar bounding box aur emotion label draw karo."""
    style  = get_style(emotion)
    color  = style["color"]
    label  = style["label"]
    conf_t = f"{confidence:.0f}%"

    # ── Bounding box ──
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # ── Top bar (label background) ──
    bar_h = 38
    cv2.rectangle(frame, (x, y - bar_h), (x+w, y), color, -1)

    # ── Emotion text ──
    cv2.putText(frame, label,
                (x+6, y - bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)

    # ── Confidence text ──
    cv2.putText(frame, conf_t,
                (x+6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)

    # ── Emotion bar (bottom of face box) ──
    bar_w = int((confidence / 100) * w)
    cv2.rectangle(frame, (x, y+h),     (x+w,     y+h+6), (50,50,50),  -1)
    cv2.rectangle(frame, (x, y+h),     (x+bar_w, y+h+6), color,       -1)

def draw_overlay(frame, fps, face_count, paused):
    """Corner mein FPS, face count aur controls dikhao."""
    h, w = frame.shape[:2]

    # ── Semi-transparent top strip ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 48), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # ── FPS ──
    cv2.putText(frame, f"FPS: {fps:4.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,100), 2, cv2.LINE_AA)

    # ── Face count ──
    cv2.putText(frame, f"Faces: {face_count}",
                (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 2, cv2.LINE_AA)

    # ── PAUSED badge ──
    if paused:
        cv2.putText(frame, "[ PAUSED ]",
                    (w//2 - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,165,255), 2, cv2.LINE_AA)

    # ── Controls (bottom) ──
    hint = "Q/ESC: Quit  |  S: Screenshot  |  P: Pause"
    cv2.putText(frame, hint,
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1, cv2.LINE_AA)

def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"\n[ERROR] Webcam (index {CAMERA_INDEX}) nahi khula.")
        print("        Doosra camera try karo: CAMERA_INDEX = 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "="*52)
    print("  🎭  Live Emotion Detection  —  by DeepFace + OpenCV")
    print("="*52)
    print("  Q / ESC  →  Quit")
    print("  S        →  Screenshot save karo")
    print("  P        →  Pause / Resume")
    print("="*52 + "\n")

    frame_num  = 0
    last_faces = []       # pichli analysis ke results cache
    paused     = False
    prev_time  = time.time()
    display    = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame nahi mila. Camera disconnect hua?")
                break

            frame = cv2.flip(frame, 1)   # mirror effect
            frame_num += 1

            # ── Har ANALYZE_EVERY frames pe DeepFace chalao ──
            if frame_num % ANALYZE_EVERY == 0:
                try:
                    results = DeepFace.analyze(
                        img_path        = frame,
                        actions         = ["emotion"],
                        enforce_detection = False,
                        silent          = True,
                        detector_backend = "opencv",
                    )
                    if not isinstance(results, list):
                        results = [results]

                    last_faces = []
                    for r in results:
                        region = r.get("region", {})
                        w_f = region.get("w", 0)
                        h_f = region.get("h", 0)
                        # Chhote / invalid faces skip karo
                        if w_f < MIN_FACE_SIZE or h_f < MIN_FACE_SIZE:
                            continue
                        last_faces.append({
                            "x": region.get("x", 0),
                            "y": region.get("y", 0),
                            "w": w_f,
                            "h": h_f,
                            "emotion":    r.get("dominant_emotion", "neutral"),
                            "confidence": r.get("emotion", {}).get(
                                              r.get("dominant_emotion","neutral"), 0),
                        })
                except Exception as e:
                    # Koi face nahi mila — silently skip
                    last_faces = []

            # ── Faces draw karo ──
            display = frame.copy()
            for f in last_faces:
                draw_face(display,
                          f["x"], f["y"], f["w"], f["h"],
                          f["emotion"], f["confidence"])

            # ── FPS calculate ──
            now      = time.time()
            fps      = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_overlay(display, fps, len(last_faces), paused)

        elif display is not None:
            # Paused state mein sirf overlay update karo
            draw_overlay(display, 0, len(last_faces), paused)

        if display is not None:
            cv2.imshow("Emotion Detection", display)

        key = cv2.waitKey(1) & 0xFF

        # ── Key controls ──
        if key in (ord("q"), 27):           # Q ya ESC
            print("\n[INFO] Band ho raha hai...")
            break

        elif key == ord("s"):               # Screenshot
            if display is not None:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SCREENSHOT_DIR, f"emotion_{ts}.jpg")
                cv2.imwrite(path, display)
                print(f"[INFO] Screenshot saved → {path}")

        elif key == ord("p"):               # Pause/Resume
            paused = not paused
            print("[INFO]", "Paused ⏸" if paused else "Resumed ▶")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. Phir milenge! 👋\n")

if __name__ == "__main__":
    main()
