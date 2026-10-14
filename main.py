import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import threading
import queue
import sys

from ocr import recognize_text
from text_render import render_clean_text


# ==========================================
# 0. SETTINGS
# ==========================================

# Camera (DroidCam over USB)
CAMERA_INDEX = 1
BACKEND = cv2.CAP_MSMF          # try cv2.CAP_DSHOW if MSMF misbehaves
CAM_W, CAM_H = 1280, 720        # 720p is plenty here and decodes much faster (use 1920, 1080 if you need it)
TARGET_FPS = 60                 # try 30: halves the camera decode load and leaves more CPU for hand tracking
USE_MJPG = True

# Optional manual exposure (many virtual cameras ignore this)
TRY_MANUAL_EXPOSURE = False
MANUAL_EXPOSURE_VALUE = -6

# Mirror the camera image horizontally? (press F while running to toggle)
MIRROR_VIEW = False

# Draw the hand skeleton on the camera view? (press L to toggle; off is a bit faster)
SHOW_LANDMARKS = True

# Whiteboard canvas / UI resolution (16:9, same as the camera, so no distortion)
W, H = 1280, 720

# Hand tracking runs on a smaller copy of the frame for speed
# (landmarks are normalised, so the result is the same on the full canvas)
TRACK_W, TRACK_H = 640, 360

# Hand-tracking confidence. A lower TRACKING value lets MediaPipe keep following the hand
# in harsh light, instead of dropping it and running the slower full re-detection every frame.
# If you ever see false hand detections, raise DETECTION_CONF back toward 0.7.
DETECTION_CONF = 0.6
TRACKING_CONF = 0.5


# ==========================================
# 1. THREADED CAMERA
# ==========================================

class FrameSynchronizer:
    """Reads the camera in a background thread and signals each new frame."""

    def __init__(self, index=CAMERA_INDEX):
        print(f"Connecting to camera on index {index}...")
        self.cap = cv2.VideoCapture(index, BACKEND)

        if USE_MJPG:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        if not self.cap.isOpened():
            print(f"\n[ERROR] Could not open camera at index {index}.")
            print("Check that DroidCam is running and connected.")
            self.cap.release()
            sys.exit(1)

        if TRY_MANUAL_EXPOSURE:
            self.set_manual_exposure(MANUAL_EXPOSURE_VALUE)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Driver reports: {w}x{h} @ {fps:.0f} FPS (may not match reality)")

        self.success = False
        self.frame = None
        self.timestamp = time.perf_counter()
        self.frame_id = 0
        self.capture_fps = 0.0

        self.running = True
        self.lock = threading.Lock()
        self.new_frame = threading.Event()

        self.exposure = MANUAL_EXPOSURE_VALUE
        self.manual_mode = TRY_MANUAL_EXPOSURE

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        count, t0 = 0, time.perf_counter()

        while self.running:
            success, frame = self.cap.read()

            if not success:
                time.sleep(0.001)
                continue

            now = time.perf_counter()

            with self.lock:
                self.success = success
                self.frame = frame
                self.timestamp = now
                self.frame_id += 1

            self.new_frame.set()

            count += 1

            if now - t0 >= 1.0:
                self.capture_fps = count / (now - t0)
                count, t0 = 0, now

    def get_live_frame(self, last_seen_id):
        """Returns (success, frame, frame_id, timestamp) only for a NEW frame."""
        with self.lock:
            if self.frame_id == last_seen_id or self.frame is None:
                return False, None, last_seen_id, 0.0

            return self.success, self.frame, self.frame_id, self.timestamp

    def set_manual_exposure(self, value):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, value)

    def set_auto_exposure(self):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


cam = FrameSynchronizer(CAMERA_INDEX)
print("Camera opened successfully.")


# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,          # lighter/faster hand model (1 is the slower default)
    min_detection_confidence=DETECTION_CONF,
    min_tracking_confidence=TRACKING_CONF
)
mp_draw = mp.solutions.drawing_utils

canvas = np.full((H, W, 3), 255, dtype=np.uint8)

PINCH_ENTER = 40
PINCH_EXIT = 55
MAX_JUMP = 80

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

PANEL_COLOR = (60, 52, 48)
PILL_COLOR = (95, 85, 80)
TEXT_COLOR = (240, 240, 240)
GREY = (110, 110, 110)
WHITE = (255, 255, 255)

COLOR_ITEMS = [
    (370, (25, 25, 25)),
    (425, (68, 68, 239)),
    (480, (94, 197, 34)),
    (535, (246, 130, 59)),
    (600, (255, 255, 255)),
]

THICK_ITEMS = [
    (665, 4),
    (715, 8),
    (765, 16)
]

BAR = (310, 12, 1010, 88)
BAR_CY = 50

VIEW_BTN = (845, 32, 985, 68)
# Moved fully inside the 1280px-wide window (it previously ran off the edge to x=1300)
CONVERT_BTN = (1030, 32, 1260, 68)

MODE_COLORS = {
    "Drawing": (94, 197, 34),
    "Hovering": (0, 215, 255),
    "Erasing": (68, 68, 239)
}

ERASER_SIZE = 40
smoothing = 0.5


# ==========================================
# 3. APPLICATION STATE
# ==========================================

current_color = (25, 25, 25)
current_thickness = 8

eraser_selected = False
show_whiteboard = False

btn_hover_last_frame = False
convert_hover_last_frame = False

ocr_result_queue = queue.Queue()
ocr_running = False
ocr_text_display = ""

clean_notes_image = None
notes_lines = []

prev_x, prev_y = None, None
pinch_state = False

strokes = []
current_stroke = None

# Stroke timestamps use the camera's capture clock (time.perf_counter)
start_time = time.perf_counter()
save_message_until = 0

os.makedirs("samples", exist_ok=True)


# ==========================================
# 4. COMPUTE LOGIC HELPERS
# ==========================================

def pinch_distance(lm, w, h):
    x1, y1 = lm[4].x * w, lm[4].y * h
    x2, y2 = lm[8].x * w, lm[8].y * h

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def new_stroke():
    return {
        "color": list(current_color),
        "thickness": current_thickness,
        "points": []
    }


def log_point(x, y, stamp):
    t = round(stamp - start_time, 3)

    current_stroke["points"].append([x, y, t])


def erase_points(x, y, radius):
    global strokes

    result = []

    for stroke in strokes:
        piece = {
            "color": stroke["color"],
            "thickness": stroke["thickness"],
            "points": []
        }

        for point in stroke["points"]:
            px, py = point[0], point[1]

            if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                if piece["points"]:
                    result.append(piece)

                    piece = {
                        "color": stroke["color"],
                        "thickness": stroke["thickness"],
                        "points": []
                    }
            else:
                piece["points"].append(point)

        if piece["points"]:
            result.append(piece)

    strokes = result


def save_strokes():
    existing = [
        f for f in os.listdir("samples")
        if f.endswith(".png")
    ]

    next_num = len(existing) + 1

    filename = f"samples/sample_{next_num:02d}.png"
    json_filename = f"samples/sample_{next_num:02d}.json"

    cv2.imwrite(filename, canvas)

    with open(json_filename, "w", encoding="utf-8") as file:
        json.dump(
            {
                "canvas_width": W,
                "canvas_height": H,
                "strokes": strokes
            },
            file,
            indent=2
        )

    return filename


def ocr_worker(image):
    try:
        result = recognize_text(image)
        ocr_result_queue.put(("success", result))
    except Exception as exc:
        ocr_result_queue.put(("error", str(exc)))


def start_ocr():
    global ocr_running

    if ocr_running:
        return

    ocr_running = True

    image_copy = canvas.copy()

    threading.Thread(
        target=ocr_worker,
        args=(image_copy,),
        daemon=True
    ).start()


# ==========================================
# 5. UI DRAWING HELPERS
# ==========================================

def rounded_rect(img, x1, y1, x2, y2, r, color):
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)

    corners = [
        (x1 + r, y1 + r),
        (x2 - r, y1 + r),
        (x1 + r, y2 - r),
        (x2 - r, y2 - r)
    ]

    for cx, cy in corners:
        cv2.circle(img, (cx, cy), r, color, -1, AA)


def draw_ui(display):
    # Blend only the toolbar region instead of the whole frame (much cheaper)
    bx1, by1, bx2, by2 = BAR
    roi = display[by1:by2, bx1:bx2]
    overlay = roi.copy()

    rounded_rect(overlay, 0, 0, bx2 - bx1, by2 - by1, 38, PANEL_COLOR)

    display[by1:by2, bx1:bx2] = cv2.addWeighted(overlay, 0.8, roi, 0.2, 0)

    # Color selection
    for cx, col in COLOR_ITEMS:
        cv2.circle(display, (cx, BAR_CY), 17, col, -1, AA)

        if col == WHITE:
            cv2.circle(display, (cx, BAR_CY), 17, (0, 0, 0), 1, AA)

        selected = (
            (col == current_color and not eraser_selected)
            or (col == WHITE and eraser_selected)
        )

        if selected:
            cv2.circle(display, (cx, BAR_CY), 24, WHITE, 2, AA)
        else:
            cv2.circle(display, (cx, BAR_CY), 20, GREY, 1, AA)

    cv2.line(display, (630, 32), (630, 68), GREY, 1, AA)

    # Brush thickness
    for cx, thickness in THICK_ITEMS:
        radius = thickness // 2 + 3

        cv2.circle(display, (cx, BAR_CY), radius, (220, 220, 220), -1, AA)

        if thickness == current_thickness:
            cv2.circle(display, (cx, BAR_CY), 24, WHITE, 2, AA)

    cv2.line(display, (830, 32), (830, 68), GREY, 1, AA)

    # View button
    x1, y1, x2, y2 = VIEW_BTN

    rounded_rect(display, x1, y1, x2, y2, 18, PILL_COLOR)

    label = "View: Notes" if show_whiteboard else "View: Camera"

    size = cv2.getTextSize(label, FONT, 0.5, 1)[0]

    tx = x1 + (x2 - x1 - size[0]) // 2
    ty = y1 + (y2 - y1 + size[1]) // 2

    cv2.putText(display, label, (tx, ty), FONT, 0.5, TEXT_COLOR, 1, AA)


def draw_convert_button(display):
    x1, y1, x2, y2 = CONVERT_BTN

    rounded_rect(display, x1, y1, x2, y2, 18, PILL_COLOR)

    label = "Converting..." if ocr_running else "Convert to Text"

    size = cv2.getTextSize(label, FONT, 0.5, 1)[0]

    tx = x1 + (x2 - x1 - size[0]) // 2
    ty = y1 + (y2 - y1 + size[1]) // 2

    cv2.putText(display, label, (tx, ty), FONT, 0.5, TEXT_COLOR, 1, AA)


def check_convert_button(x, y):
    global convert_hover_last_frame

    x1, y1, x2, y2 = CONVERT_BTN

    hovering = (x1 <= x <= x2 and y1 <= y <= y2)

    if hovering and not convert_hover_last_frame:
        start_ocr()

    convert_hover_last_frame = hovering


def draw_mode_pill(display, text):
    color = MODE_COLORS.get(text, GREY)

    size = cv2.getTextSize(text, FONT, 0.7, 2)[0]

    x1, y1 = 20, H - 70
    x2, y2 = x1 + 38 + size[0] + 18, H - 24

    rounded_rect(display, x1, y1, x2, y2, 23, PANEL_COLOR)

    mid = (y1 + y2) // 2

    cv2.circle(display, (x1 + 22, mid), 7, color, -1, AA)

    cv2.putText(
        display, text, (x1 + 38, mid + size[1] // 2),
        FONT, 0.7, TEXT_COLOR, 2, AA
    )


def draw_crosshair(display, pos, color, filled):
    x, y = pos

    cv2.line(display, (x - 20, y), (x + 20, y), color, 2, AA)
    cv2.line(display, (x, y - 20), (x, y + 20), color, 2, AA)

    cv2.circle(display, (x, y), 6, color, -1 if filled else 2, AA)


# ==========================================
# 6. TOOLBAR INTERACTION
# ==========================================

def in_toolbar(x, y):
    return (
        BAR[0] - 10 <= x <= BAR[2] + 10
        and y <= BAR[3] + 10
    ) or (
        CONVERT_BTN[0] <= x <= CONVERT_BTN[2]
        and CONVERT_BTN[1] <= y <= CONVERT_BTN[3]
    )


def check_toolbar_selection(x, y):
    global current_color
    global current_thickness
    global eraser_selected

    for cx, col in COLOR_ITEMS:
        if (x - cx) ** 2 + (y - BAR_CY) ** 2 <= 28 ** 2:
            if col == WHITE:
                eraser_selected = True
            else:
                eraser_selected = False
                current_color = col

    for cx, thickness in THICK_ITEMS:
        if (x - cx) ** 2 + (y - BAR_CY) ** 2 <= 28 ** 2:
            current_thickness = thickness


def check_view_button(x, y):
    global show_whiteboard
    global btn_hover_last_frame

    x1, y1, x2, y2 = VIEW_BTN

    hovering = (x1 <= x <= x2 and y1 <= y <= y2)

    if hovering and not btn_hover_last_frame:
        show_whiteboard = not show_whiteboard

    btn_hover_last_frame = hovering


def mouse_callback(event, x, y, flags, param):
    global show_whiteboard

    if event == cv2.EVENT_LBUTTONDOWN:
        check_toolbar_selection(x, y)

        x1, y1, x2, y2 = VIEW_BTN

        if x1 <= x <= x2 and y1 <= y <= y2:
            show_whiteboard = not show_whiteboard

        cx1, cy1, cx2, cy2 = CONVERT_BTN

        if cx1 <= x <= cx2 and cy1 <= y <= cy2:
            start_ocr()


# ==========================================
# 7. WINDOW INITIALIZATION
# ==========================================

cv2.namedWindow("Whiteboard", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Whiteboard", mouse_callback)

print("\n--- AI Whiteboard Pipeline Active ---")
print("Keys: Q quit | C clear | S save | F flip/mirror | L hand skeleton | M toggle manual exposure | E darker | R brighter\n")

last_frame_id = -1
frame_count = 0
last_fps_time = time.perf_counter()

# Per-stage timing (seconds, summed over the current 1-second window)
stage = {"prep": 0.0, "mediapipe": 0.0, "compose": 0.0, "ui+show": 0.0}
stage_n = 0


# ==========================================
# 8. MAIN INTERACTION LOOP
# ==========================================

try:
    while True:
        # Block until the camera thread delivers a new frame (no sleep polling).
        # On timeout, keep the window responsive and still allow quitting.
        if not cam.new_frame.wait(timeout=0.1):
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        cam.new_frame.clear()

        success, raw_frame, frame_id, stamp = cam.get_live_frame(last_frame_id)

        if not success or raw_frame is None:
            continue

        last_frame_id = frame_id

        t0 = time.perf_counter()

        # Resize only if the camera isn't already delivering W x H
        if raw_frame.shape[1] != W or raw_frame.shape[0] != H:
            frame = cv2.resize(raw_frame, (W, H), interpolation=cv2.INTER_AREA)
        else:
            frame = raw_frame.copy()

        # Flipping happens before hand tracking, so the cursor stays aligned either way.
        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)

        # Hand tracking on a smaller copy
        tracking_frame = cv2.resize(frame, (TRACK_W, TRACK_H), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        t1 = time.perf_counter()
        results = hands.process(rgb_frame)
        t2 = time.perf_counter()

        mode_text = ""
        cursor_pos = None
        cursor_color = (150, 150, 150)

        pen_down = False
        inking_now = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                # Pinch detection with hysteresis
                dist = pinch_distance(lm, W, H)

                if pinch_state:
                    pinching = dist < PINCH_EXIT
                else:
                    pinching = dist < PINCH_ENTER

                pinch_state = pinching

                # Cursor position: midpoint between thumb and index tips
                raw_x = int((lm[4].x + lm[8].x) / 2 * W)
                raw_y = int((lm[4].y + lm[8].y) / 2 * H)

                if prev_x is None:
                    prev_x, prev_y = raw_x, raw_y

                # Smooth cursor movement
                smooth_x = int(prev_x * smoothing + raw_x * (1 - smoothing))
                smooth_y = int(prev_y * smoothing + raw_y * (1 - smoothing))

                # Prevent sudden cursor jumps
                jump_dist = (
                    (smooth_x - prev_x) ** 2
                    + (smooth_y - prev_y) ** 2
                ) ** 0.5

                if jump_dist > MAX_JUMP:
                    smooth_x, smooth_y = prev_x, prev_y

                cursor_pos = (smooth_x, smooth_y)

                if pinching:
                    pen_down = True

                    if in_toolbar(smooth_x, smooth_y):
                        check_toolbar_selection(smooth_x, smooth_y)
                        check_view_button(smooth_x, smooth_y)
                        check_convert_button(smooth_x, smooth_y)

                        mode_text = "Hovering"

                    else:
                        if eraser_selected:
                            cv2.line(
                                canvas,
                                (prev_x, prev_y),
                                (smooth_x, smooth_y),
                                WHITE,
                                ERASER_SIZE
                            )

                            erase_radius = ERASER_SIZE // 2 + 5

                            erase_points(smooth_x, smooth_y, erase_radius)

                            erase_points(
                                (prev_x + smooth_x) // 2,
                                (prev_y + smooth_y) // 2,
                                erase_radius
                            )

                            mode_text = "Erasing"

                        else:
                            cv2.line(
                                canvas,
                                (prev_x, prev_y),
                                (smooth_x, smooth_y),
                                current_color,
                                current_thickness,
                                AA
                            )

                            inking_now = True

                            if current_stroke is None:
                                current_stroke = new_stroke()

                            log_point(prev_x, prev_y, stamp)
                            log_point(smooth_x, smooth_y, stamp)

                            mode_text = "Drawing"

                else:
                    mode_text = "Hovering"

                cursor_color = MODE_COLORS.get(mode_text, (150, 150, 150))

                prev_x, prev_y = smooth_x, smooth_y

                if SHOW_LANDMARKS:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

        else:
            prev_x, prev_y = None, None
            pinch_state = False
            btn_hover_last_frame = False
            convert_hover_last_frame = False

        # Finalize stroke when drawing stops
        if not inking_now and current_stroke is not None:
            if current_stroke["points"]:
                strokes.append(current_stroke)

            current_stroke = None

        # Process OCR results
        if not ocr_result_queue.empty():
            status, result = ocr_result_queue.get()

            if status == "success":
                if result and len(result) == 2:
                    raw, corrected = result

                    if corrected:
                        notes_lines.append(corrected)
                        ocr_text_display = corrected

                        clean_notes_image = render_clean_text(
                            "\n".join(notes_lines),
                            W,
                            H
                        )

                        # Clear the handwriting after successful conversion
                        canvas[:] = 255
                        strokes = []
                        current_stroke = None

                else:
                    print("[OCR] Unexpected result format:", result)

            else:
                print("[OCR ERROR]", result)

            ocr_running = False

        # Ink mask = every pixel that is not pure white (fast OpenCV version)
        ink_mask = cv2.bitwise_not(
            cv2.inRange(canvas, (255, 255, 255), (255, 255, 255))
        )

        # Choose display mode
        if show_whiteboard:
            if clean_notes_image is not None:
                display = clean_notes_image.copy()
                display = cv2.copyTo(canvas, ink_mask, display)
            else:
                display = canvas.copy()

        else:
            frame = cv2.copyTo(canvas, ink_mask, frame)
            display = frame

        t3 = time.perf_counter()

        # Draw interface
        draw_ui(display)
        draw_convert_button(display)

        if mode_text:
            draw_mode_pill(display, mode_text)

        if ocr_text_display:
            cv2.putText(
                display,
                f"Text: {ocr_text_display}",
                (20, H - 100),
                FONT,
                0.8,
                (0, 0, 0),
                2,
                AA
            )

        # FPS calculation
        frame_count += 1

        now = time.perf_counter()
        elapsed_fps = now - last_fps_time

        if elapsed_fps >= 1.0:
            measured_fps = frame_count / elapsed_fps

            n = max(stage_n, 1)

            print(
                f"Capture: {cam.capture_fps:5.1f} FPS | "
                f"Processed: {measured_fps:5.1f} FPS | "
                f"ms/frame -> prep {stage['prep'] / n * 1000:4.1f} | "
                f"mediapipe {stage['mediapipe'] / n * 1000:4.1f} | "
                f"compose {stage['compose'] / n * 1000:4.1f} | "
                f"ui+show {stage['ui+show'] / n * 1000:4.1f}"
            )

            for k in stage:
                stage[k] = 0.0

            stage_n = 0

            frame_count = 0
            last_fps_time = now

        # Status information
        exposure_info = (
            f"exp {cam.exposure}" if cam.manual_mode else "auto exp"
        )

        info = (
            f"Strokes: {len(strokes)}   {exposure_info}"
            "   S: save   C: clear   Q: quit"
        )

        cv2.putText(display, info, (W - 520, H - 30), FONT, 0.5, GREY, 1, AA)

        if time.time() < save_message_until:
            cv2.putText(
                display, "Saved!", (W - 400, H - 60),
                FONT, 0.6, (94, 197, 34), 2, AA
            )

        if cursor_pos:
            draw_crosshair(display, cursor_pos, cursor_color, filled=pen_down)

        cv2.imshow("Whiteboard", display)

        key = cv2.waitKey(1) & 0xFF

        stage["prep"] += t1 - t0
        stage["mediapipe"] += t2 - t1
        stage["compose"] += t3 - t2
        stage["ui+show"] += time.perf_counter() - t3
        stage_n += 1

        if key == ord("q"):
            break

        elif key == ord("c"):
            canvas[:] = 255
            strokes = []
            current_stroke = None

            clean_notes_image = None
            ocr_text_display = ""
            notes_lines = []

        elif key == ord("s"):
            saved_path = save_strokes()
            print(f"Saved whiteboard: {saved_path}")

            save_message_until = time.time() + 2

        elif key == ord("f"):
            MIRROR_VIEW = not MIRROR_VIEW

        elif key == ord("l"):
            SHOW_LANDMARKS = not SHOW_LANDMARKS

        elif key == ord("m"):
            cam.manual_mode = not cam.manual_mode

            if cam.manual_mode:
                cam.set_manual_exposure(cam.exposure)
            else:
                cam.set_auto_exposure()

        elif key == ord("e") and cam.manual_mode:
            cam.exposure -= 1
            cam.set_manual_exposure(cam.exposure)

        elif key == ord("r") and cam.manual_mode:
            cam.exposure += 1
            cam.set_manual_exposure(cam.exposure)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    cam.close()
    hands.close()
    cv2.destroyAllWindows()
    print("Whiteboard closed safely.")