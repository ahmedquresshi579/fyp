import cv2
import time
import sys
import threading

# ============================== SETTINGS ==============================
CAMERA_INDEX = 1
BACKEND = cv2.CAP_MSMF        # try cv2.CAP_DSHOW if MSMF misbehaves
WIDTH, HEIGHT = 1920, 1080    # try 1280x720 if FPS is unstable
TARGET_FPS = 60               # try 30 for a rock-steady rate in dim scenes
USE_MJPG = True

# Exposure: many virtual cameras ignore these, but it is worth testing.
# A shorter exposure = darker image but steadier FPS in dim light.
TRY_MANUAL_EXPOSURE = False
MANUAL_EXPOSURE_VALUE = -6

PREVIEW_SIZE = (960, 540)     # preview only; processing uses the full frame
# =====================================================================


class FYPFrameSynchronizer:
    """Captures frames in a background thread and signals each new frame."""

    def __init__(self, index=CAMERA_INDEX):
        print(f"Connecting to camera on index {index}...")
        self.cap = cv2.VideoCapture(index, BACKEND)

        if USE_MJPG:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera on index {index}")
            sys.exit(1)

        if TRY_MANUAL_EXPOSURE:
            self.set_manual_exposure(MANUAL_EXPOSURE_VALUE)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Driver reports: {w}x{h} @ {fps:.0f} FPS (may not match reality)")

        self.success, self.frame = self.cap.read()
        self.timestamp = time.perf_counter()

        self.running = True
        self.frame_id = 0
        self.capture_fps = 0.0
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
                self.timestamp = now   # capture-time stamp for your processing
                self.frame_id += 1
            self.new_frame.set()       # wake the main loop immediately

            count += 1
            if now - t0 >= 1.0:
                self.capture_fps = count / (now - t0)
                count, t0 = 0, now

    def get_live_frame(self, last_seen_id):
        """Returns (success, frame, frame_id, timestamp) only for a new frame."""
        with self.lock:
            if self.frame_id == last_seen_id:
                return False, None, last_seen_id, 0.0
            return self.success, self.frame, self.frame_id, self.timestamp

    # ----------------------- exposure helpers -----------------------
    def set_manual_exposure(self, value):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual on many drivers
        self.cap.set(cv2.CAP_PROP_EXPOSURE, value)

    def set_auto_exposure(self):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # auto on many drivers

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def main():
    cam = FYPFrameSynchronizer(index=CAMERA_INDEX)

    print("\n--- Camera loop active ---")
    print("Keys:  q = quit | m = toggle manual/auto exposure | e = darker/shorter | r = brighter/longer\n")

    last_id = -1
    frame_count = 0
    proc_fps = 0.0
    last_stamp = None
    worst_gap_ms = 0.0
    start = time.perf_counter()

    while True:
        # Block until a new frame arrives (no sleep polling)
        if not cam.new_frame.wait(timeout=0.1):
            continue
        cam.new_frame.clear()

        success, frame, current_id, stamp = cam.get_live_frame(last_id)
        if not success or frame is None:
            continue
        last_id = current_id

        # Track the largest gap between frames (a jitter indicator)
        if last_stamp is not None:
            worst_gap_ms = max(worst_gap_ms, (stamp - last_stamp) * 1000)
        last_stamp = stamp

        # =======================================================
        # YOUR FYP IMAGE PROCESSING / AI CODE GOES HERE
        # `frame` is the full-resolution BGR image,
        # `stamp` is its capture time (time.perf_counter()).
        # =======================================================

        frame_count += 1
        now = time.perf_counter()
        if now - start >= 1.0:
            proc_fps = frame_count / (now - start)
            print(f"Capture: {cam.capture_fps:5.1f} FPS | Processed: {proc_fps:5.1f} FPS "
                  f"| Worst frame gap: {worst_gap_ms:5.1f} ms")
            frame_count, start, worst_gap_ms = 0, now, 0.0

        # Preview: draw overlay on the small copy only, never on your data
        preview = cv2.resize(frame, PREVIEW_SIZE, interpolation=cv2.INTER_NEAREST)
        cv2.putText(preview, f"Cap {cam.capture_fps:.0f} | Proc {proc_fps:.0f} FPS",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        mode = f"manual exp {cam.exposure}" if cam.manual_mode else "auto exp"
        cv2.putText(preview, mode, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Real-Time FYP Camera Feed", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
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

    cam.close()
    cv2.destroyAllWindows()
    print("Pipeline cleanly terminated.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nStopped by user.")