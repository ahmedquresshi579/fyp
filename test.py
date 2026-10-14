import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

print("Reported resolution:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Reported FPS:", cap.get(cv2.CAP_PROP_FPS))

frame_count = 0
start_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    elapsed = time.time() - start_time

    if elapsed >= 1.0:
        print(f"Actual measured FPS: {frame_count / elapsed:.1f}")
        frame_count = 0
        start_time = time.time()

    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()