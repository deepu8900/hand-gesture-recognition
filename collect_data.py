import cv2
import os

gesture_name = "fist"
save_path = f"dataset/{gesture_name}"
os.makedirs(save_path, exist_ok=True)

camera = cv2.VideoCapture(0)
img_count = 0

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x1, y1 = 100, 100
    x2, y2 = 300, 300

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))

    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(1) & 0xFF  

    if key == ord("q"):
        break

    if key == ord("s"):
        img_path = f"{save_path}/{img_count}.jpg"
        cv2.imwrite(img_path, roi)
        img_count += 1
        print(f"Saved {img_path}")

camera.release()
cv2.destroyAllWindows()
