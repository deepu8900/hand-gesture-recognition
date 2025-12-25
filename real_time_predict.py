import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("hand_gesture_model.h5") 
class_names = ['fist', 'palm', 'thumbsup']

camera = cv2.VideoCapture(0)

x1, y1 = 100, 100
x2, y2 = 300, 300

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)  # shape (1,64,64,3)

   
    predictions = model.predict(roi_input)
    class_index = np.argmax(predictions[0])
    gesture = class_names[class_index]
    confidence = predictions[0][class_index]

    cv2.putText(frame, f"{gesture} ({confidence*100:.1f}%)", 
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Hand Gesture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
