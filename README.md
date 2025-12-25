# Hand Gesture Recognition (Palm, Fist, Thumbs Up)

A real-time **hand gesture recognition** system using **TensorFlow**, **OpenCV**, and **CNN**.  
This project can detect 3 hand gestures:
- Palm ✋
- Fist ✊
- Thumbs Up 👍

It works with your webcam and shows live predictions with confidence scores.

---

## Features

- Real-time gesture detection
- Uses a simple CNN (from scratch)
- Supports 3 gestures (`palm`, `fist`, `thumbsup`)
- Live ROI (Region of Interest) to focus on the hand
- Easy to add more gestures

---

## Folder Structure

hand-gesture-recognition/
│
├── dataset/ # Images for training
│ ├── palm/
│ ├── fist/
│ ├── thumbsup/
│
├── collect_data.py # Script to collect hand images
├── train_model.py # Script to train CNN model
├── real_time_predict.py # Real-time gesture prediction
├── README.md
├── .gitignore


---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib (for plotting training accuracy)

Install dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib

🚀 Steps to Use
1️⃣ Collect Your Hand Gesture Data

📸 Press S to save images inside the green ROI box

Run the script:

python hand_gesture_all_in_one.py


The script will guide you for each gesture:

Palm ✋

Fist ✊

Thumbs Up 👍

Keep your hand inside the green rectangle

Collect 200+ images per gesture for best results

Press Q to move to the next gesture

⚡ Tip: Move your hand slightly & change angles for better accuracy

2️⃣ Train Your CNN Model

🧠 Train your model from scratch with your collected images

The script automatically trains after data collection

Training progress and accuracy plot will be shown

Model will be saved as:

hand_gesture_model.h5


💡 Tip: Make sure all gestures have roughly same number of images

3️⃣ Real-Time Gesture Prediction

🎥 See your gestures recognized live

After training, the script opens your webcam

Put your hand inside the green ROI box

Watch the predicted gesture + confidence appear above the rectangle

Press Q to quit

✅ Works for Palm, Fist, Thumbs Up
🔮 You can add more gestures by collecting images and retraining

4️⃣ Optional Tips for Best Accuracy

Ensure good lighting

Keep hand fully inside the ROI

Collect more images for each gesture

Move hand slowly at first

5️⃣ Adding New Gestures

Add a new folder in dataset/ with the gesture name

Collect images using the same green ROI box

Update the GESTURES list in the script

Retrain the model

6️⃣ Dependencies

Install required Python packages:

pip install tensorflow opencv-python numpy matplotlib



