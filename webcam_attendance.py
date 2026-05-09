import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ----------------------------
# LOAD FACE DETECTOR
# ----------------------------

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ----------------------------
# LOAD TRAINED MODEL
# ----------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("trainer/trainer.yml")

# ----------------------------
# LOAD LABELS
# ----------------------------

label_map = np.load(
    "trainer/labels.npy",
    allow_pickle=True
).item()

# ----------------------------
# ATTENDANCE FUNCTION
# ----------------------------

def mark_attendance(name):

    os.makedirs("attendance", exist_ok=True)

    file_path = "attendance/attendance.csv"

    now = datetime.now()

    dt = now.strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Name", "DateTime"])

    # PREVENT DUPLICATE ENTRY

    if name not in df["Name"].values:

        new_data = {
            "Name": name,
            "DateTime": dt
        }

        df = pd.concat(
            [df, pd.DataFrame([new_data])],
            ignore_index=True
        )

        df.to_csv(file_path, index=False)

        print(f"{name} marked present.")

# ----------------------------
# START WEBCAM
# ----------------------------

cap = cv2.VideoCapture(0)

print("Starting Webcam Attendance System...")
print("Press Q to Exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        # LOWER CONFIDENCE = BETTER MATCH

        if confidence < 70:

            name = label_map[id]

            color = (0, 255, 0)

            mark_attendance(name)

            text = f"{name} ({round(confidence,2)})"

        else:

            name = "Unknown"

            color = (0, 0, 255)

            text = "Unknown"

        # DRAW RECTANGLE

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        # DRAW LABEL

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow(
        "Live Face Recognition Attendance",
        frame
    )

    # PRESS Q TO EXIT

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# CLEANUP
# ----------------------------

cap.release()

cv2.destroyAllWindows()
