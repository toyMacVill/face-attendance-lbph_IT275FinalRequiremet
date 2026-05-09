import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import os

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Face Recognition Attendance System",
    layout="wide"
)

st.title("Face Recognition Attendance System")

# =====================================
# CREATE FOLDERS
# =====================================

os.makedirs("dataset", exist_ok=True)
os.makedirs("trainer", exist_ok=True)
os.makedirs("attendance", exist_ok=True)

# =====================================
# LOAD FACE CASCADE
# =====================================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# =====================================
# SIDEBAR
# =====================================

menu = st.sidebar.selectbox(
    "Select Feature",
    [
        "Train New Face",
        "Recognize from Image",
        "Live Webcam Recognition",
        "Attendance Records"
    ]
)

# =====================================
# TRAIN NEW FACE
# =====================================

if menu == "Train New Face":

    st.header("Train New Face")

    person_name = st.text_input("Enter Person Name")

    uploaded_files = st.file_uploader(
        "Upload Face Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Start Training"):

        if person_name == "":
            st.warning("Please enter a name.")

        elif not uploaded_files:
            st.warning("Please upload images.")

        else:

            person_folder = os.path.join(
                "dataset",
                person_name
            )

            os.makedirs(person_folder, exist_ok=True)

            # SAVE IMAGES

            for idx, uploaded_file in enumerate(uploaded_files):

                image = Image.open(uploaded_file)

                save_path = os.path.join(
                    person_folder,
                    f"{idx}.jpg"
                )

                image.save(save_path)

            st.success("Images Uploaded!")

            # =====================================
            # TRAIN MODEL
            # =====================================

            faces = []
            labels = []

            label_map = {}

            current_id = 0

            for name in os.listdir("dataset"):

                label_map[current_id] = name

                person_dir = os.path.join(
                    "dataset",
                    name
                )

                for image_name in os.listdir(person_dir):

                    image_path = os.path.join(
                        person_dir,
                        image_name
                    )

                    pil_img = Image.open(
                        image_path
                    ).convert("L")

                    img_numpy = np.array(
                        pil_img,
                        "uint8"
                    )

                    detected_faces = face_cascade.detectMultiScale(
                        img_numpy
                    )

                    for (x, y, w, h) in detected_faces:

                        faces.append(
                            img_numpy[y:y+h, x:x+w]
                        )

                        labels.append(current_id)

                current_id += 1

            recognizer = cv2.face.LBPHFaceRecognizer_create()

            recognizer.train(
                faces,
                np.array(labels)
            )

            recognizer.save("trainer/trainer.yml")

            np.save(
                "trainer/labels.npy",
                label_map
            )

            st.success("Training Complete!")

# =====================================
# IMAGE RECOGNITION
# =====================================

elif menu == "Recognize from Image":

    st.header("Recognize Face from Image")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read("trainer/trainer.yml")

        label_map = np.load(
            "trainer/labels.npy",
            allow_pickle=True
        ).item()

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            id, confidence = recognizer.predict(
                gray[y:y+h, x:x+w]
            )

            if confidence < 70:
                name = label_map[id]
            else:
                name = "Unknown"

            cv2.rectangle(
                image_np,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image_np,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        st.image(image_np)

# =====================================
# LIVE WEBCAM RECOGNITION
# =====================================

elif menu == "Live Webcam Recognition":

    st.header("Live Webcam Face Recognition")

    start = st.button("Start Webcam")

    FRAME_WINDOW = st.image([])

    if start:

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read("trainer/trainer.yml")

        label_map = np.load(
            "trainer/labels.npy",
            allow_pickle=True
        ).item()

        cap = cv2.VideoCapture(0)

        while True:

            ret, frame = cap.read()

            if not ret:
                st.error("Failed to access webcam.")
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5
            )

            for (x, y, w, h) in faces:

                id, confidence = recognizer.predict(
                    gray[y:y+h, x:x+w]
                )

                if confidence < 70:

                    name = label_map[id]

                    # =========================
                    # MARK ATTENDANCE
                    # =========================

                    file_path = "attendance/attendance.csv"

                    now = datetime.now()

                    dt = now.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.DataFrame(
                            columns=["Name", "DateTime"]
                        )

                    if name not in df["Name"].values:

                        new_data = {
                            "Name": name,
                            "DateTime": dt
                        }

                        df = pd.concat(
                            [df, pd.DataFrame([new_data])],
                            ignore_index=True
                        )

                        df.to_csv(
                            file_path,
                            index=False
                        )

                else:
                    name = "Unknown"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    name,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            FRAME_WINDOW.image(frame)

# =====================================
# ATTENDANCE RECORDS
# =====================================

elif menu == "Attendance Records":

    st.header("Attendance Records")

    attendance_file = "attendance/attendance.csv"

    if os.path.exists(attendance_file):

        df = pd.read_csv(attendance_file)

        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            "Download Attendance CSV",
            csv,
            "attendance.csv",
            "text/csv"
        )

    else:
        st.info("No attendance records yet.")
