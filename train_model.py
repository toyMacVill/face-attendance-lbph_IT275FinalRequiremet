import cv2
import os
import numpy as np
from PIL import Image

dataset_path = "dataset"

faces = []
labels = []

label_map = {}

current_id = 0

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

for person_name in os.listdir(dataset_path):

    label_map[current_id] = person_name

    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        pil_img = Image.open(image_path).convert("L")

        img_numpy = np.array(pil_img, "uint8")

        detected_faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in detected_faces:

            faces.append(img_numpy[y:y+h, x:x+w])

            labels.append(current_id)

    current_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, np.array(labels))

os.makedirs("trainer", exist_ok=True)

recognizer.save("trainer/trainer.yml")

print("Training Complete!")
print(label_map)

# SAVE LABEL MAP

np.save("trainer/labels.npy", label_map)
