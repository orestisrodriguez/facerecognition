import cv2
import os
import numpy as np
from PIL import Image

persons = ["", "", "", "Ann Veneman", "John Paul II"]

def detect_face(img):
  face_cascade = cv2.CascadeClassifier('recognizers/haarcascade_frontalface_default.xml')

  faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

  return faces

def prepare_training_data(data_folder_path):

  dirs = os.listdir(data_folder_path)

  faces = []
  labels = []

  for dir_name in dirs:
    label = int(dir_name)

    person_dir_path = 'images/' + dir_name

    person_images_names = os.listdir(person_dir_path)

    for image_name in person_images_names:
      image_path = person_dir_path + '/' + image_name

      pil_image = Image.open(image_path).convert('L')
      image = np.array(pil_image, 'uint8')

      cv2.imshow("Training on image..", image)
      cv2.waitKey(100)

      facesLot = detect_face(image)

      for (x, y, w, h) in facesLot:
        faces.append(image[y:y+h,x:x+w])
        labels.append(label)
  
  cv2.destroyAllWindows()
  cv2.waitKey(1)
  cv2.destroyAllWindows()

  return faces, labels

print("Preparing data..")

faces, labels = prepare_training_data('images')

print("Data prepared.")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

face_recognizer.save("recognizers/training.yml")