import cv2

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("recognizers/haarcascade_frontalface_default.xml")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("./recognizers/training.yml")

persons = ["", "", "", "Ann Veneman", "John Paul II"]

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        region = gray[y:y+h, x:x+w]

        id_, confidence = face_recognizer.predict(region)

        label = persons[id_] + " " + str(confidence)
        cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
