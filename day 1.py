import mediapipe as mp
import cv2 as cv

cap = cv.VideoCapture(1)

face_mesh = mp.solutions.face_mesh.FaceMesh()
drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            drawing.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                   landmark_drawing_spec=drawing.DrawingSpec(thickness=1, circle_radius=1),
                                   connection_drawing_spec=drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1))

    cv.imshow("Frame", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
