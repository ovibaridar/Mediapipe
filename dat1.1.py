import mediapipe as mp
import cv2 as cv

cap = cv.VideoCapture(1)

face_detection = mp.solutions.face_detection.FaceDetection()
drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(frame, bbox, (0, 255, 0), 2)

    cv.imshow("Frame", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
