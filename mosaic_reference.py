import cv2
import numpy


face_cascade = cv2.CascadeClassifier("/Users/jinseo/Desktop/코딩/oss_python/Real-time-face-mosaic-processing-with-OpenCV/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0) #노트북 웹캠을 카메라로 사용

cap.set(3,640) # 너비
cap.set(4,480) # 높이

while(cap.isOpened()):
    ret, frame = cap.read() 
    if not ret:
        break
    frame = cv2.flip(frame, 1) # 좌우 대칭
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백 변환

    faces = face_cascade.detectMultiScale(gray,1.05,5) # 얼굴 감지

    centers = [] # 얼굴 감지 후 얼굴 중심점을 계산
    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2
        centers.append((center_x, center_y))
        
    print("Number of faces detected: " + str(len(faces)))

    if len(faces):
        for (x,y,w,h) in faces:
            face_img = frame[y:y+h, x:x+w] # 탐지된 얼굴 이미지 crop
            face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04) # 축소
            face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA) # 확대
            frame[y:y+h, x:x+w] = face_img # 탐지된 얼굴 영역 모자이크 처리

    cv2.imshow('result', frame)
        
    if(cv2.waitKey(25) & 0xFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
