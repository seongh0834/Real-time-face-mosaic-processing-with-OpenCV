import cv2

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 얼굴 감지용 분류기 로드
face_cascade = cv2.CascadeClassifier("C:\\opensource\\haarcascade_frontalface_default.xml")

while cap.isOpened():
    ret, frame = cap.read()

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scaleFactor 및 minSize 조정
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minSize=(30, 30))

    print("Number of faces detected: " + str(len(faces)))

    if len(faces):
        # 제일 큰 얼굴은 모자이크를 적용하지 않음
        max_face = max(faces, key=lambda face: face[2] * face[3])
        for (x, y, w, h) in faces:
            if (x, y, w, h) != tuple(max_face):
                face_img = frame[y:y + h, x:x + w]  # 탐지된 얼굴 이미지 crop
                face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04)  # 축소
                face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)  # 확대
                frame[y:y + h, x:x + w] = face_img  # 탐지된 얼굴 영역 모자이크 처리

    # 결과 표시
    cv2.imshow('result', frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# 웹캠 해제
cap.release()
cv2.destroyAllWindows()
