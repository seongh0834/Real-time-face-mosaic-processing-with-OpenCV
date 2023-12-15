# Real-time-face-mosaic-processing-with-OpenCV
---
### When multiple people are recognized on camera, mosaic processing people except the person in front of them.

> with OpenCV, haarcascade model(frontal_face), Python.

---

### How to mosaic-processing?
> front face detection and resize image frame(Expansion). after scale-up image frame, resize image frame(scale-down) again.
> Then it will be broke resolution of image frame.


### Program execute process
> -> Get the image frame from the built-in camera.

> -> After converting the received frame to a grayscale, the face is detected using the haarcascade model.

> -> Print the number of faces detected and the center coordinates. 

> -> Mosaic processing the detected faces except for the front face.

---
### What to improve later on

> When running the program, there are no problems with functional parts such as face mosaic processing, but there are cases where the processing is released due to a slight lack of stability in face detection. It will be revised later.

> The mosaic generation algorithm was referenced in the following blog("https://jinho-study.tistory.com/231").
