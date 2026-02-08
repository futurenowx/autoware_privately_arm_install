import cv2
cap = cv2.VideoCapture(6)
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame from camera")
        break
    cv2.imshow("Test Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

