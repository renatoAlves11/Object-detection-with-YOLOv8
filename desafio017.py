import ultralytics
import cv2

model = ultralytics.YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        pred = model.predict(frame)
        cv2.imshow("Detecções da rede YOLO", predd[0].plot())
    
    if cv2.waitKey(1) == ord('q'):
        break
        
