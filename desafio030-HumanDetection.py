import ultralytics
import cv2
import numpy as np

# Carrega o modelo treinado no COCO
model = ultralytics.YOLO('yolov8n.pt')

class_names = model.names #Pega nome das classes do modelo

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, conf = 0.5, verbose = False) # precisão mínima de 0.5, 
    #possível modificação do valor (base = 0.25)
    
    annotated = frame.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0]) #classe detectada
            if class_id == 0: #se a classe é "pessoa"
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = class_names[class_id]
                confidence = float(box.conf[0])

                 # Desenha retângulo e label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label} {confidence:.1f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow("Detecções da rede YOLO", annotated)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()