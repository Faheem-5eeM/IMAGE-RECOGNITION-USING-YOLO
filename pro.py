import cv2
import numpy as np

# Load YOLO network
net = cv2.dnn.readNetFromDarknet("yolo_custom.cfg", r"yolo_custom_4000.weights")

# Define class labels
classes = ['crop', 'weed']

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img = cv2.resize(img, (1280, 690))  # Resize the image to fit YOLO input size
    height, width, _ = img.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    output_layers_name = net.getUnconnectedOutLayersNames()

    # Perform forward pass and get outputs from the network
    layerOutputs = net.forward(output_layers_name)

    boxes = []
    confidences = []
    class_ids = []

    # Process each output
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            # If confidence is above threshold, extract bounding box information
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))  # Random colors for bounding boxes

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]

            # Draw the rectangle and put text on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y - 10), font, 2, color, 2)

    # Display the result
    cv2.imshow('Detection', img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
