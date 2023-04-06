import torch
import cv2

# Load the pre-trained model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Start the video capture
cap = cv2.VideoCapture('cut video.mp4')

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame
    img = torch.from_numpy(frame)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)

    # Pass the frame through the model
    output = model(img)
    _, preds = torch.max(output, 1)

    # Get the predictions
    preds = preds.squeeze()
    preds = preds.numpy()

    # Draw the predictions on the frame
    for i in range(len(preds)):
        cv2.putText(frame, str(preds[i]), (i*20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit if the 'q' key is pressed
    if key == ord("q"):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
