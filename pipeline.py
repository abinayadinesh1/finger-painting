import cv2
import mediapipe as mp
import PyDynamixel_v2 as pd
from time import sleep
import numpy as np

# import sys
# print(sys.executable)
# print(sys.path)


# set up motors
port = '/dev/tty.usbmodem58FD0172521'
baudrate = 1000000
serial = pd.DxlComm(port=port, baudrate=baudrate)

dyn1_id = 1
dyn2_id = 2
dyn1 = pd.Joint(dyn1_id)
dyn2 = pd.Joint(dyn2_id)

serial.attach_joints([dyn1, dyn2])

serial.enable_torques()

print(serial.joint_ids)

# 2 is x, 1 is y

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,      # Set to False for video
    max_num_hands=1,              # Only detect one hand
    min_detection_confidence=0.5, # Lower this if tracking is too intermittent
    min_tracking_confidence=0.5   # Lower this cautiously for faster tracking

)

# Initialize OpenCV camera capture
cap = cv2.VideoCapture(0)

# these are the points
points = []

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the BGR image to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Assuming only one hand is detected for simplicity
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the index finger tip landmark (landmark 8)
        index_tip = hand_landmarks.landmark[8]

        # Get frame dimensions
        h, w, _ = frame.shape

        # Convert normalized coordinates to pixel coordinates
        cx, cy = int(index_tip.x * w), int(index_tip.y * h) # (it was previously between zero and 1)
        points.append((cx, cy))
        theta_x = int(index_tip.x * 360)
        theta_y = int(index_tip.y * 360)

        serial.send_angles({1: theta_y, 2: theta_x})

        # Draw a red circle at the index finger tip
        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        # Display coordinates on the frame
        text = f'Index Tip: ({cx}, {cy})'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(index_tip.x, index_tip.y)
        print(h,w)
        print(cx, cy)
                
    # Show the frame with the circle and coordinates
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Create the drawing with the collected points
if points:
    # Get the dimensions from the last captured frame
    h, w, _ = frame.shape
    
    # Create a white canvas
    drawing = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Number of points
    num_points = len(points)
    
    if num_points > 0:
        # Define colors: pink (180, 105, 255) to blue (255, 0, 0) in BGR
        pink = np.array([180, 105, 255])
        blue = np.array([255, 0, 0])

        green = np.array([152,251,152])
        red = np.array([250,128,114])
        
        # Draw lines first (under the points)
        for i in range(1, num_points):
            # Calculate position for the line (0 to 1)
            position = (i - 0.5) / max(1, num_points - 1)
            
            # Calculate color using linear interpolation
            color = tuple(map(int, pink * (1 - position) + blue * position))
            
            # Calculate alpha (0-1 range) for transparency effect
            alpha = 0.2 + 0.8 * position  # 0.2 to 1.0
            
            # Draw the line with transparency effect (thicker for newer lines)
            thickness = max(1, int(2 * position + 1))
            
            # Get a copy of the current drawing
            temp = drawing.copy()
            
            # Draw the line on temp
            cv2.line(temp, points[i-1], points[i], color, thickness)
            
            # Blend temp with the drawing using the calculated alpha
            cv2.addWeighted(temp, alpha, drawing, 1 - alpha, 0, drawing)
        
        # Draw points on top of lines
        for i in range(num_points):
            # Calculate position in sequence (0 to 1)
            position = i / max(1, num_points - 1)
            
            # Calculate color
            color = tuple(map(int, pink * (1 - position) + blue * position))
            
            # Calculate alpha for transparency effect
            alpha = 0.2 + 0.8 * position  # 0.2 to 1.0
            
            # Adjust point size based on recency (newer points are larger)
            point_size = max(3, int(4 * position + 3))
            
            # Get a copy of the current drawing
            temp = drawing.copy()
            
            # Draw the point on temp
            cv2.circle(temp, points[i], point_size, color, -1)
            
            # Blend temp with the drawing using the calculated alpha
            cv2.addWeighted(temp, alpha, drawing, 1 - alpha, 0, drawing)
    
    # Save the drawing as a PNG file
    filename = 'demo_in_class23.png'
    cv2.imwrite(filename, drawing)
    print(f"Drawing saved as {filename}")
    print(f"Total points captured: {len(points)}")
else:
    print("No points were captured.")



# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()