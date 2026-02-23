import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# --- Configuration ---
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Colors: Blue, Green, Red, Yellow
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0 

# --- MediaPipe Setup ---
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
ret = True

while ret:
    # 1. Read Frame from Camera
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Create the White "Paint Window" (Fresh every frame)
    paintWindow = np.zeros((480, 640, 3)) + 255
    
    # 3. Draw UI Buttons ONLY on Paint Window
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2) # Clear
    paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1) # Blue
    paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1) # Green
    paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1) # Red
    paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1) # Yellow
    
    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

    # 4. Process Hand
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])
            
            # Draw Skeleton ONLY on Camera Frame (to help you see your hand)
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        
        # Get Finger Coordinates
        index_finger_tip = (landmarks[8][0], landmarks[8][1])
        thumb_tip = (landmarks[4][0], landmarks[4][1])
        center = index_finger_tip
        
        # Draw Small Pointer on Camera Frame (so you see your hand)
        cv2.circle(frame, center, 6, (0, 255, 255), -1)
        
        # Draw Small Cursor on Paint Window (so you see where you are painting)
        cv2.circle(paintWindow, center, 6, (100, 100, 100), -1)

        # Calculate Pinch Distance
        distance = np.sqrt((index_finger_tip[0]-thumb_tip[0])*2 + (index_finger_tip[1]-thumb_tip[1])*2)
        
        # --- Logic ---
        # If Pinch (< 30px) -> STOP Drawing (Lift Pen)
        if distance < 30:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1
            
        # If Open Hand -> Draw or Click
        elif center[1] <= 65:
            # Check button clicks (Using Paint Window coordinates)
            if 40 <= center[0] <= 140: # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                blue_index = green_index = red_index = yellow_index = 0
                
            elif 160 <= center[0] <= 255: colorIndex = 0
            elif 275 <= center[0] <= 370: colorIndex = 1
            elif 390 <= center[0] <= 485: colorIndex = 2
            elif 505 <= center[0] <= 600: colorIndex = 3
        else:
            # Add points to queues
            if colorIndex == 0: bpoints[blue_index].appendleft(center)
            elif colorIndex == 1: gpoints[green_index].appendleft(center)
            elif colorIndex == 2: rpoints[red_index].appendleft(center)
            elif colorIndex == 3: ypoints[yellow_index].appendleft(center)

    else:
        # Reset queues if hand leaves frame
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    # 5. Draw Lines ONLY on Paint Window
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                # Draw ONLY on paintWindow
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # 6. Show Windows
    cv2.imshow("Tracking", frame)      # Clean Camera Feed
    cv2.imshow("Paint", paintWindow)   # Drawing Canvas

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()