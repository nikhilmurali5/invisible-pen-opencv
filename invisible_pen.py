import cv2
import mediapipe as mp
import numpy as np

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Landmark IDs
INDEX_TIP = 8
FINGER_TIPS = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
FINGER_DIP = [6, 10, 14, 18]   # 2-joints before tip

# State variables
prev_x, prev_y = 0, 0
canvas = None

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if canvas is None:
        canvas = np.zeros_like(frame)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = handLms.landmark

            # Detect finger states
            fingers_up = 0
            for tip, dip in zip(FINGER_TIPS, FINGER_DIP):
                if lmList[tip].y < lmList[dip].y:
                    fingers_up += 1

            ix, iy = int(lmList[INDEX_TIP].x * w), int(lmList[INDEX_TIP].y * h)

            if fingers_up == 4:
                # Full palm shown - clear canvas
                canvas = np.zeros_like(frame)
                cv2.putText(frame, "PALM DETECTED - CLEARING", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                prev_x, prev_y = 0, 0
            else:
                # Draw with index fingertip
                if prev_x and prev_y:
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (255, 0, 0), 5)
                prev_x, prev_y = ix, iy
                cv2.putText(frame, "DRAWING", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0

    # Combine webcam feed and canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("Invisible Pen - Palm Eraser", combined)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
