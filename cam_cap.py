import cv2
import mediapipe as mp
import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
import pyautogui

screen_w, screen_h = pyautogui.size()

mp_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
]

def normalize_landmarks(landmarks):
    # landmarks is already list of 21 landmarks
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    
    vectors = []
    
    for a, b in mp_connections:
        vec = coords[b] - coords[a]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    
    # Flatten all normalized vectors into 1D feature vector
    return np.array(vectors).flatten()


def load_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    X, y = [], []

    for label_entry in data:
        for label, samples in label_entry.items():
            for sample in samples:
                for sample_name, coords in sample.items():
                    feature = np.array(coords, dtype=np.float32).flatten()
                    X.append(feature)
                    y.append(label)
    return np.array(X), np.array(y)

X, y = load_dataset('dataset.json')
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, y)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 

cap = cv2.VideoCapture(0)
cam_w, cam_h = int(cap.get(3)), int(cap.get(4))

cursor_x, cursor_y = 0, 0

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("exiting...")
            break
        
        mirrored_frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(mirrored_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                normalized = normalize_landmarks(hand_landmarks.landmark)
                prediction = knn.predict([normalized])
                label = prediction[0]

                if label == 'neutral':
                    cursor_x, cursor_y = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
                    cursor_x *= screen_w
                    cursor_y *= screen_h
                
                if label == 'left_click':
                    pyautogui.click()

                if label == 'right_click':
                    pyautogui.rightClick()
                
                if label == 'double_click':
                    pyautogui.doubleClick()

                pyautogui.moveTo(int(cursor_x), int(cursor_y))
                break

            cv2.putText(mirrored_frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        mirrored_frame = cv2.resize(mirrored_frame, None, fx = 0.2, fy = 0.2)
        cv2.imshow('Mirrored Frame', mirrored_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()