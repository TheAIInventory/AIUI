import cv2
from tqdm import tqdm
import mediapipe as mp
import numpy as np
import os
import json

image_dir = "recorded_frames"
json_file = "dataset.json"
label = "double_click"   #Change for each gesture you want to add

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
else:
    data = []

def get_label_entry(data, label):
    for entry in data:
        if label in entry:
            return entry[label]
    new_entry = {label:[]}
    data.append(new_entry)
    return new_entry[label]

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
    return np.array(vectors).tolist()

for filename in tqdm(os.listdir(image_dir)):
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        filepath = os.path.join(image_dir, filename)
        image = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            normalized = normalize_landmarks(landmarks)
            sample_name = os.path.splitext(filename)[0]
            sample = {sample_name: normalized}
        
            label_entry = get_label_entry(data, label)
            label_entry.append(sample)
        
with open(json_file, 'w') as f:
    json.dump(data, f, indent=4)

hands.close()
print("dataset updated")
            
