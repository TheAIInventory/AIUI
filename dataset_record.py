import cv2
import os
import time

output_dir = "recorded_frames"
frame_count = 200

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("something is wrong with cam")
    exit()

frame_num = 0
while frame_num < frame_count:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("err")
        break
    
    frame_filename = os.path.join(output_dir, f"frame_{frame_num:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_num += 1
    cv2.imshow("record", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("record complete")