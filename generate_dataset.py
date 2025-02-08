import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime

class HandGestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.recording = False
        self.current_gesture_name = ""
        self.landmarks_data = []
        self.gesture_map = {
            '1': 'Open_Hand',
            '2': 'Closed_Hand',
            '3': 'Right_Tilted_Hand',
            '4': 'Left_Tilted_Hand',
            '5': 'Other_Gesture'
        }

    def recognize_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if self.recording:
                    landmarks_row = []
                    for landmark in hand_landmarks.landmark:
                        landmarks_row.extend([landmark.x, landmark.y, landmark.z])
                    landmarks_row.append(self.current_gesture_name)
                    self.landmarks_data.append(landmarks_row)      

    def save_to_csv(self):
        if not self.landmarks_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hand_gestures_{timestamp}.csv"
        
        headers = []
        for i in range(21):
            headers.extend([f"x{i}", f"y{i}", f"z{i}"])
        headers.append("gesture")
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self.landmarks_data)
        
        print(f"Data saved to {filename}")
        self.landmarks_data = []

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognition()
    
    print("Controls:")
    print("R - Start/Stop recording")
    print("1 - Open Hand")
    print("2 - Closed Hand")
    print("3 - Right Tilted Hand")
    print("4 - Left Tilted Hand")
    print("5 - Other Gesture")
    print("S - Save to CSV")
    print("Q - Quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gesture = recognizer.recognize_gesture(frame)
        
        status = "Recording" if recognizer.recording else "Not Recording"
        cv2.putText(frame, f"Status: {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Gesture: {recognizer.current_gesture_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Hand Gesture Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.recording = not recognizer.recording
        elif key == ord('s'):
            recognizer.save_to_csv()
        elif key in [ord(str(i)) for i in range(1, 6)]:
            gesture_key = chr(key)
            recognizer.current_gesture_name = recognizer.gesture_map[gesture_key]
    
    cap.release()
    cv2.destroyAllWindows()