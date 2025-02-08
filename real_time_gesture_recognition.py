import cv2
import mediapipe as mp
import numpy as np
from predict_example import predict_new_gesture

class RealTimeGestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        
    def extract_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        landmarks_row = []
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            for landmark in hand_landmarks.landmark:
                landmarks_row.extend([landmark.x, landmark.y, landmark.z])
                
        return landmarks_row, results.multi_hand_landmarks
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks_row, hand_detected = self.extract_landmarks(frame)
            
            if hand_detected:
                gesture = predict_new_gesture(landmarks_row)
                cv2.putText(frame, f"Gesture: {gesture}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("Real-time Gesture Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = RealTimeGestureRecognition()
    recognizer.run()
