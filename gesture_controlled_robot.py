import cv2
import mediapipe as mp
import numpy as np
from controller import Robot, Motor
from predict_example import predict_new_gesture

class GestureControlledRobot:
    def __init__(self):
        # Configuração MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Configuração Webots
        self.TIME_STEP = 32
        self.MAX_SPEED = 6.4
        self.robot = Robot()
        
        # Configuração das rodas
        self.front_left_wheel = self.robot.getDevice("front left wheel")
        self.front_right_wheel = self.robot.getDevice("front right wheel")
        self.back_left_wheel = self.robot.getDevice("back left wheel")
        self.back_right_wheel = self.robot.getDevice("back right wheel")
        
        # Inicialização das rodas
        for wheel in [self.front_left_wheel, self.front_right_wheel, 
                     self.back_left_wheel, self.back_right_wheel]:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
    
    def extract_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        landmarks_row = []
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            for landmark in hand_landmarks.landmark:
                landmarks_row.extend([landmark.x, landmark.y, landmark.z])
                
        return landmarks_row, results.multi_hand_landmarks
    
    def set_motor_speeds(self, gesture):
        front_left_speed = 0.0
        front_right_speed = 0.0
        back_left_speed = 0.0
        back_right_speed = 0.0
        
        print("Gesture: " + gesture)
        if gesture == "Closed_Hand":  # forward
            front_left_speed = self.MAX_SPEED
            front_right_speed = self.MAX_SPEED
            back_left_speed = self.MAX_SPEED
            back_right_speed = self.MAX_SPEED
        elif gesture == "Open_Hand": #backward
            front_left_speed = -self.MAX_SPEED
            front_right_speed = -self.MAX_SPEED
            back_left_speed = -self.MAX_SPEED
            back_right_speed = -self.MAX_SPEED
        elif gesture == "Right_Tilted_Hand": #left
            front_left_speed = -self.MAX_SPEED / 2
            front_right_speed = self.MAX_SPEED / 2
            back_left_speed = -self.MAX_SPEED / 2
            back_right_speed = self.MAX_SPEED / 2
        elif gesture == "right":
            front_left_speed = self.MAX_SPEED / 2
            front_right_speed = -self.MAX_SPEED / 2
            back_left_speed = self.MAX_SPEED / 2
            back_right_speed = -self.MAX_SPEED / 2
            
        self.front_left_wheel.setVelocity(front_left_speed)
        self.front_right_wheel.setVelocity(front_right_speed)
        self.back_left_wheel.setVelocity(back_left_speed)
        self.back_right_wheel.setVelocity(back_right_speed)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened() and self.robot.step(self.TIME_STEP) != -1:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks_row, hand_detected = self.extract_landmarks(frame)
            
            # print("Hand detected: " + landmarks_row.isEmpty())

            if landmarks_row:
                gesture = predict_new_gesture(landmarks_row)
                self.set_motor_speeds(gesture)
                cv2.putText(frame, f"Gesture: {gesture}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Para o robô quando nenhum gesto é detectado
                self.set_motor_speeds("stop")
            
            cv2.imshow("Gesture Controlled Robot", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureControlledRobot()
    controller.run()
