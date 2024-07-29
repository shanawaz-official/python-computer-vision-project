import cv2
import mediapipe as mp
import pyautogui as pag

cam = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_w, screen_h = pag.size()
index_x_prev, index_y_prev = 0, 0
index_x_curr, index_y_curr = 0, 0
thumb_x_curr, thumb_y_curr = 0, 0
threshold = 10  

while True:
    _, image = cam.read()
    image_h, image_w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing_utils.draw_landmarks(image, hand_landmarks)
            landmarks = hand_landmarks.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * image_w)
                y = int(landmark.y * image_h)
                if id == 8: 
                    index_x_curr = screen_w / image_w * x
                    index_y_curr = screen_h / image_h * y
                if id == 4: 
                    thumb_x_curr = screen_w / image_w * x
                    thumb_y_curr = screen_h / image_h * y
                    if abs(index_y_curr - thumb_y_curr) < 20:
                        pag.click()

        
            if abs(index_x_curr - index_x_prev) > threshold or abs(index_y_curr - index_y_prev) > threshold:
                pag.moveTo(index_x_curr, index_y_curr)
                index_x_prev, index_y_prev = index_x_curr, index_y_curr

    cv2.imshow('Finger Mouse', image)
    cv2.waitKey(1)