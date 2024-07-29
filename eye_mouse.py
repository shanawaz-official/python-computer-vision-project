import cv2
import mediapipe as mp
import pyautogui as pag

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cam = cv2.VideoCapture(0)
screen_w, screen_h = pag.size()

while True:
    _, image = cam.read()
    image = cv2.flip(image, 1)
    window_h, window_w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh.process(rgb_image)
    all_face_landmarks = processed_image.multi_face_landmarks

    if all_face_landmarks:
        one_face_landmark = all_face_landmarks[0].landmark
        nose_x = 0
        nose_y = 0
        count = 0

        for id, landmark_points in enumerate(one_face_landmark[477:478]):
            x = int(landmark_points.x * window_w)
            y = int(landmark_points.y * window_h)
            cv2.circle(image, (x, y), 3, (0, 0, 255))

            nose_x += x
            nose_y += y
            count += 1

        if count > 0:
            nose_x = int(nose_x / count)
            nose_y = int(nose_y / count)

            mouse_x = int(screen_w / window_w * nose_x)
            mouse_y = int(screen_h / window_h * nose_y)
            pag.moveTo(mouse_x, mouse_y)

        left_eye = [one_face_landmark[145], one_face_landmark[159]]
        for landmark_points in left_eye:
            x = int(landmark_points.x * window_w)
            y = int(landmark_points.y * window_h)
            cv2.circle(image, (x, y), 3, (0, 255, 255))

        if abs(left_eye[0].y - left_eye[1].y) < 0.01:
            pag.click()
            pag.sleep(2)
            print("mouse clicked")

    cv2.imshow("Eye controlled mouse", image)
    key = cv2.waitKey(100)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
