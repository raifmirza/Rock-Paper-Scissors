import random
import cv2 as cv
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

mp_solution = mp_hands.Hands(max_num_hands=1,
                             model_complexity=1,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)


tip_of_finger = [4, 8, 12, 16, 20]
dip_of_finger = [3, 7, 11, 15, 19]
font = cv.FONT_HERSHEY_SIMPLEX


def create_blank(width, height):
    return np.zeros((width, (2*height)+1, 3), np.uint8)


def ai_turn(w, h):
    a = random.randint(0, 2)
    name = ""
    if a == 0:
        img = cv.imread('paper.png')
        name = 'Paper'
    if a == 1:
        img = cv.imread('scissor.png')
        name = 'Scissor'
    if a == 2:
        img = cv.imread('rock.png')
        name = 'Rock'
    img = cv.resize(img, (h, w), cv.INTER_AREA)
    return img, name


def open_finger(results, w, h):
    finger = []

    which_hand = results.multi_handedness[0].classification[-1].label

    for i, handLms in enumerate(results.multi_hand_landmarks):
        z = -1
        LmX = []
        LmY = []
        for k in range(21):
            LmX.append(handLms.landmark[k].x)
            LmY.append(handLms.landmark[k].y)
            if k % 4 == 0 and k != 4:
                z += 1
                if float(handLms.landmark[tip_of_finger[z]].y) < float(handLms.landmark[dip_of_finger[z]].y):
                    finger.append(1)
                else:
                    finger.append(0)
        X_max = int(max(LmX) * h)
        X_min = int(min(LmX) * h)
        Y_max = int(max(LmY) * w)
        Y_min = int(min(LmY) * w)

    return finger, (X_max+10, Y_max+10), (X_min-10, Y_min-10), which_hand


def which(finger):
    if all(finger[1:-1]):
        return "Paper"
    elif finger[1] == 1 and finger[2] == 1 and not any(finger[3:-1]):
        return "Scissor"
    elif finger[0] == 1 and not any(finger[1:-1]):
        return "Rock"
    else:
        return " "


score = {"player": 0, "computer": 0}


def game_play(player, computer):
    if player == 'Paper' and computer == 'Scissor':
        score["computer"] += 1
    if player == 'Paper' and computer == 'Rock':
        score["player"] += 1
    if player == 'Scissor' and computer == 'Rock':
        score['computer'] += 1
    if player == 'Scissor' and computer == 'Paper':
        score['player'] += 1
    if player == 'Rock' and computer == 'Paper':
        score['computer'] += 1
    if player == 'Rock' and computer == 'Scissor':
        score['player'] += 1
    return score


def find_finger(frame):
    which_select = ''
    frame = cv.flip(frame, 1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = mp_solution.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            finger, start, end, hand = open_finger(results, w, h)
            which_select = which(finger)
            cv.rectangle(frame, start, end, color=(255, 0, 0), thickness=3)
            cv.putText(frame, hand + " " + which_select, end,
                       fontFace=font, fontScale=1, color=(0, 255, 0), thickness=3)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    return frame, which_select


def title_text(blank):
    text = "Rock Paper Scissors"
    textsize = cv.getTextSize(text, font, 1, 2)[0]
    textX = (blank.shape[1] - textsize[0]) / 2
    textY = textsize[1]
    cv.putText(blank, text, (int(textX), int(textY)),
               font, 1, (255, 255, 255), 2)


def score_text(blank, score):
    player = score['player']
    computer = score['computer']
    h = blank.shape[1]
    computer_textX = h / 4
    computer_textY = 30
    player_textX = 3*(h/4)
    cv.putText(blank, str(computer), (int(computer_textX),
               computer_textY), font, 1, (255, 255, 255), 2)
    cv.putText(blank, str(player), (int(player_textX),
               computer_textY), font, 1, (255, 255, 255), 2)


cap = cv.VideoCapture(0)
prev_time = time.time()
delta = 0
blank = create_blank(510, 640)

a = ''
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        w, h, z = frame.shape
        current_time = time.time()
        delta += current_time - prev_time
        prev_time = current_time
        if delta > 2:  # 2sec delay
            blank = create_blank(510, 640)
            ai_img, a = ai_turn(w, h)
            blank[30:, :640, :] = ai_img
            score_text(blank, game_play(what, a))
            delta = 0

        frame, what = find_finger(frame)
        frame = cv.resize(frame, (640, 480), cv.INTER_AREA)
        blank[30:, 640:-1, :] = frame
        title_text(blank)
        cv.imshow("Frame", blank)
    if cv.waitKey(20) & 0XFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
