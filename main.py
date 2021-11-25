import mediapipe as mp
import cv2 as cv
from pyKey import pressKey, releaseKey

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

RANGE_FOR_VALUE = 0.05


def check_range(first_value, second_value):
    if second_value - RANGE_FOR_VALUE <= first_value <= second_value + RANGE_FOR_VALUE:
        return True
    return False


def gestures(hand):
    thumptip_x, thumptip_y = (hand[mp_hands.HandLandmark.THUMB_TIP].x,
                              hand[mp_hands.HandLandmark.THUMB_TIP].y)

    indextip_x, indextip_y = (hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                              hand[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)

    midtip_x, midtip_y = (hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                          hand[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)

    ringtip_x, ringtip_y = (hand[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                            hand[mp_hands.HandLandmark.RING_FINGER_TIP].y)

    pinkytip_x, pinkytip_y = (hand[mp_hands.HandLandmark.PINKY_TIP].x,
                              hand[mp_hands.HandLandmark.PINKY_TIP].y)

    if check_range(thumptip_x, indextip_x) and \
            check_range(thumptip_y, indextip_y):
        pressKey("a")
    else:
        releaseKey("a")

    if check_range(thumptip_x, midtip_x) and \
            check_range(thumptip_y, midtip_y):
        pressKey("w")
    else:
        releaseKey("w")

    if check_range(thumptip_x, ringtip_x) and \
            check_range(thumptip_y, ringtip_y):
        pressKey("d")
    else:
        releaseKey("d")

    if check_range(thumptip_x, pinkytip_x) and \
            check_range(thumptip_y, pinkytip_y):
        pressKey("s")
    else:
        releaseKey("s")


cap = cv.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    hand_output = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                gestures(hand_landmark.landmark)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Flip the image horizontally for a selfie-view display.
        cv.imshow('WASD handtrack', cv.flip(image, 1))
        if cv.waitKey(5) & 0xFF == 27:
            break
cap.release()
