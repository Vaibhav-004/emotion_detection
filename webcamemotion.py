
#USAGE : python test.py
from playsound import playsound
from keras.models import load_model
from time import sleep
from numpy import asarray
import os, random
from keras.preprocessing import image
import cv2
import numpy as np
import pygame
import time

timelaps=20000
def play_angry():
    print("Angry song")
    song = random.choice(os.listdir("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Angry\\"))
    print(song)
    pygame.mixer.music.load("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Angry\\"+ song)
    pygame.mixer.music.play()
    pygame.time.delay(timelaps)
    pygame.mixer.music.stop()



def play_Happy():
    print("Angry song")
    song = random.choice(os.listdir("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Happy\\"))
    print(song)
    pygame.mixer.music.load("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Happy\\"+ song)
    pygame.mixer.music.play()
    pygame.time.delay(timelaps)
    pygame.mixer.music.stop()


def play_Neutral():
    print("Angry song")
    song = random.choice(os.listdir("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Neutral\\"))
    print(song)
    pygame.mixer.music.load("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Neutral\\"+ song)
    pygame.mixer.music.play()
    pygame.time.delay(timelaps)
    pygame.mixer.music.stop()


def play_Sad():
    print("Angry song")
    song = random.choice(os.listdir("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Sad\\"))
    print(song)
    pygame.mixer.music.load("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Sad\\"+ song)
    pygame.mixer.music.play()
    pygame.time.delay(timelaps)
    pygame.mixer.music.stop()

def play_Surprise():
    print("Angry song")
    song = random.choice(os.listdir("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Surprise\\"))
    print(song)
    pygame.mixer.music.load("C:\\Users\\Asus\\PycharmProjects\\EmotionDetection\\songs\\Surprise\\"+ song)
    pygame.mixer.music.play()
    pygame.time.delay(timelaps)
    pygame.mixer.music.stop()

def runwebcam():
    face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    classifier = load_model('./Emotion_Detection.h5')

    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = asarray(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]
                print("\nprediction = ", preds)
                label = class_labels[preds.argmax()]
                print("\nprediction max = ", preds.argmax())
                print("\nlabel = ", label)
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                if label == "Angry":
                    play_angry()
                if label == "Happy":
                    play_Happy()
                if label == "Neutral":
                    play_Neutral()
                if label == "Sad":
                    play_Sad()
                if label == "Surprise":
                    play_Surprise()



            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            print("\n\n")
        cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



























