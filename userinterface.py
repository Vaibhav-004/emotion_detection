import webcamemotion

import pygame
import time
import os, random
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from playsound import playsound
import numpy
import cv2
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from numpy import asarray

pygame.mixer.init()
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

# initialise GUI
timelaps=20000
top = tk.Tk()
top.geometry('800x600')
top.title('Music Emotion Detection')
img= PhotoImage(file='img.png', master= top)
img_label= Label(top,image=img)

#define the position of the image
img_label.place(x=0, y=0)


label = Label(top)
label = Label(top, background='#CDCDCD', font=('arial', 18, 'bold'))
label.place(relx=0.80, rely=0.30)
sign_image = Label(top)


def classify(file_path):
    print("classification code");
    global label_packed

    frame = cv2.imread(file_path)
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
            elabel = class_labels[preds.argmax()]
            # emotionlabel=elabel
            label.configure(foreground='#011638', text=elabel)

            print("\nprediction max = ", preds.argmax())
            print("\nlabel = ", elabel)
            label_position = (x, y)
            cv2.putText(frame, elabel, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            predicted_emotion=""
            predicted_emotion=elabel
            if predicted_emotion == "Angry":
                play_angry()
            if predicted_emotion == "Happy":
                play_Happy()
            if predicted_emotion == "Neutral":
                play_Neutral()
            if predicted_emotion == "Sad":
                play_Sad()
            if predicted_emotion == "Surprise":
                play_Surprise()



        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        print("\n\n")
    cv2.imshow('Emotion Detector', frame)


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.80, rely=0.70)

def liveEmotion():
    print("webcam")
    webcamemotion.runwebcam()




def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25),(top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


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
    pygame.mixer.music.load("C:\\Users\\Asus\\PycharmProjects\\EmotionDetectionq\\songs\\Surprise\\"+ song)
    pygame.mixer.music.play()
    pygame.time.delay(timelaps)
    pygame.mixer.music.stop()


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.place(relx=0.80, rely=0.80)


Angry = Button(top, text="Angry", command=play_angry, padx=5, pady=15)
Angry.configure(background='red', foreground='white', font=('arial', 10, 'bold'))
Angry.pack(side=LEFT, pady=5, padx=5)

Happy = Button(top, text="Happy", command=play_Happy, padx=5, pady=15)
Happy.configure(background='blue', foreground='white', font=('arial', 10, 'bold'))
Happy.pack(side=LEFT, pady=5, padx=5)

Neutral = Button(top, text="Neutral", command=play_Neutral, padx=5, pady=15)
Neutral.configure(background='Green', foreground='white', font=('arial', 10, 'bold'))
Neutral.pack(side=LEFT, pady=5, padx=5)

Sad = Button(top, text="Sad", command=play_Sad, padx=5, pady=15)
Sad.configure(background='brown', foreground='white', font=('arial', 10, 'bold'))
Sad.pack(side=LEFT, pady=5, padx=5)

Surprise = Button(top, text="Surprise", command=play_Surprise, padx=5, pady=15)
Surprise.configure(background='yellow', foreground='black', font=('arial', 10, 'bold'))
Surprise.pack(side=LEFT, pady=5, padx=5)



webcam = Button(top, text="Live Emotion Detection", command=liveEmotion, padx=21, pady=5)

webcam.place(relx=0.80, rely=0.90)

sign_image.pack(side=TOP, expand=True)
label.place(relx=0.60, rely=0.10)
heading = Label(top, text="Emotion Detection ", pady=20, font=('arial', 20, 'bold'))

heading.configure(background='#CDCDCD', foreground='#364156')
heading.place(relx=0.10, rely=0.20)
top.mainloop()
