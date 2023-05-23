import webcamemotion

import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import cv2
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from numpy import asarray
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

# initialise GUI

top = tk.Tk()
top.geometry('800x600')
top.title('Image Classification CIFAR10')
top.configure(background='#CDCDDD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
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
            #emotionlabel=elabel
            label.configure(foreground='#011638', text=elabel)
            print("\nprediction max = ", preds.argmax())
            print("\nlabel = ", elabel)
            label_position = (x, y)
            cv2.putText(frame, elabel, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)



        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        print("\n\n")
    cv2.imshow('Emotion Detector', frame)



def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def liveEmotion():
    print("webcam")
    webcamemotion.runwebcam()


    #os.system('python test.py')




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



upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)

webcam = Button(top, text="Live Emotion Detection", command=liveEmotion, padx=21, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
webcam.pack(side=BOTTOM,pady=70)
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Emotion Detection ", pady=20, font=('arial', 20, 'bold'))

heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
