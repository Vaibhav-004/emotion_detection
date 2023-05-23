import os, random
import pygame


pygame.mixer.init()


song=random.choice(os.listdir("D:\\Python Project old disk\\python project\\Face_Emotion_Recognization\\songs\\Sad\\"))
print(song)
pygame.mixer.music.load("D:\\Python Project old disk\\python project\\Face_Emotion_Recognization\\songs\\Sad\\"+song)
pygame.mixer.music.play()
pygame.time.delay(20000)
pygame.mixer.music.stop()