# This is a sample Python script.
import matplotlib.pyplot as plt
import pickle
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Sender import *
from Receiver import *

import sounddevice as sd
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    sender = Sender(1000, 5000)
    sender.load_image("./data/1_20_Imagen1.png") # Image1 20x20 pixels
    sender.load_text("./data/text.txt") # Text file

    audio = sender.send_all_data()

    plt.plot(audio)
    plt.show()

    receiver = Receiver(1000, 5000)
    #fs, audio = wavfile.read("audio_nuevo.wav")

    img, text = receiver.demux_audio(audio)
    print(text)
    plt.imshow(img, cmap='gray')
    plt.show()

