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

    sender = Sender(600, 800)
    sender.load_text('data/text.txt')
    audio = sender.send_text()
    sender.playText(audio)



    # receiver = Receiver(600, 800)
    # audio = receiver.listen(60)
    # wavfile.write("audio2.wav", 44100, audio)
    # print("Done listening")

    # fs, audio = wavfile.read("audio1.wav")

    # audio = audio[44100:]

    # # Normalize signal

    # audio = audio/np.max(np.abs(audio))
    # audio = audio - np.mean(audio)

    # # Plot signal

    # plt.plot(audio)
    # plt.show()

    # # Demodulate signal

    #receiver = Receiver(600, 800)

    #bits = receiver.demodulateImage(audio)
    #decoded_text = receiver.bits_to_image(bits)
    #cv2.imshow(decoded_text)
    #print(decoded_text)
    

