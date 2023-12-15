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

    # sender1 = Sender(1000, 5000)
    # sender2 = Sender(6000, 10000)

    # sender1.load_image("./data/1_20_Imagen1.png") # Image1 20x20 pixels
    # sender1.load_text("./data/text.txt") # Text file
    # audio1 = sender1.send_all_data()
    # sender1.playText(audio1)

    # sender2.load_image("./data/1_14_Imagen2.png") # Image1 20x20 pixels
    # sender2.load_text("./data/text2.txt") # Text file
    # audio2 = sender2.send_all_data()

    # # padding if len(audio1) != len(audio2)
    # if len(audio1) > len(audio2):
    #     audio2 = np.pad(audio2, (0, len(audio1) - len(audio2)),'constant')
    # else:
    #     audio1 = np.pad(audio1, (0, len(audio2) - len(audio1)),'constant')

    # audio = audio1 + audio2
    

    # plt.plot(audio)
    # plt.show()

    receiver1 = Receiver(1000, 5000, 20)
    receiver2 = Receiver(6000, 10000, 14)

    audio = receiver2.listen(70)

    wavfile.write("sender1y2_fromCellphone_andTablet.wav", 44100, audio)
    # fs, audio = wavfile.read("sender2_fromCellphone.wav")

    img1, text1 = receiver1.demux_audio(audio)
    img2, text2 = receiver2.demux_audio(audio)

    print(text1)
    print(text2)
    plt.imshow(img1, cmap='gray')
    plt.show()
    plt.imshow(img2, cmap='gray')
    plt.show()

