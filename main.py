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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # send text
    sender = Sender(600, 800)
    sender.load_text('./data/text.txt')
    print(sender.textBinData[0:4])
    audio = sender.send_text()

    # padding the audio file
    duration = 1.13
    padding = np.zeros(int(duration * sender.sampleRate))
    audio = np.concatenate((padding, audio, padding))

    # Add noise
    noise = np.random.normal(0, 0.1, len(audio))
    audio = audio + noise

    # Normalization
    audio = audio / np.max(np.abs(audio))
    audio = audio - np.mean(audio)

    # play audio
    #sd.play(audio, sender.sampleRate)
    #sd.wait()

    # plot fft
    #freqs = np.fft.fftfreq(len(audio), 1 / sender.sampleRate)
    #fft = np.fft.fft(audio)
    #plt.plot(freqs[0:100000], np.abs(fft)[0:100000])
    #plt.plot(audio)
    #plt.show()

    # receive text
    receiver = Receiver(600, 800)
    binaries = receiver.demodulateText(audio)
    print(binaries[0:4])
    decoded_text = receiver.bits_to_text(binaries)
    print(decoded_text)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
