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

    # sender = Sender(600, 800)
    # sender.load_text('./data/text.txt')
    # audio = sender.send_text()
    # sender.playText(audio)
    # padded_signal = np.concatenate((np.zeros(int(44100 * 3.607)), audio, np.zeros(int(44100 * 3.607))))
    
    # audio = padded_signal + np.random.normal(0, 0.1, len(padded_signal))

    
    
    receiver = Receiver(600, 800)
    # audio = receiver.listen(60)
    # wavfile.write("audio5.wav", 44100, audio)
    # print("Done listening")

    fs, audio = wavfile.read("audio6.wav")

    # # audio = audio[44100:]

    # Normalize signal

    audio = audio/np.max(np.abs(audio))
    audio = audio - np.mean(audio)

    # Plot signal

    receiver.plot_fft(audio)

    # Demodulate signal

    # receiver = Receiver(600, 800)

    bits = receiver.demodText_fft(audio)
    decoded_text = receiver.bits_to_text(bits)
    print(decoded_text)
