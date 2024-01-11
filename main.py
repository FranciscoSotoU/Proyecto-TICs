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
from utils import * 
    
# sender1 = Sender(1000, 2000)
sender2 = Sender(4000, 6000)
print(sender2.FreqDict)

# sender1.load_image("./data/1_20_Imagen1.png") # Image1 20x20 pixels
# sender1.load_text("./data/text.txt") # Text file

sender2.load_image("./data/1_14_Imagen2.png") # Image2 14x14 pixels
sender2.load_text("./data/text2.txt") # Text file

# audio1 = sender1.send_all_data()
audio2 = sender2.send_all_data()

# if len(audio1) != len(audio2):
#     if len(audio1) > len(audio2):        
#         secs = 2
#         silence = np.zeros(secs * sender1.sampleRate)
#         audio2 = np.concatenate((silence, audio2))
        
#         audio2 = np.pad(audio2, (0, len(audio1) - len(audio2)),'constant')
#     else:
#         audio1 = np.pad(audio1, (0, len(audio2) - len(audio1)),'constant')

#     audio = audio1 + audio2

sender2.playText(audio2)

# ---------------------------------------------------------------

# receiver1 = Receiver(1000, 2000, 20)
# receiver2 = Receiver(3000, 5000, 14)

# audio1 = wavfile.read("sender1_lowFreq_0701.wav")[1]
# audio = receiver1.listen(100)
# wavfile.write("audio_finisimo.wav", 44100, audio)
# audio2 = receiver2.listen(35)
# audio2 = wavfile.read("sender2_lowFreq_0701.wav")[1]
# audio = wavfile.read("audio_finisimo.wav")[1]


# audio2 = audio2/np.max(np.abs(audio2))
# audio1 = audio1/np.max(np.abs(audio1))
# audio = audio/np.max(np.abs(audio))

# img2, txt2 = receiver2.decode_audio(audio)
# print(txt2)
# plt.imshow(img2, cmap='gray')

# img1, txt1 = receiver1.decode_audio(audio) 
# print(txt1)
# plt.imshow(img1, cmap='gray')
# print("done")






# Text file
    # audio1 = sender1.send_all_data()
    # sender1.playText(audio1)

    # sender2if len(audio1) != len(audio2)
    # if len(audio1) > len(audio2):
    #     audio2 = np.pad(audio2, (0, len(audio1) - len(audio2)),'constant')
    # else:
    #     audio1 = np.pad(audio1, (0, len(audio2) - len(audio1)),'constant')

    # audio = audio1 + audio2.load_image("./data/1_14_Imagen2.png") # Image1 20x20 pixels
    # sender2.load_text("./data/text2.txt") # Text file
    # audio2 = sender2.send_all_data()

    # # padding 

    #plt.plot(audio)
    #plt.show()

    # receiver1 = Receiver(1000, 5000, 20)
    # receiver2 = Receiver(6000, 10000, 14)

    # audio = receiver2.listen(70)

    # wavfile.write("wavfile_test_1.wav", 44100, audio)
    # # fs, audio = wavfile.read("sender2_fromCellphone.wav")

    # img1, text1 = receiver1.demux_audio(audio)
    # img2, text2 = receiver2.demux_audio(audio)

    # print(text1)
    # print(text2)
    # plt.imshow(img1, cmap='gray')
    # plt.show()
    # plt.imshow(img2, cmap='gray')
    # plt.show()

