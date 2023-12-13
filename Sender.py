import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as signal


class Sender:
    """ Class that represents the sender of the communication channel"""

    def __init__(self, channelFreq: float = 600, bandwidth: float = 800):
        self.Data = None
        self.textBinData = None
        self.RData = None
        self.GData = None
        self.BData = None
        self.sampleRate = 44100
        self.freqDuration = 0.05
        self.headerDuration = self.freqDuration*20 # 1 second header
        self.bandwidth = bandwidth
        self.channelFreq = channelFreq
        self.headerF1 = 80
        self.headerF2 = 500

        self.textFreqDict = create_freq_dict(self.channelFreq, self.bandwidth, 2)


    def send_image(self):
        """ Sends the image data with the header """
        pass

    def send_text(self) -> np.ndarray:
        """ Write audio data from frequency list"""

        bitList = [item for sublist in self.textBinData for item in sublist] # flatten the list
        audio = []
        tHeader = np.linspace(0, self.headerDuration, int(self.sampleRate * self.headerDuration))

        # Create chirp header. Duration 10 times freqDuration = 1 second.
        header = signal.chirp(tHeader, self.headerF1, self.headerDuration, self.headerF2, method='linear')

        t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
        

        phase = 0
        for bit in bitList:
            freq = self.textFreqDict[int(bit)]
            phase += 2 * np.pi * freq * t[-1]  # Calculate the phase at the end of the frequency
            audio.append(np.sin(2 * np.pi * freq * t + phase))  # Start the next frequency at this phase

        audio = np.hstack(audio)
        # print("the length of the audio signal is", len(audio))

        # Add header to the beginning of the audio.
        audio = np.concatenate((header, audio))
        # print("The length of the audio signal with the header is", len(audio))
        return audio

    def playText(self, audio):
        """ Plays the audio data including the 50 Hz header """
        sd.play(audio, self.sampleRate)
        sd.wait()

    def audioToCSV(self, audio, filename):
        """ Converts the audio data to csv file """
        np.savetxt(filename, audio, delimiter=",")

    def load_image(self, path: str):
        """ Loads the image from the path """
        pass

    def load_text(self, path: str):
        """ Loads the text from the path """
        with open(path, 'r', encoding='utf-8') as file:
            rawData = file.read()

        self.textBinData = string_to_bits(rawData)

    def dataToFrequency(self, n: int) -> list:
        """ Converts the data to list of frequencies
        :param
        n: the number of symbols in the n-fsk modulation
        :return: the list of frequencies """
        freqList = []  # np.array(len(self.textBinData)*2)
        if n == 2:
            for index, byte in enumerate(self.textBinData):
                for bit in byte:
                    freqList.append(self.textFreqDict[int(bit)])
        else:
            for index, byte in enumerate(self.textBinData):
                semiByte1 = byte[0:4]
                semiByte2 = byte[4:8]
                freqList.append(self.textFreqDict[int(semiByte1, 2)])
                freqList.append(self.textFreqDict[int(semiByte2, 2)])

        return freqList


def string_to_bits(s):
    ascii_list = [ord(ch) for ch in s]  # Ascii values of the characters
    return [format(i, '08b') for i in ascii_list]  # Convert to binary


def create_freq_dict(channelFreq: float, bandwidth: float, n: int) -> dict:
    """ Creates a dictionary of frequencies for the given channel """
    freqDict = {}
    freqs = np.linspace(channelFreq - bandwidth / 2 + bandwidth / (2 * n),
                        channelFreq + bandwidth / 2 - bandwidth / (2 * n), n)

    for index, item in enumerate(freqs):
        freqDict[index] = item

    return freqDict

# %%
