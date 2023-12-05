import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt


class Sender:
    """ Class that represents the sender of the communication channel"""

    def __init__(self, channelFreq: float = 600, bandwidth: float = 800):
        self.Data = None
        self.textBinData = None
        self.RData = None
        self.GData = None
        self.BData = None
        self.sampleRate = 44100
        self.freqDuration = 0.1
        self.bandwidth = bandwidth
        self.channelFreq = channelFreq

        self.textFreqDict = create_freq_dict(self.channelFreq, self.bandwidth, 16)

    def send_image(self):
        """ Sends the image data with the header """
        pass

    def send_text(self) -> np.ndarray:
        """ Write audio data from frequency list"""

        freqList = self.dataToFrequency()
        # to sinewaves
        audio = []
        tHeader = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
        header = np.sin(2 * np.pi * 50 * tHeader)

        for freq in freqList:
            t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
            audio.append(np.sin(2 * np.pi * freq * t))

        audio = np.hstack(audio)
        audio = np.concatenate((header, audio, header))

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
        with open(path, 'r') as file:
            rawData = file.read()

        self.textBinData = string_to_bits(rawData)

    def dataToFrequency(self) -> list:
        """ Converts the data to list of frequencies """

        freqList = []  # np.array(len(self.textBinData)*2)
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
