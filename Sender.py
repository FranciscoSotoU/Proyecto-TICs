import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as signal
import cv2
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


    def send_image(self) -> np.ndarray:
        """ Write audio data from frequency list"""
        bitList = [item for sublist in self.redBinData for item in sublist]
        audio = []
        tHeader = np.linspace(0, self.headerDuration, int(self.sampleRate * self.headerDuration))

        # Create chirp header. Duration 10 times freqDuration = 1 second.
        header = signal.chirp(tHeader, self.headerF1, self.headerDuration, self.headerF2, method='linear')

        for bit in bitList:
            #print(bit)
            t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
            audio.append(np.sin(2 * np.pi * self.textFreqDict[int(bit)] * t))
        
        audio = np.hstack(audio)

        # Add header to the beginning and end of the audio. The end header is flipped for reverse correlation.
        audio = np.concatenate((header, audio, np.flip(header)))
        return audio


    def send_text(self) -> np.ndarray:
        """ Write audio data from frequency list"""

        bitList = [item for sublist in self.textBinData for item in sublist] # flatten the list
        audio = []
        tHeader = np.linspace(0, self.headerDuration, int(self.sampleRate * self.headerDuration))

        # Create chirp header. Duration 10 times freqDuration = 1 second.
        header = signal.chirp(tHeader, self.headerF1, self.headerDuration, self.headerF2, method='linear')

        for bit in bitList:
            #print(bit)
            t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
            audio.append(np.sin(2 * np.pi * self.textFreqDict[int(bit)] * t))

        audio = np.hstack(audio)

        # Add header to the beginning and end of the audio. The end header is flipped for reverse correlation.
        audio = np.concatenate((header, audio, np.flip(header)))
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
        img = cv2.imread(path)
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        r = r.flatten()
        r_binary = np.array([format(i, '08b') for i in r])
        self.redBinData = r_binary
    def load_text(self, path: str):
        """ Loads the text from the path """
        with open(path, 'r', encoding='utf-8') as file:
            rawData = file.read()

        self.textBinData = string_to_bits(rawData)
        print(self.textBinData)

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
    def set_freq_bands(self,min_frequency,max_frequency):
        """ Sends all the data """
        range = max_frequency - min_frequency
        bands_range = range/5
        self.text_band = min_frequency + bands_range
        self.r_band = min_frequency + 2*bands_range
        self.g_band = min_frequency + 3*bands_range
        self.b_band = min_frequency + 4*bands_range


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

