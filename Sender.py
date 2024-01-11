import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as signal
import cv2

class Sender:
    """ Class that represents the sender of the communication channel"""

    def __init__(self, min_frequency,max_frequency):
        self.Data = None
        self.textBinData = None
        self.RData = None
        self.GData = None
        self.BData = None
        self.sampleRate = 44100
        self.freq_text_duration = 0.01
        self.freqDuration = 0.01
        # self.freq_text_duration = 0.01*1.75
        self.headerDuration = self.freqDuration * 100 # 1 second header
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency
        self.bandwidth = self.max_frequency - self.min_frequency
        self.headerF1 = 200
        self.headerF2 = 500
        self.set_freq_bands()
        self.set_freq_dicts()


    def set_freq_dicts(self):
        """ Sets the frequency dictionaries for the sender """
        self.FreqDict = create_freq_dict(self.min_frequency, self.bandwidth, 4)
        
    def send_image(self) -> np.ndarray:
        """ Write audio data from frequency list"""

        bitListred = [item for sublist in self.redBinData for item in sublist]
        bitListgreen = [item for sublist in self.greenBinData for item in sublist]
        bitListblue = [item for sublist in self.blueBinData for item in sublist]
        bitListblue = self.encode_all(bitListblue)
        bitListgreen = self.encode_all(bitListgreen)
        bitListred = self.encode_all(bitListred)
        #create a list of strings where each string are 2 element of a bitList
        bitListred = [str(bitListred[i]) + str(bitListred[i+1]) for i in range(0, len(bitListred), 2)]
        bitListgreen = [str(bitListgreen[i]) + str(bitListgreen[i+1]) for i in range(0, len(bitListgreen), 2)]
        bitListblue = [str(bitListblue[i]) + str(bitListblue[i+1]) for i in range(0, len(bitListblue), 2)]
        audio = []

        tHeader = np.linspace(0, self.headerDuration, int(self.sampleRate * self.headerDuration))

        header = signal.chirp(tHeader, self.headerF1+self.min_frequency, self.headerDuration, self.headerF2+self.min_frequency, method='linear')
        
        audio = []
        for bit in bitListred:
            t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
            red_signal = np.sin(2 * np.pi * self.FreqDict[bit] * t)
            audio.append(red_signal)
        for bit in bitListgreen:
            t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
            green_signal = np.sin(2 * np.pi * self.FreqDict[bit] * t)
            audio.append(green_signal)
        for bit in bitListblue:
            t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
            blue_signal = np.sin(2 * np.pi * self.FreqDict[bit] * t)
            audio.append(blue_signal)
        audio = np.concatenate(audio)
        audio =  np.concatenate([header, audio])

        return audio
    
    def send_all_data(self) -> np.ndarray:
        """ Write the data in the form of audio wave"""

        audio_img = self.send_image()
        audio_texto = self.send_text()

        audio = np.concatenate([audio_img, audio_texto])
        return audio



    def send_text(self) -> np.ndarray:
            """ Write audio data from frequency list
            :return: the audio data """

            bitList = [item for sublist in self.textBinData for item in sublist] # flatten the list
            bitList = self.encode_all(bitList)
            bitList = [str(bitList[i]) + str(bitList[i+1]) for i in range(0, len(bitList), 2)]
            audio = []
            N = int(self.sampleRate * self.freq_text_duration)
            # Create the audio signal
            for bit in bitList:
                t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
                text_signal = np.sin(2 * np.pi * self.FreqDict[bit] * t)
                audio.append(text_signal)

            return np.concatenate(audio)
        
    def hamming_encode(self,bits):
        """Encode bits using Hamming(7,4) code."""
        G = np.array([[1, 1, 0, 1],
                    [1, 0, 1, 1],
                    [1, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=bool)
        return np.dot(G, bits) % 2

    def encode_all(self,bitlist):
        new_bit_list = []
        for i in range(0, len(bitlist), 4):
            chunk = bitlist[i:i+4]
            chunk = np.array(chunk, dtype=int)
            encoded_chunk = self.hamming_encode(chunk)
            new_bit_list.append(encoded_chunk)
        return np.concatenate(new_bit_list)
    
    def playText(self, audio):
        """ Plays the audio data """
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
        g = g.flatten()
        g_binary = np.array([format(i, '08b') for i in g])
        b = b.flatten()
        b_binary = np.array([format(i, '08b') for i in b])
        #print(b_binary)
        self.redBinData = r_binary
        self.greenBinData = g_binary
        self.blueBinData = b_binary

    def load_text(self, path: str):
        """ Loads the text from the path """

        with open(path, 'r', encoding='utf-8') as file:
            rawData = file.read()

        self.textBinData = string_to_bits(rawData)
        # print(self.textBinData)

    def set_freq_bands(self):
        """ Sends all the data """


def string_to_bits(s):
    ascii_list = [ord(ch) for ch in s]  # Ascii values of the characters
    return [format(i, '08b') for i in ascii_list]  # Convert to binary


def create_freq_dict(minfreq: float, bandwidth: float, n: int) -> dict:
    """ Creates a dictionary of frequencies for the given channel """
    freqDict = {}
    freqs = np.linspace(minfreq, bandwidth + minfreq, n)
    #Create a srqt(n) bits  per frequency
    for index, item in enumerate(freqs):
        key =bin(index)[2:]
        if len(key) == 1:
            key = '0' + key
        freqDict[key] = item

    return freqDict

