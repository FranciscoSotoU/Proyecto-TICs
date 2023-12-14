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
        self.freqDuration = 0.01
        self.headerDuration = self.freqDuration * 100 # 1 second header
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency
        self.headerF1 = 200
        self.headerF2 = 500
        self.set_freq_bands()
        self.set_freq_dicts()

    def set_freq_dicts(self):
        """ Sets the frequency dictionaries for the sender """
        self.textFreqDict = create_freq_dict(self.text_band, self.bands_range, 2)
        self.redFreqDict = create_freq_dict(self.r_band, self.bands_range, 2)
        self.greenFreqDict = create_freq_dict(self.g_band, self.bands_range, 2)
        self.blueFreqDict = create_freq_dict(self.b_band, self.bands_range, 2)

    def send_image(self) -> np.ndarray:
        """ Write audio data from frequency list"""
        bitListred = [item for sublist in self.redBinData for item in sublist]
        bitListgreen = [item for sublist in self.greenBinData for item in sublist]
        bitListblue = [item for sublist in self.blueBinData for item in sublist]
        audio = []
        bitListblue = self.encode_all(bitListblue)
        bitListgreen = self.encode_all(bitListgreen)
        bitListred = self.encode_all(bitListred)
        tHeader = np.linspace(0, self.headerDuration, int(self.sampleRate * self.headerDuration))

        # Create chirp header. Duration 10 times freqDuration = 1 second.
        header = signal.chirp(tHeader, self.headerF1, self.headerDuration, self.headerF2, method='linear')
        r_header = signal.chirp(tHeader, self.headerF1+self.r_band, self.headerDuration, self.headerF2+self.r_band, method='linear')
        g_header = signal.chirp(tHeader, self.headerF1+self.g_band, self.headerDuration, self.headerF2+self.g_band, method='linear')
        b_header = signal.chirp(tHeader, self.headerF1+self.b_band, self.headerDuration, self.headerF2+self.b_band, method='linear')
        
        red_audio =  []
        green_audio = []
        blue_audio = []
        for bit in bitListred:

            t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
            red_signal = np.sin(2 * np.pi * self.redFreqDict[int(bit)] * t)
            red_audio.append(red_signal)
        for bit in bitListgreen:
            #print(bit)
            t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
            green_signal = np.sin(2 * np.pi * self.greenFreqDict[int(bit)] * t)
            green_audio.append(green_signal)
        for bit in bitListblue:
            #print(bit)
            t = np.linspace(0, self.freqDuration, int(self.sampleRate * self.freqDuration))
            blue_signal = np.sin(2 * np.pi * self.blueFreqDict[int(bit)] * t)
            blue_audio.append(blue_signal)
        red_audio = np.hstack(red_audio) 
        red_audio =  np.concatenate((r_header, red_audio,np.flip(r_header)))
        green_audio = np.hstack(green_audio)
        green_audio =  np.concatenate((g_header, green_audio,np.flip(g_header)))
        blue_audio = np.hstack(blue_audio)
        blue_audio =  np.concatenate((b_header, blue_audio,np.flip(b_header)))
        audio = red_audio + green_audio + blue_audio
        
        # Add header to the beginning and end of the audio. The end header is flipped for reverse correlation.

        return audio


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
            audio = self.hamming_encode(audio)
            audio = np.hstack(audio)
            # print("the length of the audio signal is", len(audio))

            # Add header to the beginning of the audio.
            audio = np.concatenate((header, audio))
            # print("The length of the audio signal with the header is", len(audio))
            return audio
        
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
    def set_freq_bands(self):
        """ Sends all the data """

        range = self.max_frequency - self.min_frequency
        self.bands_range = (range/4)*0.5
        real_band_range = range/4
        self.text_band = self.min_frequency 
        self.r_band = self.min_frequency + 1*real_band_range
        self.g_band = self.min_frequency + 2*real_band_range
        self.b_band = self.min_frequency + 3*real_band_range

def string_to_bits(s):
    ascii_list = [ord(ch) for ch in s]  # Ascii values of the characters
    return [format(i, '08b') for i in ascii_list]  # Convert to binary


def create_freq_dict(minfreq: float, bandwidth: float, n: int) -> dict:
    """ Creates a dictionary of frequencies for the given channel """
    freqDict = {}
    freqs = np.linspace(minfreq, bandwidth + minfreq, n)
    for index, item in enumerate(freqs):
        freqDict[index] = item

    return freqDict

