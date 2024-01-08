import numpy as np
import sounddevice as sd
import matplotlib
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import pearsonr
from scipy.io import wavfile
#import cv2


class Receiver:
    """ Class that represents the receiver of the communication channel"""

    def __init__(self,  min_frequency, max_frequency, image_width):
        self.buffer = None
        self.textData = None
        # self.channel = channel
        self.samplerate = 44100
        self.freqDuration = 0.01
        # self.freq_text_duration = 0.01*1.75
        self.headerDuration = 1 # 1 second header
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency
        self.headerF1 = 200
        self.headerF2 = 500
        self.set_freq_dict()
        self.textLength = 105
        self.image_width = image_width; 
        self.image_bit_size = self.image_width**2 * 8 
        self.text_bit_size = self.textLength*8  
        self.redBits = None
        self.greenBits = None
        self.blueBits = None

    def listen(self, duration):
        """ Listens to the channel for a message """
        # record audio
        data = sd.rec(frames=int(self.samplerate * duration),
                      samplerate=self.samplerate, channels=1)
        data = data[:, 0]
        print("Listening")
        sd.wait()
        print("Done listening")
        self.buffer = data
        # wavfile.write("audio1.wav", self.samplerate, data)
        return data   
    
    
    def demodulate_audio(self, audio_signal: np.ndarray, channel ='text', initial_index=None) -> (int, int, list):
        """ Demodulates the audio signal into binary
        :param audio_signal: the audio signal to be decoded
        :param channel: the channel of the audio signal
        :param initial_index: the initial index of the signal
        :return: the initial index, the last index and the string bytes list"""
        
        # initial_index = 0

        if initial_index == None:
            initial_index = self.find_header(audio_signal, self.headerDuration, channel)
        index = initial_index + int(self.headerDuration * self.samplerate)        
        bits_list = []

        
        red_index = 0
        green_index = self.image_bit_size // 8
        blue_index = (self.image_bit_size * 2) // 8
        text_index = (self.image_bit_size * 3) // 8
        
        bits_list = self.demodText_fft(audio_signal, index, channel, self.freqDict)
        bits_list= np.array(bits_list)
        grouped_values = [''.join(str(bit) for bit in bits_list[i:i+7]) for i in range(0, len(bits_list), 7)] # list of strings of 7 bits
        bits_list_decoded = self.decode_all(grouped_values)

        bytes_list = [''.join(str(bit) for bit in bits_list_decoded[i:i+8]) for i in range(0, len(bits_list_decoded), 8)] #return as bytes
        
        red_bytes = bytes_list[red_index:green_index]
        green_bytes = bytes_list[green_index:blue_index]
        blue_bytes = bytes_list[blue_index:text_index]
        text_bytes = bytes_list[text_index:]
        
        return red_bytes, green_bytes, blue_bytes, text_bytes
    
    
    def demodText_fft(self, audio_signal, index, channel, freqDict) -> list:
        """ Demodulates the signal into binary
        :param audio_signal: the signal to be demodulated
        :return: the binary list """

        # initial_index = self.find_header(audio_signal, self.headerDuration)

        delta = int(self.freqDuration * self.samplerate)
        last_index = index + int((self.text_bit_size+3*self.image_bit_size) * self.samplerate * self.freqDuration)
        zeroFrequency = freqDict['00']
        oneFrequency = freqDict['01']
        twoFrequency = freqDict['10']
        threeFrequency = freqDict['11']

        bits_list = []
        freq = np.fft.fftfreq(delta, 1 / self.samplerate)
        while index + delta < last_index:
            try:
                window = audio_signal[index:index + delta]
                print(index+delta)
                print(len(window))
                fft_result = np.fft.fft(window)            
                max_idx = np.argmax(np.abs(fft_result))
                max_freq = freq[max_idx]
                max_freq = abs(max_freq)

                distances = np.array([abs(max_freq - zeroFrequency), 
                            abs(max_freq - oneFrequency), 
                            abs(max_freq - twoFrequency), 
                            abs(max_freq - threeFrequency)])

                min_distance = np.argmin(distances)

                if min_distance == 0:
                    bits_list += ['0','0']
                elif min_distance == 1:
                    bits_list += ['0','1']
                elif min_distance == 2:
                    bits_list += ['1','0']
                elif min_distance == 3:
                    bits_list += ['1','1']

                index += delta
                # turn the bits list int a byte list
            except:
                break
        return bits_list

    
    def hamming_decode(self, bits):
        bits = list(map(int,bits)) # list of bits
        if len(bits)<7:
            bits.extend([0] * (7 - len(bits)))
        
        """Decode bits using Hamming(7,4) code."""

        H = np.array([[1, 0, 1, 0, 1, 0, 1],
                      [0, 1, 1, 0, 0, 1, 1],
                      [0, 0, 0, 1, 1, 1, 1]], dtype=bool)
        syndrome = np.dot(H, bits) % 2
        error_bit = int(''.join(map(str, syndrome)), 2)
        if error_bit:
            bits[error_bit - 1] ^= 1
        bits = bits[2:]
        bits.pop(1)
        return bits
    
    def decode_all(self, byte: list) -> np.ndarray:
        bits_list = []
        for bit in byte:
            bits_list.append(self.hamming_decode(bit))
        
        return np.concatenate(bits_list)
    
    def decode_audio(self, audio: np.ndarray) -> (np.ndarray, str):
        """ Decodes the audio into an image and text. Main method of the class 
        :param audio: the audio to be decoded
        :return: the image and the text """

        filtered_signal = filter_signal(audio, self.samplerate, self.min_frequency, self.max_frequency)

        #initial_index = self.find_header(audio, self.headerDuration, 'r')
        
        r_bytes, g_bytes, b_bytes, text_bytes = self.demodulate_audio(filtered_signal,'b')

        r_channel = self.bits_to_image(r_bytes, self.image_width)
        g_channel = self.bits_to_image(g_bytes, self.image_width)
        b_channel = self.bits_to_image(b_bytes, self.image_width)
        text_channel = self.bytes_to_text(text_bytes)

        image = np.dstack([b_channel,g_channel,r_channel])
        return image, text_channel

    def bits_to_image(self, bits_list,img_size=20):
        values = [int(bits_str, 2) for bits_str in bits_list] # list of ints in {1,0} set
        values = np.array(values)
        if (len(values))<img_size**2:
            zeros = np.zeros(img_size**2-len(values))
            values = np.concatenate([values,zeros])
        else:
            values = values[:img_size**2]
        values = values.reshape(img_size,img_size)
        return values

    def bytes_to_text(self, bytes_list: list) -> str:
        """ Decodes the bytes list into text
        :param bytes_list: the bytes list to be decoded
        :return: the decoded text """

        return ''.join(chr(int(byte, 2)) for byte in bytes_list)
    
    def set_freq_dict(self):
        """ Sets the frequency dictionaries for the sender """
        self.freqDict = self.create_freq_dict(minfreq=self.min_frequency, 
                                             bandwidth=self.max_frequency - self.min_frequency, 
                                             n=4)
        

    def freq2bin(self, peaks, channelFreq, bandwidth):
        """ Converts the frequencies to binary """
        # correlate peaks with frequencies in freqDict
        keyFreqs = []
        for peak in peaks:
            keyFreq = int((peak - abs(channelFreq - bandwidth / 2)) // 50)
            keyFreqs.append(keyFreq)

        # convert keyFreqs to binary
        binary = [bin(i)[2:] for i in keyFreqs]
        binary_pairs = [''.join(binary[i:i + 2]) for i in range(0, len(binary), 2)]
        return binary_pairs

    def play_recorded(self):
        """Plays the recorded data."""
        sd.play(self.buffer, self.samplerate)
        sd.wait()
        return

    
    def find_header(self, audio, duration, channel = 'text', reversed=False) -> int:
        """ Finds the header in the audio
        :param audio: the audio to be analyzed
        :param duration: the duration of the header
        :param channel: the channel of the header
        :param reversed: if the header is reversed
        """
        headerF1 = self.min_frequency + self.headerF1
        headerF2 = self.min_frequency + self.headerF2

        tHeader = np.linspace(0, duration, int(self.samplerate * self.headerDuration))

        # Create header using chirp, 
        header = signal.chirp(tHeader, headerF1, self.headerDuration, headerF2, method='linear')

        if reversed:
            audio = np.flip(audio)
            header = np.flip(header)

        # Compute the correlation of the audio and the header
        correlation = np.correlate(audio[0:len(audio)//4], header, mode='valid')

        # Find the index of the maximum correlation
        max_idx = np.argmax(np.abs(correlation))

        return max_idx

    def plot_fft(self, audio):
        """Plots the FFT of the recorded data."""

        self.buffer = audio

        if self.buffer is None:
            print("No data to plot. Please record data first.")
            return

        # Compute FFT
        # fft_result = np.fft.fft(self.buffer[:, 0])  # Use the first channel for FFT
        fft_result = np.fft.fft(self.buffer)
        L = len(fft_result)

        # Compute corresponding frequencies
        freq = np.fft.fftfreq(len(fft_result), 1 / self.samplerate)

        max_idx = np.argmax(np.abs(fft_result))
        max_freq = freq[max_idx]

        # Plot FFT
        plt.figure(figsize=(10, 4))
        plt.semilogx(freq[0:L // 2], np.abs(fft_result)[0:L // 2])
        plt.title("FFT of Recorded Data")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        # Add text about the maximum frequency
        plt.text(max_freq, np.abs(fft_result[max_idx]),
                 'Max at {:.2f} Hz'.format(max_freq),
                 ha='center', va='bottom')

        plt.show()


    def create_freq_dict(self, minfreq: float, bandwidth: float, n: int) -> dict:
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




def bandpass(audio, lowFreq, highFreq, n):
    """ Filters the audio data with a bandpass filter """
    b, a = audio.butter(n, [lowFreq, highFreq], btype='bandpass', fs=44100)
    return audio.filtfilt(b, a, audio)


def bin2str(binaryList) -> str:
    """ Decodes the message from binary to text
    :param binaryList: the binary list to be decoded. """
    return ''.join(chr(int(binary, 2)) for binary in binaryList)


def filter_signal(audio_signal, samplerate, low_freq, high_freq):
    """ Filters the audio signal with a bandpass filter """
    b, a = signal.butter(5, [low_freq, high_freq], btype='bandpass', fs=samplerate, output='ba')
    return signal.filtfilt(b, a, audio_signal)

def plotFFT_window(window, sampleRate):
    
    fft_result = np.fft.fft(window)
    L = len(fft_result)

    # Compute corresponding frequencies
    freq = np.fft.fftfreq(len(fft_result), 1 / sampleRate)

    max_idx = np.argmax(np.abs(fft_result))
    max_freq = freq[max_idx]

    # Plot FFT
    plt.figure(figsize=(10, 4))
    plt.plot(freq[0:L // 2], np.abs(fft_result)[0:L // 2])
    plt.title("FFT of Recorded Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")