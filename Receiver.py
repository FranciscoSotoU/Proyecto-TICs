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
        self.freq_text_duration = 0.01
        # self.freq_text_duration = 0.01*1.75
        self.headerDuration = 1 # 1 second header
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency
        self.headerF1 = 200
        self.headerF2 = 500
        self.set_freq_bands()
        self.set_freq_dicts()
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
            print(initial_index)
        
        index = initial_index + int(self.headerDuration * self.samplerate)
        
        bits_list = []
        if channel == 'text':
            t = np.linspace(0, self.freq_text_duration, int(self.samplerate * self.freq_text_duration))
            delta = int(self.freq_text_duration * self.samplerate)
        else:
            t = np.linspace(0, self.freqDuration, int(self.samplerate * self.freqDuration))
            delta = int(self.freqDuration * self.samplerate)


        if channel== 'r':
            FreqDict = self.redFreqDict
            last_index = int(self.image_bit_size * 1.75*self.samplerate*self.freqDuration)

        elif channel == 'g':
            FreqDict = self.greenFreqDict
            last_index = int(self.image_bit_size * 1.75*self.samplerate*self.freqDuration)

        elif channel == 'b':
            FreqDict = self.blueFreqDict
            last_index = int(self.image_bit_size * 1.75*self.samplerate*self.freqDuration)

            
        else:
            FreqDict = self.textFreqDict
            last_index = int(self.text_bit_size * 1.75*self.samplerate*self.freq_text_duration)
            
        
        bits_list = self.demodText_fft(audio_signal, index, channel, FreqDict)
        bits_list= np.array(bits_list)
        grouped_values = [''.join(str(bit) for bit in bits_list[i:i+7]) for i in range(0, len(bits_list), 7)] # list of strings of 7 bits
        bits_list_decoded = self.decode_all(grouped_values)

        grouped_values = [''.join(str(bit) for bit in bits_list_decoded[i:i+8]) for i in range(0, len(bits_list_decoded), 8)] #return as bytes
        return initial_index, last_index, grouped_values
    
    
    def demodText_fft(self, audio_signal, index, channel, freqDict) -> list:
        """ Demodulates the signal into binary
        :param audio_signal: the signal to be demodulated
        :return: the binary list """

        # initial_index = self.find_header(audio_signal, self.headerDuration)

        if channel == 'text':
            delta = int(self.freq_text_duration * self.samplerate)
            last_index = int(self.text_bit_size * 1.75*self.samplerate*self.freq_text_duration)
        else:
            delta = int(self.freqDuration * self.samplerate)
            last_index = int(self.image_bit_size * 1.75*self.samplerate*self.freqDuration)

        last_index = last_index + index
        zeroFrequency = freqDict[0]
        oneFrequency = freqDict[1]
        bits_list = []
        freq = np.fft.fftfreq(delta, 1 / self.samplerate)
        while index + delta <= last_index:
            window = audio_signal[index:index + delta]

            fft_result = np.fft.fft(window)            
            max_idx = np.argmax(np.abs(fft_result))
            max_freq = freq[max_idx]
            max_freq = abs(max_freq)

            distanceToZero = abs(max_freq - zeroFrequency)
            distanceToOne = abs(max_freq - oneFrequency)

            if distanceToZero < distanceToOne:
                bits_list.append(0)
            else:
                bits_list.append(1)

            index += delta
            # turn the bits list int a byte list
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

        text_audio = filter_signal(audio,self.samplerate,self.text_band-self.bands_range*0.25,self.text_band+self.bands_range*1.5)
        r_audio =  filter_signal(audio,self.samplerate,self.r_band-self.bands_range*0.25,self.r_band+self.bands_range*1.5)
        #print(r_audio)
        g_audio =  filter_signal(audio,self.samplerate,self.g_band-self.bands_range*0.25,self.g_band+self.bands_range*1.5)
        b_audio =  filter_signal(audio,self.samplerate,self.b_band-self.bands_range*0.25,self.b_band+self.bands_range*1.5)

        #initial_index = self.find_header(audio, self.headerDuration, 'r')
        
        initial_index, _, b_bits = self.demodulate_audio(b_audio,'b')
        _,_,r_bits = self.demodulate_audio(r_audio,'r',initial_index)
        _,_,g_bits = self.demodulate_audio(g_audio,'g',initial_index)
        _,_,text_bytes = self.demodulate_audio(text_audio,'text',initial_index)
        self.textData = text_bytes
        self.redBits = r_bits
        self.greenBits = g_bits
        self.blueBits = b_bits

        r_channel = self.bits_to_image(r_bits, self.image_width)
        g_channel = self.bits_to_image(g_bits, self.image_width)
        b_channel = self.bits_to_image(b_bits, self.image_width)
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
    

    def set_freq_bands(self):
        """ Sends all the data """
        range = self.max_frequency - self.min_frequency
        self.bands_range = range/5
        self.text_band = self.min_frequency + self.bands_range
        self.r_band = self.min_frequency + 2*self.bands_range
        self.g_band = self.min_frequency + 3*self.bands_range
        self.b_band = self.min_frequency + 4*self.bands_range
        self.bands_delta = self.bands_range//2

    def set_freq_dicts(self):
        """ Sets the frequency dictionaries for the sender """
        self.textFreqDict = create_freq_dict(self.text_band, self.bands_range, 2)
        self.redFreqDict = create_freq_dict(self.r_band, self.bands_range, 2)
        self.greenFreqDict = create_freq_dict(self.g_band, self.bands_range, 2)
        self.blueFreqDict = create_freq_dict(self.b_band, self.bands_range, 2)

        

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

        if channel == 'r':
            headerF1 = self.headerF1 + self.r_band
            headerF2 = self.headerF2 + self.r_band
        elif channel == 'g':
            headerF1 = self.headerF1 + self.g_band
            headerF2 = self.headerF2 + self.g_band
        elif channel == 'b':
            headerF1 = self.headerF1 + self.b_band
            headerF2 = self.headerF2 + self.b_band
        else:
            headerF1 = self.headerF1
            headerF2 = self.headerF2

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

    def set_freq_bands(self):
        """ Sends all the data """

        range = self.max_frequency - self.min_frequency
        self.bands_range = (range/4)*0.5
        real_band_range = range/4
        self.text_band = self.min_frequency 
        self.r_band = self.min_frequency + 1*real_band_range
        self.g_band = self.min_frequency + 2*real_band_range
        self.b_band = self.min_frequency + 3*real_band_range


def create_freq_dict(minfreq: float, bandwidth: float, n: int) -> dict:
    """ Creates a dictionary of frequencies for the given channel """
    freqDict = {}
    freqs = np.linspace(minfreq, bandwidth + minfreq, n)
    for index, item in enumerate(freqs):
        freqDict[index] = item

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