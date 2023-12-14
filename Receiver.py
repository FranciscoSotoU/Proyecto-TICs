import numpy as np
import sounddevice as sd
import matplotlib
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import pearsonr
from scipy.io import wavfile
import cv2


class Receiver:
    """ Class that represents the receiver of the communication channel"""

    def __init__(self,  min_frequency, max_frequency):
        self.buffer = None
        # self.channel = channel
        self.samplerate = 44100
        self.freqDuration = 0.01
        self.freq_text_duration = 0.01*1.75
        self.headerDuration = 1 # 1 second header
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency
        #self.textFreqDict = create_freq_dict(self.channelFreq, self.bandwidth, 2)
        self.headerF1 = min_frequency
        self.headerF2 = max_frequency
        self.set_freq_bands()
        self.set_freq_dicts()
        self.textLength = 105
        self.image_width = 20; 
        self.image_bit_size = self.image_width**2 * 8 
        self.text_bit_size = self.textLength*8  

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

    def demodulateText(self, audio_signal) -> list:
        """ Demodulates the signal into binary
        :param audio_signal: the signal to be demodulated
        :return: the binary list """

        initial_index = self.find_header(audio_signal, self.headerDuration)
        delta = int(self.freq_text_duration * self.samplerate)
        index = initial_index + int(self.headerDuration * self.samplerate)
        # index = self.find_header(audio_signal, 10) + int(self.headerDuration * self.samplerate)
        print("indice inicial (después del header)", index)
        # last_index = self.find_header(audio_signal, self.headerDuration, reversed=True)
        last_index = index + self.textLength * 8 * int(self.freq_text_duration * self.samplerate)
        
        bits_list = []
        while index + delta <= last_index:
            window = audio_signal[index:index + delta]
            t = np.linspace(0, self.freq_text_duration, int(self.samplerate * self.freq_text_duration))
            max_mean = 0
            idx = 0
            for key in self.textFreqDict:
                freq = self.textFreqDict[key]
                cosine = np.sin(2 * np.pi * freq * t)
                if np.abs(np.mean(window * cosine)) > max_mean:
                    max_mean = np.abs(np.mean(window * cosine))
                    idx = key
            bits_list.append(idx)
            index += delta
            # turn the bits list int a byte list

        bits_list = self.decode_all(bits_list)

        return bits_list    
    

    def demodText_fft(self, audio_signal) -> list:
        """ Demodulates the signal into binary
        :param audio_signal: the signal to be demodulated
        :return: the binary list """

        # initial_index = self.find_header(audio_signal, self.headerDuration)
        delta = int(self.freq_text_duration * self.samplerate)
        # index = initial_index + int(self.headerDuration * self.samplerate)
        index = self.header_correlation(audio_signal, 10)
        # last_index = self.find_header(audio_signal, self.headerDuration, reversed=True)
        last_index = index + self.textLength * 8 * int(self.freq_text_duration * self.samplerate)

        bits_list = []
        while index + delta <= last_index:
            window = audio_signal[index:index + delta]

            fft_result = np.fft.fft(window)
            L = len(fft_result)
            freq = np.fft.fftfreq(len(fft_result), 1 / self.samplerate)
            max_idx = np.argmax(np.abs(fft_result))
            max_freq = freq[max_idx]
            max_freq = abs(max_freq)

            # Subtract max_freq from each value in self.textFreqDict
            differences = {key: np.abs(value - max_freq) for key, value in self.textFreqDict.items()}

            # Find the key with the smallest difference
            closest_key = min(differences, key=differences.get)

            # Now closest_key is the key in self.textFreqDict whose value is closest to max_freq
            
                    
            bits_list.append(closest_key)

            index += delta
            # turn the bits list int a byte list
        return bits_list
    
    
    
    def demodulateImage(self, audio_signal):
        """ Demodulates the signal into binary
        :param audio_signal: the signal to be demodulated
        :return: the binary list """

        initial_index = self.find_header(audio_signal, self.headerDuration)
        print('header')
        delta = int(self.freqDuration * self.samplerate)
        index = initial_index + int(self.headerDuration * self.samplerate)
        last_index = self.find_header(audio_signal, self.headerDuration, reversed=True)

        bits_list = []
        while index + delta < last_index:
            window = audio_signal[index:index + delta]
            t = np.linspace(0, self.freqDuration, int(self.samplerate * self.freqDuration))
            max_mean = 0
            idx = 0
            for key in self.textFreqDict:
                freq = self.textFreqDict[key]
                cosine = np.sin(2 * np.pi * freq * t)
                if np.abs(np.mean(window * cosine)) > max_mean:
                    max_mean = np.abs(np.mean(window * cosine))
                    idx = key
            bits_list.append(idx)
            index += delta
            # turn the bits list int a byte list
        return bits_list
    
    def decode_audio(self, audio_signal: np.ndarray, color='text', initial_index=None) -> (int, int, list):
        """ Decodes the audio signal into binary
        :param audio_signal: the audio signal to be decoded
        :param color: the color of the audio signal
        :param initial_index: the initial index of the signal
        :param last_index: the last index of the signal
        :return: the initial index, the last index and the string bytes list"""

        if not initial_index:
            initial_index = self.find_header(audio_signal, self.headerDuration,color)
            print(initial_index)
        delta = int(self.freqDuration * self.samplerate)
        
        index = initial_index + int(self.headerDuration * self.samplerate)
        
        bits_list = []
        t = np.linspace(0, self.freqDuration, int(self.samplerate * self.freqDuration))


        if color== 'r':
            FreqDict = self.redFreqDict
            last_index = int(self.image_bit_size * 1.75*self.samplerate*self.freqDuration)
        elif color == 'g':
            FreqDict = self.greenFreqDict
            last_index = int(self.image_bit_size * 1.75*self.samplerate*self.freqDuration)
        elif color == 'b':
            FreqDict = self.blueFreqDict
            last_index = int(self.image_bit_size * 1.75*self.samplerate*self.freqDuration)
        else:
            FreqDict = self.textFreqDict
            last_index = int(self.text_bit_size * 1.75*self.samplerate*self.freq_text_duration)

        while index + delta < last_index + index:

            window = audio_signal[index:index + delta]
            freqs = np.array(list(FreqDict.values()))
            cosines = np.sin(2 * np.pi * freqs[:, None] * t)
            means = np.abs(np.mean(window * cosines, axis=1))
            idx = np.argmax(means)
            bits_list.append(list(FreqDict.keys())[idx])
            index += delta
        bits_list  = np.array(bits_list)
        grouped_values = [''.join(str(bit) for bit in bits_list[i:i+7]) for i in range(0, len(bits_list), 7)] # list of strings of 7 bits
        bits_list_decoded = self.decode_all(grouped_values)

        grouped_values = [''.join(str(bit) for bit in bits_list_decoded[i:i+8]) for i in range(0, len(bits_list_decoded), 8)]
        return initial_index, last_index, grouped_values
    
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
    
    def demux_audio(self,audio):

        text_audio = filter_signal(audio,self.samplerate,self.text_band-self.bands_range*0.25,self.text_band+self.bands_range*1.5)
        r_audio =  filter_signal(audio,self.samplerate,self.r_band-self.bands_range*0.25,self.r_band+self.bands_range*1.5)
        #print(r_audio)
        g_audio =  filter_signal(audio,self.samplerate,self.g_band-self.bands_range*0.25,self.g_band+self.bands_range*1.5)
        b_audio =  filter_signal(audio,self.samplerate,self.b_band-self.bands_range*0.25,self.b_band+self.bands_range*1.5)

        #initial_index = self.find_header(audio, self.headerDuration, 'r')
        
        initial_index, _, b_bits = self.decode_audio(b_audio,'b')
        _,_,r_bits = self.decode_audio(r_audio,'r',initial_index)
        _,_,g_bits = self.decode_audio(g_audio,'g',initial_index)
        _,_,text_bits = self.decode_audio(text_audio,'text',initial_index)


        r_channel = self.bits_to_image(r_bits, self.image_width)
        g_channel = self.bits_to_image(g_bits, self.image_width)
        b_channel = self.bits_to_image(b_bits, self.image_width)
        text_channel = self.bits_to_text(text_bits)

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

    def bits_to_text(self, bits_list):
        values = [int(bits_str, 2) for bits_str in bits_list] # list of ints in {1,0} set
        values = np.array(values)
        return values
    

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

    def find_header2(self, audio, duration, reversed=False):
        """ Finds the header in the audio """
        tHeader = np.linspace(0, duration, int(self.samplerate * self.headerDuration))

        # Create header using chirp, 
        header = signal.chirp(tHeader, self.headerF1, self.headerDuration, self.headerF2, method='linear')

        if reversed:
            audio = np.flip(audio)
            header = np.flip(header)

        index = 0
        delta = len(header)
        max_val = -np.inf
        max_idx = 0
        debug_list = []
        while index + delta < len(audio):
            window = audio[index:index + delta]
            correlation_coefficient = np.abs(np.mean(window * header))  # pearsonr(window, header)[0]
            # if correlation_coefficient > 0.4:
            #     if reversed:
            #         return len(audio) - index
            #     return index
            debug_list.append(correlation_coefficient)
            if correlation_coefficient > max_val:
                max_val = correlation_coefficient
                max_idx = index
            index += 1

        # if reversed:
        #     return len(audio) - max_idx
        # else:
        #     return max_idx
        return max_idx
    
    def find_header(self, audio, duration, color = 'text', reversed=False) -> int:
        """ Finds the header in the audio
        :param audio: the audio to be analyzed
        :param duration: the duration of the header
        :param color: the color of the header
        :param reversed: if the header is reversed
        """

        if color == 'r':
            headerF1 = self.headerF1 + self.r_band
            headerF2 = self.headerF2 + self.r_band
        elif color == 'g':
            headerF1 = self.headerF1 + self.g_band
            headerF2 = self.headerF2 + self.g_band
        elif color == 'b':
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
        correlation = np.correlate(audio, header, mode='valid')

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
    freqs = np.linspace(minfreq-10, bandwidth + minfreq-10, n)
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