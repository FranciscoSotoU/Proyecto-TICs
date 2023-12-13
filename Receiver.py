import numpy as np
import sounddevice as sd
import matplotlib
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import pearsonr
from scipy.io import wavfile


class Receiver:
    """ Class that represents the receiver of the communication channel"""

    def __init__(self, channelFreq: float = 600, bandwidth: float = 800):
        self.buffer = None
        # self.channel = channel
        self.samplerate = 44100
        self.freqDuration = 0.005
        self.headerDuration = self.freqDuration * 20  # 1 second header
        self.channelFreq = channelFreq
        self.bandwidth = bandwidth
        self.textFreqDict = create_freq_dict(self.channelFreq, self.bandwidth, 2)
        self.headerF1 = 80
        self.headerF2 = 500
        self.textLength = len("¡Laboratorio de Tecnologías de Información y de Comunicación EL5207! Transmisor número 1, primavera 2023.")

    def listen(self, duration):
        """ Listens to the channel for a message """
        # record audio
        data = sd.rec(frames=int(self.samplerate * duration),
                      samplerate=self.samplerate, channels=1)
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
        delta = int(self.freqDuration * self.samplerate)
        # index = initial_index + int(self.headerDuration * self.samplerate)
        index = self.header_correlation(audio_signal, 5)
        print("indice inicial (después del header)", index)
        # last_index = self.find_header(audio_signal, self.headerDuration, reversed=True)
        last_index = index + self.textLength * 8 * int(self.freqDuration * self.samplerate)
        
        bits_list = []
        while index + delta <= last_index:
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
    

    def demodText_fft(self, audio_signal) -> list:
        """ Demodulates the signal into binary
        :param audio_signal: the signal to be demodulated
        :return: the binary list """

        # initial_index = self.find_header(audio_signal, self.headerDuration)
        delta = int(self.freqDuration * self.samplerate)
        # index = initial_index + int(self.headerDuration * self.samplerate)
        index = self.header_correlation(audio_signal, 5)
        # last_index = self.find_header(audio_signal, self.headerDuration, reversed=True)
        last_index = index + self.textLength * 8 * int(self.freqDuration * self.samplerate)

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

    def bits_to_image(self, bits_list):
        values = [int(bits_str, 2) for bits_str in bits_list]
        values = np.array(values)
        sqr_shape = int(np.sqrt(len(values)))
        values = values.reshape(sqr_shape,sqr_shape)
        return values

    def bits_to_text(self, bits_list):
        # Convert list of bits into a string
        bits_str = ''.join(str(bit) for bit in bits_list)
        # Split the string into chunks of 8 bits
        chunks = [bits_str[i:i + 8] for i in range(0, len(bits_str), 8)]
        # Convert each chunk into a character
        chars = [chr(int(chunk, 2)) for chunk in chunks]
        # Join all characters together to form the final text
        text = ''.join(chars)
        return text
    
    def set_freq_bands(self,min_frequency,max_frequency):
        """ Sends all the data """
        range = max_frequency - min_frequency
        bands_range = range/5
        self.text_band = min_frequency + bands_range
        self.r_band = min_frequency + 2*bands_range
        self.g_band = min_frequency + 3*bands_range
        self.b_band = min_frequency + 4*bands_range

        

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

    def find_header(self, audio, duration, reversed=False):
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
    

    def header_correlation(self, audio, seconds=10):
        """ Returns the audio data after the header"""

        # Search in the first 10 seconds of audio
        audio = audio[0:self.samplerate*seconds]
        tHeader = np.linspace(0, self.headerDuration, int(self.samplerate * self.headerDuration))
        header = signal.chirp(tHeader, self.headerF1, self.headerDuration, self.headerF2, method='linear')

        correlation = np.correlate(audio, header, mode='full')
        max_idx = np.argmax(correlation)

        displacement = max_idx - len(header) + 1

        initial_index = displacement + len(header)

        return initial_index

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


def create_freq_dict(channelFreq: float, bandwidth: float, n: int) -> dict:
    """ Creates a dictionary of frequencies for the given channel """
    freq_dict = {}
    freqs = np.linspace(channelFreq - bandwidth / 2 + bandwidth / (2 * n),
                        channelFreq + bandwidth / 2 - bandwidth / (2 * n), n)

    for index, item in enumerate(freqs):
        freq_dict[index] = item

    return freq_dict


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
