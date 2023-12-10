import numpy as np
import sounddevice as sd
import matplotlib
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import pearsonr

class Receiver:
    """ Class that represents the receiver of the communication channel"""

    def __init__(self, channelFreq: float = 600, bandwidth: float = 800):
        self.buffer = None
        # self.channel = channel
        self.samplerate = 44100
        self.freqDuration = 0.05
        self.channelFreq = channelFreq
        self.bandwidth = bandwidth
        self.textFreqDict = create_freq_dict(self.channelFreq, self.bandwidth, 2)
        self.headerFreq = 50

    def listen(self, duration):
        """ Listens to the channel for a message """
        # record audio
        data = sd.rec(frames=int(self.samplerate * duration),
                      samplerate=self.samplerate, channels=1)
        print("Listening")
        sd.wait()
        print("Done listening")
        self.buffer = data
        return data

    def demodulateText(self, audio_signal) -> list:
        """ Demodulates the signal into binary
        :param audio_signal: the signal to be demodulated
        :return: the binary list """

        initial_index = self.find_header(audio_signal, self.headerFreq, self.freqDuration)
        delta = int(self.freqDuration * self.samplerate)
        index = initial_index + delta*3
        last_index = self.find_header(audio_signal, self.headerFreq, self.freqDuration, reversed=True)

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

    def find_header(self, audio, headerFreq, duration, reversed=False):
        """ Finds the header in the audio """
        tHeader = np.linspace(0, duration, int(self.samplerate * duration))
        header = np.concatenate((np.sin(2 * np.pi * 50 * tHeader),
                                 np.sin(2 * np.pi * 100 * tHeader),
                                 np.sin(2 * np.pi * 50 * tHeader)))
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
            correlation_coefficient = np.abs(np.mean(window*header)) #pearsonr(window, header)[0]
            if correlation_coefficient > 0.4:
                if reversed:
                    return len(audio) - index
                return index
            debug_list.append(correlation_coefficient)
            if correlation_coefficient > max_val:
                max_val = correlation_coefficient
                max_idx = index
            index += 1

        if reversed:
            return len(audio) - max_idx
        else:
            return max_idx

    def plot_fft(self):
        """Plots the FFT of the recorded data."""
        if self.buffer is None:
            print("No data to plot. Please record data first.")
            return

        # Compute FFT
        fft_result = np.fft.fft(self.buffer[:, 0])  # Use the first channel for FFT
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



