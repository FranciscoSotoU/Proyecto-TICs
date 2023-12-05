import numpy as np
import sounddevice as sd
import matplotlib
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy.signal as signal


class Receiver:
    """ Class that represents the receiver of the communication channel"""

    def __init__(self, channelFreq: float = 600, bandwidth: float = 800):
        self.buffer = None
        # self.channel = channel
        self.samplerate = 44100
        self.duration = 0.1
        self.channelFreq = channelFreq
        self.bandwidth = bandwidth
        self.textFreqDict = create_freq_dict(self.channelFreq, self.bandwidth, 16)
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

    def demodulate(self, audio_signal) -> list:
        """ Demodulates the signal into binary
        :param audio_signal: the signal to be demodulated
        :return: the binary list """

        initial_index = find_header(audio_signal, self.headerFreq, self.duration)
        delta = int(self.duration * self.samplerate)
        index = initial_index + delta
        last_header = len(audio_signal)#index + find_header(audio_signal[index:], self.headerFreq, self.duration)
        freq_dict = create_freq_dict(self.channelFreq, self.bandwidth, 16)
        peaks = []
        while index + delta < last_header:
            window = audio_signal[index:index + delta]
            t = np.linspace(0, self.duration, int(self.samplerate * self.duration))
            max_mean = 0
            idx = 0
            for key in freq_dict:
                freq = freq_dict[key]
                cosine = np.sin(2 * np.pi * freq * t)
                if np.abs(np.mean(window * cosine)) > max_mean:
                    max_mean = np.abs(np.mean(window * cosine))
                    idx = key
            peaks.append(idx)
            index += delta
        binary = [bin(i) for i in peaks]
        bin_list = []
        for item in binary:
            aux_item = item[2:]
            if len(aux_item) == 3:
                aux_item = '0' + aux_item
            if len(aux_item) == 2:
                aux_item = '00' + aux_item
            if len(aux_item) == 1:
                aux_item = '000' + aux_item
            bin_list.append(aux_item)

        final_bin_list = []
        i = 0
        while i + 1 < len(bin_list[1:]):
            final_bin_list.append(bin_list[1:][i] + bin_list[1:][i + 1])
            i += 2
        return final_bin_list

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


def find_header(audio, headerFreq, duration):
    """ Finds the header in the audio """
    t = np.linspace(0, duration, int(44100 * duration))
    cosine = np.sin(2 * np.pi * headerFreq * t)
    index = 0
    delta = int(duration * 44100)
    while index + delta < len(audio):
        window = audio[index:index + delta]
        if np.abs(np.mean(window * cosine)) > 0.25:
            return index
        index += 1
    return -1
