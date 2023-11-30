import numpy as np
import sounddevice as sd
import matplotlib
#matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy.signal as signal



class Receiver:
    """ Class that represents the receiver of the communication channel"""

    def __init__(self):
        self.buffer = None
        #self.channel = channel
        self.samplerate = 44100
        self.duration = 0.2
        self.textFreqDict = freqDict(600, 800, 16)

    def listen(self, duration):
        """ Listens to the channel for a message """
        # record audio
        data = sd.rec(frames=int(self.samplerate*duration),
                      samplerate=self.samplerate, channels=1)
        print("Listening")
        sd.wait()
        print("Done listening")
        self.buffer = data
        return data
    
    def demodulate(self, signal, channelFreq, bandwidth):
        """ Demodulates the signal """
        # bandpass filter
        #signal = bandpass(signal, 50, 1000, 5)
        # find peaks
        index = 0
        delta = int(self.duration*self.samplerate)
        headerID = False

        peaks = []
        while index+delta < len(signal):
            window = signal[index:index+delta]
            wfft = np.fft.fft(window)
            wfft = wfft[len(wfft)//2:-1]
            
            freq = np.fft.fftfreq(len(wfft), 1/self.samplerate)
            max_idx = np.argmax(np.abs(wfft))
            max_freq = freq[max_idx]

            if headerID:
                peaks.append(max_idx)

            if abs(50+30)> max_freq > abs(50-30): # se identifica header
                headerID = True

            index += delta
        
        # correlate peaks with frequencies in freqDict
        keyFreqs = []
        for peak in peaks:
            keyFreq = int((peak - abs(channelFreq - bandwidth/2))//50)
            keyFreqs.append(keyFreq)

        # convert keyFreqs to binary
        binary = [bin(i)[2:] for i in keyFreqs]
        binary_pairs = [''.join(binary[i:i+2]) for i in range(0, len(binary), 2)]
        return binary_pairs


    def decode(self, binaryList):
        """ Decodes the message from binary to text """
        return ''.join(chr(int(binary, 2)) for binary in binaryList)
        

    def play_recorded(self):
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
        freq = np.fft.fftfreq(len(fft_result), 1/self.samplerate)

        max_idx = np.argmax(np.abs(fft_result))
        max_freq = freq[max_idx]

        # Plot FFT
        plt.figure(figsize=(10, 4))
        plt.semilogx(freq[0:L//2], np.abs(fft_result)[0:L//2])
        plt.title("FFT of Recorded Data")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        # Add text about the maximum frequency
        plt.text(max_freq, np.abs(fft_result[max_idx]),
                 'Max at {:.2f} Hz'.format(max_freq),
                 ha='center', va='bottom')

        plt.show()


def freqDict(channelFreq:float, bandwidth:float, n:int) -> dict:
    """ Creates a dictionary of frequencies for the given channel """
    freqDict = {}
    freqs = np.linspace(channelFreq - bandwidth/2 + bandwidth/(2*n), 
                        channelFreq + bandwidth/2 - bandwidth/(2*n), n)
    
    for index, item in enumerate(freqs):
        freqDict[index] = item

    return freqDict

def bandpass(audio, lowFreq, highFreq, n):
    """ Filters the audio data with a bandpass filter """
    b, a = signal.butter(n, [lowFreq, highFreq], btype='bandpass', fs=44100)
    return signal.filtfilt(b, a, audio)
    