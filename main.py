# This is a sample Python script.
import matplotlib as plt
import pickle
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Sender import Sender
from Receiver import Receiver

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    receiver = Receiver()
    with open('./data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    text = receiver.decode(receiver.demodulate(data, 600, 800))
    print(text)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
