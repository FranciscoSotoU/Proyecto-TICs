# This is a sample Python script.
import matplotlib as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Sender import Sender
from Receiver import Receiver

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sender = Sender()
    path = "./data/text.txt"
    sender.load_text(path)
    sender.playText(sender.send_text(sender.dataToFrequency()))
    print("s")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
