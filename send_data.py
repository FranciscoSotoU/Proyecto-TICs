from src.Sender import Sender
import numpy as np
import os
from argparse import ArgumentParser
import sys
import yaml
def send_data(config:dict):
    """
    Sends the data using the Sender class.
    :param min_frequency: minimum frequency
    :param max_frequency: maximum frequency
    :param path_data: path to the data
    :param N: sender ID
    :return: audio wave
    """
    min_frequency = config['min_frequency']
    max_frequency = config['max_frequency']
    path_data = config['path_data']
    N = config['number']
    sender = Sender(min_frequency, max_frequency)
    if N==1:
        path_img = os.path.join(path_data,'1_20_Imagen1.png')
        path_text = os.path.join(path_data,'text.txt')
    elif N==2:
        path_img = os.path.join(path_data,'1_14_Imagen2.png')
        path_text = os.path.join(path_data,'text2.txt')
    sender.load_image(path_img)
    sender.load_text(path_text)

    audio = sender.send_all_data()
    sender.playText(audio)

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs.yaml', help='Path to the config file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f'Config file {args.config} not found')
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(config)
    send_data(config)