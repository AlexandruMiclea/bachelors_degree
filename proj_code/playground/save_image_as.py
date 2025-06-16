from curses.panel import new_panel
from LSB_steg import LSB
from PVD_steg import PVD
from DCT_steg import DCT, DCT_jpeg
from conv_net import ConvNet
from huffman import MessageParser
from aux import Metrics
from phase_encoding import PhaseEncoding

from bitstring import BitArray
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import jpegio as jio
import copy

old_path = '/home/alexmiclea/Documents/Facultate/Licenta/images/airplane/airplane.bmp'
nenw_path = '/home/alexmiclea/Documents/Facultate/Licenta/images/airplane/airplane.jpeg'

open_image = Image.open(old_path).convert("RGB")
image = np.array(open_image)

plt.imsave(nenw_path, image)