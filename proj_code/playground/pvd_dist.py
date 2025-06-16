from LSB_steg import LSB
from PVD_steg import PVD
from DCT_steg import DCT, DCT_jpeg
from conv_net import ConvNet
from huffman import MessageParser
from aux import Metrics
from phase_encoding import PhaseEncoding
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from bitstring import BitArray
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import jpegio as jio
import copy
import timeit
import scipy.io.wavfile as wavfile

x = np.load("/home/alexmiclea/Documents/Facultate/Licenta/dataset/dist_clean.npy")
print(x.shape)

plt.figure(figsize=(6,4))
plt.xlabel("Valoarea diferenței între doi pixeli adiacenți")
plt.ylabel("Probabilitatea de apariție")
plt.stem(x[:,0], x[:,1])
plt.show()