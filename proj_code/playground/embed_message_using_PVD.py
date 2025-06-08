# TODO embed message in photos in path using PVD

import copy
from bitstring import BitArray
import jpegio as jio
import numpy as np
import matplotlib.pyplot as plt
import glob
from PVD_steg import PVD
import os

start_dir = '/home/alexmiclea/Documents/Facultate/Licenta/dataset/to_process'
end_dir = '/home/alexmiclea/Documents/Facultate/Licenta/dataset/pvd'

photos = glob.glob(start_dir + '/*.png')
photos = np.sort(photos)
data = []

for (idx,photo) in enumerate(photos):
    model = PVD(photo)

    embedding_capacity = model.get_embedding_capacity()
    # multiply with 0.05 and round so you create random bytes
    embed_size_bytes = round(0.05 * embedding_capacity)
    # # generate 10k bytes and get only required bit amount

    random_bytes = os.urandom(embed_size_bytes)

    image = model.get_pvd_with_embedded_message(random_bytes)

    plt.imsave(end_dir + f'/{2501+idx}.png', image)

# data = np.array(data)
# print(data.shape)
# np.save(end_dir, data)