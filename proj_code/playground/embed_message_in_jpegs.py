import copy
from bitstring import BitArray
import jpegio as jio
import numpy as np
import matplotlib.pyplot as plt
import glob
from DCT_steg import DCT_jpeg
import os


start_dir = '/home/alexmiclea/Documents/Facultate/Licenta/dataset/alaska2/Cover'
end_dir = '/home/alexmiclea/Documents/Facultate/Licenta/dataset/alaska2_coef_bincount/Cover_DCT_jpeg/data.npy'

photos = glob.glob(start_dir + '/*.jpg')
photos = np.sort(photos)
data = []

for (idx,photo) in enumerate(photos):
    model = DCT_jpeg(photo)

    embedding_capacity = model.print_embedding_capacity()

    # multiply with 0.05 and round so you create random bytes
    embed_size_bytes = round(0.05 * embedding_capacity)
    # generate 10k bytes and get only required bit amount

    random_bytes = os.urandom(embed_size_bytes)

    old_dct_coefs, new_dct_coefs = model.helper_get_dct_coefs_embed(random_bytes)

    if (idx % 100 == 0):
        print(idx)

    coef_array = np.zeros((2049))

    unique, counts = np.unique(new_dct_coefs, return_counts=True)
    unique = unique.astype(np.int32)

    for x in zip(unique, counts):
        coef_array[x[0] + 1024] = x[1]

    plt.stem(coef_array)
    plt.show()

    # data.append(coef_array)

#     # min_loc, max_loc = np.min(coef_arrays), np.max(coef_arrays)

#     # if min_loc < minn:
#     #     minn = min_loc

#     # if max_loc > maxx:
#     #     maxx = max_loc

# data = np.array(data)
# print(data.shape)
# np.save(end_dir, data)