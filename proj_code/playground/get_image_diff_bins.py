from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np

dir = '/home/alexmiclea/Documents/Facultate/Licenta/dataset/clean'
photos = glob.glob(dir + '/*.png')
diffs = []

for image_path in photos:

    image = plt.imread(image_path)
    image_read = Image.open(image_path)
    image =  np.array(image_read).astype(np.int32)
    image_h, image_w, color_channels = image.shape

    for c in range(color_channels):
        for i in range(image_h):
            if (i % 2 == 0):
                for j in range(0, image_w - 1, 2):

                    diff = image[i,j,c] -image[i,j+1,c]
                    diffs.append(diff)
                    
                if (image_w % 2 != 0 and i != image_h - 1):

                    diff = image[i, image_w - 1, c] - image[i + 1, image_w - 1, c]
                    diffs.append(diff)
            else:
                for j in range(image_w - image_w % 2 - 1, 0, -2):

                    diff = image[i, j, c] - image[i, j - 1, c]
                    diffs.append(diff)

unique, counts = np.unique(diffs, return_counts=True)
keys = np.argsort(unique)

unique = unique[keys]
counts = counts[keys]

elems_list = np.array(list(zip(unique, counts)))
elems_list = np.array([x for x in elems_list if -7 <= x[0] <= 7]).astype(np.float64)

sum_elems = np.sum(elems_list[:,1])

elems_list[:,1] /= sum_elems

np.save('/home/alexmiclea/Documents/Facultate/Licenta/dataset/dist_clean.npy', elems_list)

plt.stem([x[0] for x in elems_list], [x[1] for x in elems_list])
plt.show()

