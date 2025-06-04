import copy
import jpegio as jio
import numpy as np
import matplotlib.pyplot as plt
import glob

folders = ['Cover', 'JMiPOD', 'JUNIWARD', 'Test', 'UERD']

start_dir_fn = lambda x: f'/home/alexmiclea/Documents/Facultate/Licenta/dataset/alaska2/{x}'
end_dir_fn = lambda x: f'/home/alexmiclea/Documents/Facultate/Licenta/dataset/alaska2_coef_bincount/{x}/labels.npy'

for folder in folders:
    print(folder)
    start_dir = start_dir_fn(folder)
    end_dir = end_dir_fn(folder)

    photos = glob.glob(start_dir + '/*.jpg')
    photos = np.sort(photos)
    data = []
    # maxx = 0
    # minn = 1e9

    for (idx,photo) in enumerate(photos):

        if (idx % 100 == 0):
            print(idx)

        jpeg_file = jio.read(photo)
        coef_arrays = copy.deepcopy(jpeg_file.coef_arrays[0])

        coef_array = np.zeros((2049))

        unique, counts = np.unique(coef_arrays, return_counts=True)
        unique = unique.astype(np.int32)

        for x in zip(unique, counts):
            coef_array[x[0] + 1024] = x[1]

        plt.stem(coef_array)
        plt.show()

        # data.append(coef_array)

        # min_loc, max_loc = np.min(coef_arrays), np.max(coef_arrays)

        # if min_loc < minn:
        #     minn = min_loc

        # if max_loc > maxx:
        #     maxx = max_loc

    # data = np.array(data)
    # print(data.shape)
    # np.save(end_dir, data)
# print(minn, maxx)
    # print(np.max(coef_arrays))

    # height = len(coef_arrays_modif)
    # width = len(coef_arrays_modif[0])

    # new_array = np.zeros((height, width), dtype=np.int16)

# # FOARTE important. trebuie modificata matricea care este deja legata de coef_arrays[0]
# # si din acest motiv folosesc [:]. daca nu puneam acolo acel operator, modificarile nu ar fi avut loc
# jpeg_file.coef_arrays[0][:] = new_array

# print(jpeg_file.coef_arrays[0])

# # jpeg_file.image_height = 8
# # jpeg_file.image_width = 8

# jio.write(jpeg_file, 'fisiere_extrase/fmi_2.jpg')