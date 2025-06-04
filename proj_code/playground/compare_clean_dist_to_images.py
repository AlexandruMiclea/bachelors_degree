from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np


def KL_divergence(dist_1, dist_2):
    score = 0
    for i in range(dist_1.shape[0]):
        score += dist_1[i][1] * np.log(dist_1[i][1] / dist_2[i][1])

    return score

def get_score_between_distributions(dist_1, dist_2):
    # dist_1 is the clear distribution, dist_2 is the image distribution
    # we get the 1/2 value from the 

    F = 0

    for i in range(1,8):
        values_dist_1 = np.array(dist_1[-i][1], dist_1[i][1])
        values_dist_2 = np.array(dist_2[-i][1], dist_2[i][1])

        F += ((values_dist_2 - values_dist_1)**2) / (values_dist_1)
    
    return np.mean(F)

dir_clean = '/home/alexmiclea/Documents/Facultate/Licenta/dataset/clean'

dir_pvd = '/home/alexmiclea/Documents/Facultate/Licenta/dataset/pvd'

dir_dist = '/home/alexmiclea/Documents/Facultate/Licenta/dataset/dist_clean.npy'

COMPARISON_DISTRIBUTION = np.load(dir_dist)

photos_clean = glob.glob(dir_clean + '/*.png')
photos_pvd = glob.glob(dir_pvd + '/*.png')

clean_scores = []
pvd_scores = []

for image_path in photos_clean:
    diffs = []

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

    clean_scores.append(KL_divergence(COMPARISON_DISTRIBUTION, elems_list))

for image_path in photos_pvd:
    diffs = []

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

    pvd_scores.append(KL_divergence(COMPARISON_DISTRIBUTION, elems_list))

print(np.mean(clean_scores))
print(np.mean(pvd_scores))

min_pvd = np.min(pvd_scores)
print(min_pvd)

print(np.sum(clean_scores >= min_pvd))

space = np.arange(1,2501,1)


plt.plot(space, clean_scores, 'bo')
plt.plot(space + 2500,pvd_scores, 'r.')
plt.show()