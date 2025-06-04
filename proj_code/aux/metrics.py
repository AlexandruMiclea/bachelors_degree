from math import inf
import numpy as np
from scipy.signal import convolve2d

class Metrics:

    @staticmethod
    def get_mse(image_1: np.ndarray, image_2: np.ndarray):
        image_1 = image_1.copy().astype(np.int32)
        image_2 = image_2.copy().astype(np.int32)

        return np.mean((image_1 - image_2)**2)

    @staticmethod
    def get_psnr(image_1: np.ndarray, image_2: np.ndarray):
        image_1 = image_1.copy().astype(np.int32)
        image_2 = image_2.copy().astype(np.int32)

        mse = Metrics.get_mse(image_1, image_2)

        return 10 * np.log10(255**2 / mse) if mse != 0 else inf

    @staticmethod
    def _get_ssim(image_1: np.ndarray, image_2: np.ndarray) -> np.ndarray:
        image_1 = image_1.copy().astype(np.float32)
        image_2 = image_2.copy().astype(np.float32)
        gaussian_distribution = lambda x, mu, sigma:  (1 / np.sqrt(2 * np.pi * sigma**2)) * (np.e ** (-(((x - mu)**2) / (2*sigma**2))))

        values = np.linspace(-5, 5, 11)

        y = gaussian_distribution(values, 0, 1.5)
        y /= np.sum(y)

        # fac produsul extern pentru a obține filtrul 2d
        kernel = np.outer(y, y)

        # fac convoluție ca să urmez fidel implementarea din articolul original
        mu_x = convolve2d(image_1, kernel, mode = 'valid', boundary = 'symm')
        mu_y = convolve2d(image_2, kernel, mode = 'valid', boundary = 'symm')

        sigma_x = np.sqrt(convolve2d(image_1 ** 2, kernel, mode = 'valid', boundary = 'symm') - mu_x ** 2)
        sigma_y = np.sqrt(convolve2d(image_2 ** 2, kernel, mode = 'valid', boundary = 'symm') - mu_y ** 2)
        sigma_xy = convolve2d(image_1 * image_2, kernel, mode = 'valid', boundary = 'symm') - mu_x * mu_y

        # constante luate din articol
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x**2 + sigma_y**2 + C2))

        return ssim

    @staticmethod
    def get_mssim(image_1: np.ndarray, image_2: np.ndarray):
        res = list()

        for c in range(3):
            array = Metrics._get_ssim(image_1[:,:,c], image_2[:,:,c])
            array = np.nan_to_num(array)
            res.append(array)

        return np.mean(res)
    
    @staticmethod
    def _get_score_between_distributions(dist_1, dist_2):
        # dist_1 este distribuția așteptată, dist_2 este cea care este obținută

        F = 0

        for i in range(1,8):
            values_dist_1 = np.array(dist_1[-i][1], dist_1[i][1])
            values_dist_2 = np.array(dist_2[-i][1], dist_2[i][1])

            F += ((values_dist_2 - values_dist_1)**2) / (values_dist_1)
        
        return np.mean(F)

    @staticmethod
    def KL_divergence(dist_1, dist_2):
        score = 0
        for i in range(dist_1.shape[0]):
            score += dist_1[i][1] * np.log(dist_1[i][1] / dist_2[i][1])

        return score

    @staticmethod
    def PVD_detect_if_image_is_manipulated(image:np.ndarray):
        dir_dist = '/home/alexmiclea/Documents/Facultate/Licenta/dataset/dist_clean.npy'

        COMPARISON_DISTRIBUTION = np.load(dir_dist)
        # valoare hardcodată, aleasă să fie puțin sub scorul cel mai mic al unei distribuții pentru o imagine PVD
        THRESHOLD = np.float64(0.2)

        diffs = []

        image =  image.astype(np.int32)
        if (image.ndim == 2):
            image = image[..., np.newaxis]
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

        return Metrics.KL_divergence(COMPARISON_DISTRIBUTION, elems_list) > THRESHOLD
    

    @staticmethod
    def DCT_get_frequencies(coefs_1, coefs_2):

        coefs_1 = coefs_1.flatten()
        coefs_2 = coefs_2.flatten()

        unique, counts = np.unique(coefs_1, return_counts=True)
        keys = np.argsort(unique)

        unique = unique[keys]
        counts = counts[keys]

        elems_list_true = np.array(list(zip(unique, counts)))
        elems_list_true = np.array([x for x in elems_list_true if -32 <= x[0] <= 32]).astype(np.int32)

        unique, counts = np.unique(coefs_2, return_counts=True)
        keys = np.argsort(unique)

        unique = unique[keys]
        counts = counts[keys]

        elems_list_DCT = np.array(list(zip(unique, counts)))
        elems_list_DCT = np.array([x for x in elems_list_DCT if -32 <= x[0] <= 32]).astype(np.int32)

        return elems_list_DCT, elems_list_true