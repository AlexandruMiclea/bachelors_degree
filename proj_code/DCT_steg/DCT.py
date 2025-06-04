import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dctn, idctn
from bitstring import BitArray

Q_luminance = [[16, 11, 10, 16, 24, 40, 51, 61],
               [12, 12, 14, 19, 26, 28, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56],
               [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77],
               [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]]

Q_chrominance = [[17, 18, 24, 47, 99, 99, 99, 99],
                 [18, 21, 26, 66, 99, 99, 99, 99],
                 [24, 26, 56, 99, 99, 99, 99, 99],
                 [47, 66, 99, 99, 99, 99, 99, 99],
                 [99, 99, 99, 99, 99, 99, 99, 99],
                 [99, 99, 99, 99, 99, 99, 99, 99],
                 [99, 99, 99, 99, 99, 99, 99, 99],
                 [99, 99, 99, 99, 99, 99, 99, 99]]

RGB_YCbCr = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
YCbCr_RGB = np.array([[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]])

class DCT:
    def __init__(self, file_path):
        self.file_path = file_path

    def _read_image(self) -> np.ndarray:

        image_read = Image.open(self.file_path).convert('RGB')
        return np.array(image_read)

    def _rgb_to_ycbcr(self, original_image: np.ndarray) -> np.ndarray:

        original_image = original_image.astype(np.int32)

        X_YCbCr = original_image @ RGB_YCbCr.T
        X_YCbCr[:, :, 1] += 128
        X_YCbCr[:, :, 2] += 128

        X_YCbCr = np.clip(X_YCbCr, 0, 255).astype(np.int32)

        return X_YCbCr

    def _ycbcr_to_rgb(self, original_image: np.ndarray) -> np.ndarray:
        original_image = original_image.astype(np.int32)

        original_image[:, :, 1] -= 128
        original_image[:, :, 2] -= 128
        X_RGB = original_image @ YCbCr_RGB.T

        X_RGB = np.clip(X_RGB, 0.0, 255.0).astype(np.int32)

        return X_RGB

    def _get_color_channel_mean(self, color_channel: np.ndarray) -> np.ndarray:
        h, w = color_channel.shape

        w_mean = np.ceil(w / 2).astype(np.int32)
        h_mean = np.ceil(h / 2).astype(np.int32)

        return_mean = np.zeros((h_mean, w_mean))

        for i in range(0, h, 2):
            for j in range(0, w, 2):
                mean_patch = np.mean(color_channel[i:i + 2, j:j + 2])
                return_mean[i // 2, j // 2] = mean_patch

        return_mean = return_mean.astype(np.int32)
        return return_mean

    def _reverse_color_channel_mean(self, color_channel: np.ndarray) -> np.ndarray:
        h, w = color_channel.shape

        return_image = np.zeros((h * 2, w * 2))

        for i in range(0, h * 2):
            for j in range(0, w * 2):
                return_image[i, j] = color_channel[i // 2, j // 2]

        return return_image

    # am ales 16 ca dimensiune de padding pentru că vreau să țin cont de media de 2x2 pe canalele de culoare
    # astfel încât la final toate matricile mele au dimensiuni multiplu de 8
    def _pad_image(self, original_image: np.ndarray) -> tuple[int, int, np.ndarray]:
        h_original, w_original, _ = original_image.shape
        new_image = original_image.copy()

        # pixeli în jos
        if h_original % 16 != 0:
            rows = (16 - (h_original % 16))
            for i in range(rows):
                new_image = np.insert(new_image, h_original + i, new_image[h_original + i - 1, :, :], axis=0)

        # pixeli la dreapta
        if w_original % 16 != 0:
            cols = (16 - (w_original % 16))
            for i in range(cols):
                new_image = np.insert(new_image, w_original + i, new_image[:, w_original + i - 1, :], axis=1)

        return h_original, w_original, new_image

    def _trim_image(self, h_orig: int, w_orig: int, image: np.ndarray) -> np.ndarray:
        return image[:h_orig, :w_orig]

    def _get_quantized_dct_coeffs(self, image: np.ndarray, Q_matrix: np.ndarray) -> np.ndarray:
        image = image.astype(np.float64)
        assert image.shape == (8,8)
        image = image - 128

        coefs_block = dctn(image, norm = "ortho")
        encoded_block = (coefs_block / Q_matrix).astype(np.int32)

        return encoded_block

    def _get_zigzagged_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.fliplr(matrix)
        diagonals = [np.diagonal(matrix, offset=i) for i in range(7,-8, -1)]

        for i in range(0, 16, 2):
            diagonals[i] = np.flip(diagonals[i])

        result = np.concat(diagonals)
        return result
    
    def _zigzag_list(self, coefs_list: np.ndarray) -> np.ndarray:
        assert(coefs_list.shape[0] == 64)

        diagonals = [coefs_list[(n * (n + 1)) // 2: ((n * (n + 1)) // 2) + (n+1)] for n in range(0,8)]
        diagonals.extend([coefs_list[64 - (n * (n + 1)) // 2: 64 - ((n * (n + 1)) // 2) + n] for n in range(7,0,-1)])
        diagonals.reverse()

        ret_matrix = np.zeros((8,8))

        for idx, diag in enumerate(diagonals):
            if (idx % 2 == 0):
                diag = diag[::-1]

            ret_matrix += np.fliplr(np.diag(diag, idx - 7))

        return ret_matrix.astype(np.int32)

    def _get_embedding_capacity(self, coeffs_list: np.ndarray) -> np.int32:
        ans = -1
        for elem in coeffs_list:
            if elem != 0 and elem != 1:
                ans += 1
        if (coeffs_list[0] == 0 or coeffs_list[0] == 1):
            ans += 1

        return ans
    
    def _get_matrix_embedding_capacity(self, coeffs_matrix: np.ndarray) -> np.int32:
        image_h, image_w = coeffs_matrix.shape
        ans = 0

        for i in range(0, image_h // 8):
            for j in range(0, image_w // 8):
                block_start_h = i * 8
                block_start_w = j * 8
                block = coeffs_matrix[block_start_h:block_start_h + 8, block_start_w:block_start_w + 8]
                ans += self._get_embedding_capacity(block.flatten())
        return ans
    
    def _embed_bits_in_dct_block(self, dct_block: np.ndarray, message_bits: BitArray):
        coeffs_list = self._get_zigzagged_matrix(dct_block)
        
        ret_coeffs_list = coeffs_list.copy()
        payload_length = len(message_bits)
        bit_pos = 0

        for (idx, elem) in enumerate(coeffs_list):
            if idx == 0: continue
            if bit_pos == payload_length:
                break
            if elem != 0 and elem != 1:
                elem_modif = (elem & (-2)) | int(message_bits[bit_pos])
                ret_coeffs_list[idx] = elem_modif
                bit_pos += 1

        return self._zigzag_list(ret_coeffs_list)

    def _extract_bits_in_dct_block(self, dct_block: np.ndarray):
        message_bits = BitArray()

        coeffs_list = self._get_zigzagged_matrix(dct_block)

        for (idx, elem) in enumerate(coeffs_list):
            if idx == 0: continue
            if elem != 0 and elem != 1:
                bit = (elem & 1)
                message_bits += BitArray(bin=str(int(bit)))
        
        return message_bits

    def _embed_message(self, coeffs_matrix: np.ndarray, message_bytes: bytes) -> np.ndarray:

        total_capacity = self._get_matrix_embedding_capacity(coeffs_matrix)

        payload_bits = BitArray(message_bytes).bin
        assert(len(payload_bits) <= total_capacity)

        image_h, image_w = coeffs_matrix.shape
        return_image = np.zeros((image_h, image_w)).astype(np.int32)

        for i in range(0, image_h // 8):
            for j in range(0, image_w // 8):
                block_start_h = i * 8
                block_start_w = j * 8
                block = coeffs_matrix[block_start_h:block_start_h + 8, block_start_w:block_start_w + 8]

                if len(payload_bits) > 0:
                    len_block_bits = self._get_embedding_capacity(block.copy().flatten())

                    if (len(payload_bits) <= len_block_bits):
                        new_block = self._embed_bits_in_dct_block(block, payload_bits)
                        payload_bits = BitArray()
                    else:
                        block_bits = payload_bits[:len_block_bits]
                        new_block = self._embed_bits_in_dct_block(block, block_bits)
                        payload_bits = payload_bits[len_block_bits:]

                    return_image[block_start_h:block_start_h + 8, block_start_w:block_start_w + 8] = new_block
                else:
                    return_image[block_start_h:block_start_h + 8, block_start_w:block_start_w + 8] = block

        return return_image
    
    def _extract_message(self, coeffs_matrix: np.ndarray) -> BitArray:

        image_h, image_w = coeffs_matrix.shape
        message_bits = BitArray()

        for i in range(0, image_h // 8):
            for j in range(0, image_w // 8):
                block_start_h = i * 8
                block_start_w = j * 8
                block = coeffs_matrix[block_start_h:block_start_h + 8, block_start_w:block_start_w + 8]

                message_bits += self._extract_bits_in_dct_block(block)

        return message_bits

    def _apply_dct(self, image: np.ndarray, Q_matrix: np.ndarray, mode: str, return_capacity: bool = False) -> np.ndarray | tuple[np.ndarray, bool]:
        assert(mode in ['forward','backward'])
        total_capacity = 0
        image = image.astype(np.int32)

        image_h, image_w = image.shape
        return_image = np.zeros((image_h, image_w)).astype(np.int32)

        for i in range(0, image_h // 8):
            for j in range(0, image_w // 8):
                block_start_h = i * 8
                block_start_w = j * 8
                block = image[block_start_h:block_start_h + 8, block_start_w:block_start_w + 8]
                if mode == 'forward':
                    block_dct = self._get_quantized_dct_coeffs(block, Q_matrix)
                else:
                    block_dct = self._recover_from_dct_coeffs(block, Q_matrix)

                return_image[block_start_h:block_start_h + 8, block_start_w:block_start_w + 8] = block_dct

        if return_capacity:
            return return_image, total_capacity
        else:
            return return_image
        
    def _recover_from_dct_coeffs(self, coeffs: np.ndarray, Q_matrix: np.ndarray) -> np.ndarray:
        coeffs = coeffs.astype(np.float64)
        assert coeffs.shape == (8, 8)

        coeffs *= Q_matrix
        return_image = idctn(coeffs, norm = "ortho").astype(np.int32)
        return_image += 128

        return_image = np.clip(return_image, 0.0, 255.0).astype(np.int32)

        return return_image

    @staticmethod
    def histogram_of_dct_coeffs(image: np.ndarray) -> None:
        image = image.astype(np.int32)

        image = image.flatten()
        image = image[np.abs(image) < 128]
        max_y = np.max(np.bincount(np.abs(image))) // 8
        print(max_y)

        plt.figure()
        plt.ylim((0, max_y))
        plt.xlim((-128, 128))
        plt.hist(image, bins=257)
        # plt.yscale('log')
        plt.show()

    def get_dct_compressed_image(self):
        image = self._read_image()
        
        orig_h, orig_w, resize_image = self._pad_image(image)
        
        ycbcr_image = self._rgb_to_ycbcr(resize_image)
        
        y_channel, cb_channel, cr_channel = ycbcr_image[:,:,0], ycbcr_image[:,:,1], ycbcr_image[:,:,2]
        
        cb_mean = self._get_color_channel_mean(cb_channel)
        cr_mean = self._get_color_channel_mean(cr_channel)
        
        y_dct = self._apply_dct(y_channel, Q_luminance, 'forward')
        cb_dct = self._apply_dct(cb_mean, Q_chrominance, 'forward')
        cr_dct = self._apply_dct(cr_mean, Q_chrominance, 'forward')

        y_dct_return = y_dct.copy()
        # self._histogram_of_dct_coeffs(y_dct_return)

        y_revert = self._apply_dct(y_dct, Q_luminance, 'backward')
        cb_revert = self._apply_dct(cb_dct, Q_chrominance, 'backward')
        cr_revert = self._apply_dct(cr_dct, Q_chrominance, 'backward')
        
        y_revert = y_revert[:orig_h, :orig_w]
        cb_full = self._reverse_color_channel_mean(cb_revert)[:orig_h, :orig_w]
        cr_full = self._reverse_color_channel_mean(cr_revert)[:orig_h, :orig_w]
        
        decompressed_image = np.stack([y_revert, cb_full, cr_full], axis = 2)
        
        reverted_image = self._ycbcr_to_rgb(decompressed_image)

        return reverted_image, y_dct_return
        
    def get_dct_with_embedded_message(self, message_bytes: bytes):
        image = self._read_image()
        
        orig_h, orig_w, resize_image = self._pad_image(image)
        
        ycbcr_image = self._rgb_to_ycbcr(resize_image)
        
        y_channel, cb_channel, cr_channel = ycbcr_image[:,:,0], ycbcr_image[:,:,1], ycbcr_image[:,:,2]
        
        cb_mean = self._get_color_channel_mean(cb_channel)
        cr_mean = self._get_color_channel_mean(cr_channel)
        
        y_dct = self._apply_dct(y_channel, Q_luminance, 'forward')
        cb_dct = self._apply_dct(cb_mean, Q_chrominance, 'forward')
        cr_dct = self._apply_dct(cr_mean, Q_chrominance, 'forward')

        y_dct_embed = self._embed_message(y_dct.astype(np.int32), message_bytes)

        # self._histogram_of_dct_coeffs(y_dct_embed)

        y_revert = self._apply_dct(y_dct_embed, Q_luminance, 'backward')
        cb_revert = self._apply_dct(cb_dct, Q_chrominance, 'backward')
        cr_revert = self._apply_dct(cr_dct, Q_chrominance, 'backward')

        y_revert = y_revert[:orig_h, :orig_w]
        cb_full = self._reverse_color_channel_mean(cb_revert)[:orig_h, :orig_w]
        cr_full = self._reverse_color_channel_mean(cr_revert)[:orig_h, :orig_w]
        
        decompressed_image = np.stack([y_revert, cb_full, cr_full], axis = 2)
        
        reverted_image = self._ycbcr_to_rgb(decompressed_image)

        return reverted_image, y_dct_embed
    
    def get_message_bytes_from_encoded_y_channel(self, encoded_channel: np.ndarray):

        return self._extract_message(encoded_channel)
    
    def print_embedding_capacity(self):
        image = self._read_image()
        
        orig_h, orig_w, resize_image = self._pad_image(image)
        
        ycbcr_image = self._rgb_to_ycbcr(resize_image)
        
        y_channel, cb_channel, cr_channel = ycbcr_image[:,:,0], ycbcr_image[:,:,1], ycbcr_image[:,:,2]
        
        y_dct = self._apply_dct(y_channel, Q_luminance, 'forward')

        return self._get_matrix_embedding_capacity(y_dct)