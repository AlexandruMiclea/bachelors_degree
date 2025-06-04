import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image
from bitstring import BitArray
import jpegio as jio

class DCT_jpeg:
    def __init__(self, file_path):
        self.file_path = file_path

    def _read_image(self) -> np.ndarray:
        image_metadata = jio.read(self.file_path)
        return image_metadata

    # metodă ce îmi întoarce o listă de coeficienți dată fiind o matrice de coeficienți
    # parcurgerea se face în zigzag
    def _get_zigzagged_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.fliplr(matrix)
        diagonals = [np.diagonal(matrix, offset=i) for i in range(7,-8, -1)]

        for i in range(0, 16, 2):
            diagonals[i] = np.flip(diagonals[i])

        result = np.concat(diagonals)
        return result
    
    # metodă ce îmi întoarce o matrice de coeficienți dată fiind o listă de coeficienți
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

    # în metoda de mai jos, nu număr coeficientul DC ca parte din spațiul de îmbibare
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
                bit = (elem % 2)
                message_bits += BitArray(bin=str(int(bit)))
        
        return message_bits

    def _embed_message(self, coeffs_matrix: np.ndarray, message_bytes: bytes) -> np.ndarray:

        total_capacity = self._get_matrix_embedding_capacity(coeffs_matrix)

        payload_bits = BitArray(message_bytes).bin
        assert(len(payload_bits) <= total_capacity)

        image_h, image_w = coeffs_matrix.shape
        return_image = np.zeros((image_h, image_w))

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
        plt.show()

    def print_embedding_capacity(self):
        image_metadata = self._read_image()

        metadata = image_metadata.coef_arrays[0].copy()

        return self._get_matrix_embedding_capacity(metadata)
    
    def helper_get_dct_coefs_embed(self, message_bytes):
        image_metadata = self._read_image()

        embedded_matrix = copy.deepcopy(image_metadata.coef_arrays[0].copy())
        
        return embedded_matrix, self._embed_message(embedded_matrix, message_bytes)

        
    def get_dct_with_embedded_message(self, message_bytes: bytes, output_path: str):
        image_metadata = self._read_image()

        embedded_matrix = copy.deepcopy(image_metadata.coef_arrays[0].copy())
        
        aux = self._embed_message(embedded_matrix, message_bytes)

        image_metadata.coef_arrays[0][:] = aux

        jio.write(image_metadata, output_path)

        image_read = Image.open(output_path).convert('RGB')

        # image_read = Image.open(self.file_path).convert('RGB')
        return np.array(image_read)

    def get_message_bytes(self, image_path: str):

        image_metadata = jio.read(image_path)

        return self._extract_message(image_metadata.coef_arrays[0].copy())
