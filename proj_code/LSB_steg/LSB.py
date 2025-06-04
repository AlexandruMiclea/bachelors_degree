import numpy as np
from PIL import Image
from bitstring import BitArray
import matplotlib.pyplot as plt

class LSB:

    def __init__(self, file_path: str, k_lsb: int):
        assert (1 <= k_lsb <= 8)
        self.file_path = file_path
        self.k_lsb = k_lsb
        self._read_image()

    def _read_image(self) -> np.ndarray:

        image_read = Image.open(self.file_path).convert('RGB')
        image = np.array(image_read)
        self.image_size = image.shape
        return image

    def _get_message_bits(self, image_content: np.ndarray):
        message_bits = image_content & (2**self.k_lsb - 1)

        r, c, col = message_bits.shape
        # aceste apeluri de reshape se asigură că în lista finală, ordinea pixelilor
        # este aceeași cu cea în care am îmbibat mesajul
        message_bits = np.reshape(message_bits, (r, c * col))
        message_bits = np.reshape(message_bits, (r * c * col))

        # funcție lambda care extrage k-LSB în format binar
        bin_str_repr = lambda x: format(x, f'0{self.k_lsb}b')
        aux_matrix = [bin_str_repr(el) for el in message_bits]

        return BitArray(bin=''.join(aux_matrix))

    def get_embedding_capacity(self):
        return self.k_lsb * self.image_size[0] * self.image_size[1] * self.image_size[2]

    def embed_message(self, message: BitArray) -> np.ndarray | str:

        image = self._read_image()

        capacity = self.get_embedding_capacity()

        # verific dacă am loc să pun mesajul meu in imagine
        if len(message) > capacity: 
            return "Cannot embed message! Capacity is smaller than message size."

        # mă apuc să modific pixelii din imagine -> ordinea de îmbibare
        # este dată de R-G-B pentru fiecare pixel de pe o linie, pentru fiecare coloană
        # iar pentru fiecare canal RGB îmbib câte k biți din mesaj
        modified_image = np.copy(image)

        for i in range(0, len(message), self.k_lsb):
            partial_message = message[i : i + self.k_lsb]

            relative_position = i // self.k_lsb

            color_channel = relative_position % 3
            
            pixel_pos = relative_position // 3
            pixel_line = pixel_pos // modified_image.shape[1]
            pixel_col = pixel_pos % modified_image.shape[1]

            pixel = modified_image[pixel_line, pixel_col, color_channel].astype(np.uint32)

            pixel_modif = (pixel & (255 - 2 ** self.k_lsb + 1)) | int(partial_message.bin, 2)

            modified_image[pixel_line, pixel_col, color_channel] = pixel_modif

        return modified_image
    
    def extract_message(self, image: np.ndarray):
        text = self._get_message_bits(image)
        return text
    
    def save_image(self, image: np.ndarray, path: str):
        plt.imsave(path, image)