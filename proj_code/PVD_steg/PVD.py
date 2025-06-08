# TODO documentează

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bitstring import BitArray

class PVD:

    def __init__(self, file_path, quantization_list = None):
        self.file_path = file_path

        if (not quantization_list):
            self.quantization_list = [1, 1, 2, 4, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64]
        else:
            self.quantization_list = quantization_list

        self.quantization_ranges = self._determine_quantization_ranges()

    def _determine_quantization_ranges(self):
        ranges = list()
        start = 0

        for i in self.quantization_list:
            ranges.append([np.int32(start), np.int32(start + i - 1)])
            start += i

        return ranges

    def _read_image(self) -> np.ndarray:

        image_read = Image.open(self.file_path)
        return np.array(image_read)
    
    def _get_pixel_differences(self, image: np.ndarray):
        image_h, image_w, color_channels = image.shape
        image = image.copy().astype(np.int32)
        differences = list()

        for c in range(color_channels):
            for i in range(image_h):
                if (i % 2 == 0):

                    for j in range(0, image_w - 1, 2):
                        differences.append(np.abs(image[i,j,c] - image[i,j + 1,c]))

                    if (image_w % 2 != 0 and i != image_h - 1):
                        differences.append(np.abs(image[i, image_w - 1, c] - image[i + 1, image_w - 1, c]))

                else:
                    for j in range(image_w - image_w % 2 - 1, 0, -2):
                        differences.append(np.abs(image[i, j, c] - image[i, j - 1, c]))

        return differences

    def get_embedding_capacity(self):
        differences = self._get_pixel_differences(self._read_image())
        capacities = self._get_capacities(differences)
        return np.sum(np.bincount(capacities, minlength=8) * np.array([0,1,2,3,4,5,6,7]))
    
    def _get_capacities(self, differences: list):
        bit_count_fn = lambda x: [np.log2(elem[1] - elem[0] + 1).astype(np.int32) for elem in self.quantization_ranges if elem[0] <= x <= elem[1]][0]
        return np.array(list(map(bit_count_fn, differences)))
    
    def _modify_pixel_intensities(self, pixel_1, pixel_2, value):

        pixel_1 = pixel_1.astype(np.int32)
        pixel_2 = pixel_2.astype(np.int32)

        mean = (pixel_1 + pixel_2) // 2

        ascending = True

        if pixel_1 > pixel_2:
            ascending = False

        pixel_1 = mean - (value // 2)
        pixel_2 = mean + (value - value // 2)

        if pixel_1 < 0 or pixel_2 < 0:
            min_val = min(pixel_1, pixel_2)
            undershoot = -min_val
            pixel_1 += undershoot
            pixel_2 += undershoot

        if pixel_2 > 255 or pixel_1 > 255:
            max_val = max(pixel_1, pixel_2)
            overshoot = max_val - 255
            pixel_1 -= overshoot
            pixel_2 -= overshoot
        
        assert(pixel_1 >= 0 and pixel_2 <= 255)
        if ascending:
            return pixel_1.astype(np.uint8), pixel_2.astype(np.uint8)
        else:
            return pixel_2.astype(np.uint8), pixel_1.astype(np.uint8)
    
    def _get_quantization_range_lower(self, value):
        for (idx,elem) in enumerate(self.quantization_ranges):
            if elem[0] <= value <= elem[1]:
                return elem[0]
    
    def _embed_message(self, image: np.ndarray, message_bytes: bytes, capacities) -> np.ndarray:
        image_h, image_w, color_channels = image.shape
        ret_image = image.copy().astype(np.int32)

        total_capacity = self.get_embedding_capacity()

        payload_bits = BitArray(message_bytes).bin
        assert(len(payload_bits) <= total_capacity)
        
        capacity_idx = 0

        for c in range(color_channels):
            if len(payload_bits) == 0:
                break

            for i in range(image_h):
                if (i % 2 == 0):
                    for j in range(0, image_w - 1, 2):
                        if len(payload_bits) == 0:
                            break

                        capacity = capacities[capacity_idx]

                        if capacity == 0:
                            capacity_idx += 1
                            continue

                        message_bits = payload_bits[:capacity]

                        diff = np.abs(ret_image[i,j,c] - ret_image[i,j+1,c])
                        int_value = int(message_bits, 2) + self._get_quantization_range_lower(diff)
                        
                        ret_image[i,j,c], ret_image[i, j + 1, c] = \
                            self._modify_pixel_intensities(image[i,j,c],image[i,j+1,c],int_value)
                        
                        payload_bits = payload_bits[capacity:]

                        capacity_idx += 1

                    if (image_w % 2 != 0 and i != image_h - 1):
                        if len(payload_bits) == 0:
                            break

                        capacity = capacities[capacity_idx]

                        if capacity == 0:
                            capacity_idx += 1
                            continue

                        message_bits = payload_bits[:capacity]

                        diff = np.abs(ret_image[i, image_w - 1, c] - ret_image[i + 1, image_w - 1, c])
                        int_value = int(message_bits, 2) + self._get_quantization_range_lower(diff)

                        ret_image[i, image_w - 1, c], ret_image[i + 1, image_w - 1, c] = \
                            self._modify_pixel_intensities(image[i, image_w - 1, c], image[i + 1, image_w - 1, c],int_value)
                        
                        payload_bits = payload_bits[capacity:]
                        
                        capacity_idx += 1

                else:
                    for j in range(image_w - image_w % 2 - 1, 0, -2):
                        if len(payload_bits) == 0:
                            break

                        capacity = capacities[capacity_idx]

                        if capacity == 0:
                            capacity_idx += 1
                            continue

                        message_bits = payload_bits[:capacity]

                        diff = np.abs(ret_image[i, j, c] - ret_image[i, j - 1, c])
                        int_value = int(message_bits, 2) + self._get_quantization_range_lower(diff)

                        ret_image[i, j, c], ret_image[i, j - 1, c] = \
                            self._modify_pixel_intensities(image[i, j, c], image[i, j - 1, c],int_value)
                        
                        payload_bits = payload_bits[capacity:]

                        capacity_idx += 1

        assert(not np.any((ret_image < 0)|(ret_image > 255)))
        return ret_image.astype(np.uint8)

    def _extract_message(self, image: np.ndarray, capacities) -> BitArray:
        image_h, image_w, color_channels = image.shape
        image = image.copy().astype(np.int32)

        capacity_idx = 0
        binary_string = ''

        for c in range(color_channels):
            for i in range(image_h):
                if (i % 2 == 0):
                    for j in range(0, image_w - 1, 2):

                        capacity = capacities[capacity_idx]

                        if capacity == 0:
                            capacity_idx += 1
                            continue

                        capacity_mask = 2 ** capacity - 1

                        diff = np.abs(image[i,j,c] - image[i, j + 1, c])

                        value = capacity_mask & diff

                        aux = format(value, f'0{capacity}b')

                        binary_string += aux

                        capacity_idx += 1

                    if (image_w % 2 != 0 and i != image_h - 1):
                        capacity = capacities[capacity_idx]
                        
                        if capacity == 0:
                            capacity_idx += 1
                            continue

                        capacity_mask = 2 ** capacity - 1
                        diff = np.abs(image[i, image_w - 1, c] - image[i + 1, image_w - 1, c])

                        value = capacity_mask & diff
                        aux = format(value, f'0{capacity}b')

                        binary_string += aux

                        capacity_idx += 1

                else:
                    for j in range(image_w - image_w % 2 - 1, 0, -2):
                        capacity = capacities[capacity_idx]

                        if capacity == 0:
                            capacity_idx += 1
                            continue

                        capacity_mask = 2 ** capacity - 1
                        diff = np.abs(image[i, j, c] - image[i, j - 1, c])

                        value = capacity_mask & diff

                        aux = format(value, f'0{capacity}b')

                        binary_string += aux

                        capacity_idx += 1

        return BitArray(bin=binary_string)
        
    def get_pvd_with_embedded_message(self, message_bytes: bytes):

        image = self._read_image()
        differences = self._get_pixel_differences(image)
        capacities = self._get_capacities(differences)
        result_image = self._embed_message(image, message_bytes, capacities)

        return result_image
    
    def extract_message(self, image: np.ndarray):

        differences = self._get_pixel_differences(image)
        capacities = self._get_capacities(differences)
        message = self._extract_message(image, capacities)

        return message