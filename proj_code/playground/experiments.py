from LSB_steg import LSB
from PVD_steg import PVD
from DCT_steg import DCT, DCT_jpeg
from conv_net import ConvNet
from huffman import MessageParser
from aux import Metrics
from phase_encoding import PhaseEncoding
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from bitstring import BitArray
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import jpegio as jio
import copy
import timeit
import scipy.io.wavfile as wavfile

def bits_per_pixel(embedding_cap):
    return embedding_cap / (512 * 512)

png_path = '/home/alexmiclea/Documents/Facultate/Licenta/images/airplane/airplane.png'
jpg_path = '/home/alexmiclea/Documents/Facultate/Licenta/images/airplane/airplane.jpeg'
jpg_e_path = '/home/alexmiclea/Documents/Facultate/Licenta/images/airplane/airplane_embed.jpeg'

audio_path = '/home/alexmiclea/Documents/Facultate/Licenta/audio/numbers/numbers.wav'
audio_e_path = '/home/alexmiclea/Documents/Facultate/Licenta/audio/numbers/numbers_embed.wav'

orig_png_image = Image.open(png_path).convert("RGB")
orig_jpg_image = Image.open(jpg_path).convert("RGB")

orig_png_image = np.array(orig_png_image)
orig_jpg_image = np.array(orig_jpg_image)

_, orig_audio = wavfile.read(audio_path)

orig_audio_shape = orig_audio.shape[0]

lsb_1 = LSB(png_path, 1)
lsb_2 = LSB(png_path, 2)

pvd_1 = PVD(png_path)
pvd_2 = PVD(png_path, [8, 8, 16, 32, 64, 128])

dct_png = DCT(png_path)
dct_jpeg = DCT_jpeg(jpg_path)

ph_en = PhaseEncoding(audio_path, 128)
ph_en_256 = PhaseEncoding(audio_path, 256)

message_bytes = os.urandom(500000)
message_bits = BitArray(message_bytes)

print(orig_audio.dtype)

# # LSB 1

# print("LSB 1")

# ec_1 = lsb_1.get_embedding_capacity()

# print("Embedding capacity for LSB 1")
# print(bits_per_pixel(ec_1))

# t1 = timeit.default_timer()
# image_embed = lsb_1.embed_message(message_bits[:ec_1])
# t2 = timeit.default_timer()

# print("Time to embed LSB 1")
# time_embed_lsb_1 = t2 - t1
# print(time_embed_lsb_1)

# t1 = timeit.default_timer()
# lsb_1.extract_message(image_embed)
# t2 = timeit.default_timer()

# print("Time to extract LSB 1")
# time_extract_lsb_1 = t2 - t1
# print(time_extract_lsb_1)


# mse = Metrics.get_mse(orig_png_image, image_embed)
# psnr = Metrics.get_psnr(orig_png_image, image_embed)
# ssim = Metrics.get_mssim(orig_png_image, image_embed)

# print(f"MSE: {mse}")
# print(f"PSNR: {psnr}")
# print(f"MSSIM: {ssim}")

# # LSB 2

# print()
# print("LSB 2")

# ec_2 = lsb_2.get_embedding_capacity()

# print("Embedding capacity for LSB 2")
# print(bits_per_pixel(ec_2))

# t1 = timeit.default_timer()
# image_embed = lsb_2.embed_message(message_bits[:ec_2])
# t2 = timeit.default_timer()

# print("Time to embed LSB 2")
# time_embed_lsb_2 = t2 - t1
# print(time_embed_lsb_2)

# t1 = timeit.default_timer()
# lsb_2.extract_message(image_embed)
# t2 = timeit.default_timer()

# print("Time to extract LSB 2")
# time_extract_lsb_2 = t2 - t1
# print(time_extract_lsb_2)


# mse = Metrics.get_mse(orig_png_image, image_embed)
# psnr = Metrics.get_psnr(orig_png_image, image_embed)
# ssim = Metrics.get_mssim(orig_png_image, image_embed)

# print(f"MSE: {mse}")
# print(f"PSNR: {psnr}")
# print(f"MSSIM: {ssim}")

# # PVD 1
# print()
# print("PVD 1")

# ec = pvd_1.get_embedding_capacity()

# print("Embedding capacity for PVD 1")
# print(bits_per_pixel(ec))

# t1 = timeit.default_timer()
# image_embed = pvd_1.get_pvd_with_embedded_message(message_bytes[:ec // 8])
# t2 = timeit.default_timer()

# print("Time to embed PVD 1")
# time_embed = t2 - t1
# print(time_embed)

# t1 = timeit.default_timer()
# pvd_2.extract_message(image_embed)
# t2 = timeit.default_timer()

# print("Time to extract")
# time_extract = t2 - t1
# print(time_extract)


# mse = Metrics.get_mse(orig_png_image, image_embed)
# psnr = Metrics.get_psnr(orig_png_image, image_embed)
# ssim = Metrics.get_mssim(orig_png_image, image_embed)

# print(f"MSE: {mse}")
# print(f"PSNR: {psnr}")
# print(f"MSSIM: {ssim}")

# # PVD 2
# print()
# print("PVD 2")

# ec = pvd_2.get_embedding_capacity()

# print("Embedding capacity")
# print(bits_per_pixel(ec))

# t1 = timeit.default_timer()
# image_embed = pvd_2.get_pvd_with_embedded_message(message_bytes[:ec // 8])
# t2 = timeit.default_timer()

# print("Time to embed")
# time_embed = t2 - t1
# print(time_embed)

# t1 = timeit.default_timer()
# pvd_2.extract_message(image_embed)
# t2 = timeit.default_timer()

# print("Time to extract")
# time_extract = t2 - t1
# print(time_extract)

# mse = Metrics.get_mse(orig_png_image, image_embed)
# psnr = Metrics.get_psnr(orig_png_image, image_embed)
# ssim = Metrics.get_mssim(orig_png_image, image_embed)

# print(f"MSE: {mse}")
# print(f"PSNR: {psnr}")
# print(f"MSSIM: {ssim}")


# print()
# print("DCT PNG")

# ec = dct_png.print_embedding_capacity()

# print("Embedding capacity")
# print(bits_per_pixel(ec))

# image_dct, _ = dct_png.get_dct_compressed_image()
# t1 = timeit.default_timer()
# image_embed, coefs = dct_png.get_dct_with_embedded_message(message_bytes[:ec // 8])
# t2 = timeit.default_timer()

# print("Time to embed")
# time_embed = t2 - t1
# print(time_embed)

# t1 = timeit.default_timer()
# dct_png.get_message_bytes_from_encoded_y_channel(coefs)
# t2 = timeit.default_timer()

# print("Time to extract")
# time_extract = t2 - t1
# print(time_extract)

# mse = Metrics.get_mse(image_dct, image_embed)
# psnr = Metrics.get_psnr(image_dct, image_embed)
# ssim = Metrics.get_mssim(image_dct, image_embed)

# print(f"MSE: {mse}")
# print(f"PSNR: {psnr}")
# print(f"MSSIM: {ssim}")


# print()
# print("DCT JPEG")

# ec = dct_jpeg.print_embedding_capacity()

# print("Embedding capacity")
# print(bits_per_pixel(ec))

# t1 = timeit.default_timer()
# image_embed = dct_jpeg.get_dct_with_embedded_message(message_bytes[:ec // 8], jpg_e_path)
# t2 = timeit.default_timer()

# print("Time to embed")
# time_embed = t2 - t1
# print(time_embed)

# t1 = timeit.default_timer()
# dct_jpeg.get_message_bytes(jpg_e_path)
# t2 = timeit.default_timer()

# print("Time to extract")
# time_extract = t2 - t1
# print(time_extract)

# mse = Metrics.get_mse(orig_jpg_image, image_embed)
# psnr = Metrics.get_psnr(orig_jpg_image, image_embed)
# ssim = Metrics.get_mssim(orig_jpg_image, image_embed)

# print(f"MSE: {mse}")
# print(f"PSNR: {psnr}")
# print(f"MSSIM: {ssim}")

# print()
# print("PE 1")
# # 1918

# t1 = timeit.default_timer()
# audio_embed = ph_en.embed_message(message_bits[:1918], audio_e_path)
# t2 = timeit.default_timer()

# print("Time to embed")
# time_embed = t2 - t1
# print(time_embed)

# t1 = timeit.default_timer()
# message = ph_en.extract_message(audio_e_path)
# t2 = timeit.default_timer()

# print("Time to extract")
# time_extract = t2 - t1
# print(time_extract)

# wrong_bits = (message_bits[:1918] ^ message[:1918]).count(1)

# print(f"BER: {((wrong_bits / 1918)) * 100}")

# mse = mean_squared_error(audio_embed[:orig_audio_shape], orig_audio)
# psnr = peak_signal_noise_ratio(audio_embed[:orig_audio_shape], orig_audio)
# ssim = structural_similarity(audio_embed[:orig_audio_shape], orig_audio)

# print(f"MSE: {mse}")
# print(f"PSNR: {psnr}")
# print(f"MSSIM: {ssim}")


# print()
# print("PE 2")
# # 959

# t1 = timeit.default_timer()
# audio_embed = ph_en_256.embed_message(message_bits[:959], audio_e_path)
# t2 = timeit.default_timer()

# print("Time to embed")
# time_embed = t2 - t1
# print(time_embed)

# t1 = timeit.default_timer()
# message = ph_en_256.extract_message(audio_e_path)
# t2 = timeit.default_timer()

# print("Time to extract")
# time_extract = t2 - t1
# print(time_extract)

# wrong_bits = (message_bits[:959] ^ message[:959]).count(1)

# print(f"BER: {((wrong_bits / 959)) * 100}")

# mse = mean_squared_error(audio_embed[:orig_audio_shape], orig_audio)
# psnr = peak_signal_noise_ratio(audio_embed[:orig_audio_shape], orig_audio)
# ssim = structural_similarity(audio_embed[:orig_audio_shape], orig_audio)

# print(f"MSE: {mse}")
# print(f"PSNR: {psnr}")
# print(f"MSSIM: {ssim}")