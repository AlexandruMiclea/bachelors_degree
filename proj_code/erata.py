# pagina 17 -> eroare pentru graficul PVD

from PVD_steg import PVD

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import jpegio as jio
import copy

#deveselu_original
#deveselu_embed

deveselu_path = '/home/alexmiclea/Documents/Facultate/Licenta/images/deveselu/deveselu.png'
image_read = Image.open(deveselu_path).convert('RGB')

deveselu_original = np.array(image_read)

model = PVD(deveselu_path)

fmi_path = '/home/alexmiclea/Documents/Facultate/Licenta/images/fmi/fmi.jpg'

with open(fmi_path, 'rb') as exec_file:
    fmi_bytes = exec_file.read()

deveselu_embed = model.get_pvd_with_embedded_message(fmi_bytes)

fig, axs = plt.subplots(1,2)
fig.subplots_adjust(bottom = 0.3)
axs[0].set_axis_off()
axs[1].set_axis_off()
axs[0].imshow((deveselu_original[:,:,0] & 1) * 255, cmap='gray')
axs[0].set_title('Imaginea originală')
axs[1].imshow((deveselu_embed[:,:,0] & 1) * 255, cmap='gray')
axs[1].set_title('Imaginea cu mesaj ascuns')

def update_images(val):
    mask = slider_mask.val
    channel = slider_rgb.val

    mask = 2 ** mask

    axs[0].set_axis_off()
    axs[1].set_axis_off()
    axs[0].imshow((deveselu_original[:,:,channel] & mask) * 255, cmap='gray')
    axs[0].set_title('Imaginea originală')
    axs[1].imshow((deveselu_embed[:,:,channel] & mask) * 255, cmap='gray')
    axs[1].set_title('Imaginea cu mesaj ascuns')


ax_rgb = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_mask = plt.axes([0.25, 0.1, 0.65, 0.03])
slider_rgb = plt.Slider(ax = ax_rgb, label="Canal Culoare", valmin = 0, valmax = 2, valinit = 0, valstep= 1)
slider_mask = plt.Slider(ax = ax_mask, label = "Mască", valmin = 0, valmax = 7, valinit = 0, valstep= 1)
slider_rgb.on_changed(update_images)
slider_mask.on_changed(update_images)

plt.show()