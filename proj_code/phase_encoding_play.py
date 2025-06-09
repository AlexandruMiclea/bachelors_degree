from phase_encoding import PhaseEncoding
from bitstring import BitArray
import numpy as np
import matplotlib.pyplot as plt

FREQ = 44100

model = PhaseEncoding('/home/alexmiclea/Documents/Facultate/Licenta/audio/numbers/numbers.wav', 512)
text_message = ''
message_bytes = text_message.encode()
# print(message_bytes)
message = BitArray(bytes = message_bytes)
# print(message.bin)
model.embed_message(message, '/home/alexmiclea/Documents/Facultate/Licenta/audio/numbers/numbers_embed.wav')

x = model.original_file_content[:256]
z = model.return_audio_file[:256]

print(x.shape)
print(z.shape)

x_fft = np.fft.fft(x)
z_fft = np.fft.fft(z)

x_phase = np.angle(x_fft)
z_phase = np.angle(z_fft)

plt.plot(x_phase, label = 'orig')
plt.plot(z_phase, label = 'modif')
plt.legend()
plt.show()
