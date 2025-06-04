# TODO documentează

import numpy as np
import scipy.io.wavfile as wavfile
from bitstring import BitArray

class PhaseEncoding():

    def __init__(self, file_path, block_size):
        self.file_path = file_path
        self.BLOCK_SIZE = block_size

    def read_audio(self):
        bitrate, file_content = wavfile.read(self.file_path)

        self.bitrate = bitrate
        self.original_file_content = file_content.copy()

        if len(file_content.shape) == 2:
            file_content = file_content[:,0]

        self.modified_file_content = file_content.copy()

    def write_audio(self, out_path):
        wavfile.write(out_path, self.bitrate, self.return_audio_file)

    def get_file_split_in_blocks(self):

        size = self.modified_file_content.shape[0]

        if (size % self.BLOCK_SIZE != 0):
            segments = int(np.ceil(size / self.BLOCK_SIZE))
            # fac padding cu zero la finalul semnalului audio
            self.modified_file_content.resize((segments * self.BLOCK_SIZE))

        return np.reshape(self.modified_file_content, (segments, self.BLOCK_SIZE))

    def restore_blocks_to_file(self, blocks: np.ndarray):

        h, w = blocks.shape
        return blocks.reshape((h * w))
    
    def apply_fft_to_blocks(self, blocks):

        processed_blocks = np.fft.fft(blocks, axis = 1)
        return processed_blocks
    
    def parse_message_bits_to_phase_shifts(self, message_bits: BitArray):

        phase_shifts = list()

        for bit in message_bits:
            if bit == 0:
                phase_shifts.append(np.pi / 2)
            else:
                phase_shifts.append(-np.pi / 2)

        return np.array(phase_shifts)

    def get_embedding_capacity(self, blocks: np.ndarray) -> np.int32:
        return blocks.shape[0] * 2 
    
    def embed_bits_in_blocks(self, blocks: np.ndarray, message_bits: BitArray):

        magnitudes, phases = np.abs(blocks), np.angle(blocks)

        phase_shifts = self.parse_message_bits_to_phase_shifts(message_bits)
        assert(len(phase_shifts) <= self.get_embedding_capacity(blocks))

        segment_middle = self.BLOCK_SIZE // 2

        
        new_phases = phases.copy()

        # aplic câte un phase shift pentru fiecare segment din semnal
        for phase in new_phases:
            used_phase_shifts = phase_shifts[:1]
            len_used_phase_shifts = len(used_phase_shifts)

            phase[segment_middle - len_used_phase_shifts : segment_middle] = used_phase_shifts
            phase[segment_middle + 1 : segment_middle + len_used_phase_shifts + 1] = -used_phase_shifts

            phase_shifts = phase_shifts[len_used_phase_shifts:]

        return magnitudes, new_phases

    def extract_bits_from_blocks(self, blocks: np.ndarray):

        magnitudes, phases = np.abs(blocks), np.angle(blocks)
        message = []
        segment_middle = self.BLOCK_SIZE // 2

        for phase in phases:
            len_used_phase_shifts = 1
            
            message.append(phase[segment_middle - len_used_phase_shifts : segment_middle])

        message = np.concatenate(message) < 0
        message = ''.join(['1' if x else '0' for x in message])
        message = f'0b{message}'

        message_bitarray = BitArray(message)

        return message_bitarray

    def embed_message(self, message_bits: BitArray):

        self.read_audio()
        blocks = self.get_file_split_in_blocks()
        fft_blocks = self.apply_fft_to_blocks(blocks)
        magnitudes, new_phases = self.embed_bits_in_blocks(fft_blocks, message_bits)

        new_audio_signal_blocks = np.fft.ifft(magnitudes * np.exp(1j * new_phases)).real

        new_audio_signal = self.restore_blocks_to_file(new_audio_signal_blocks)

        new_audio_signal = new_audio_signal.astype(np.int16)

        return_audio_file = None

        if len(self.original_file_content.shape) == 2:
            return_audio_file = np.zeros((new_audio_signal.shape[0], 2)).astype(np.int16)
            return_audio_file[:,0] = new_audio_signal
            return_audio_file[:self.original_file_content.shape[0],1] = self.original_file_content[:,1]
        else:
            return_audio_file = new_audio_signal

        if len(return_audio_file.shape) == 2:
            return_audio_file[:,0] = new_audio_signal

        self.return_audio_file = return_audio_file

        return return_audio_file
    
    def extract_message(self):

        audio_for_extraction = self.return_audio_file
        
        if len(audio_for_extraction.shape) == 2:
            audio_for_extraction = audio_for_extraction[:,0]

        segments = int(np.ceil(audio_for_extraction.shape[0] / self.BLOCK_SIZE))
            # pads the end of the file with zeroes
        audio_for_extraction_blocks = np.reshape(audio_for_extraction, (segments, self.BLOCK_SIZE))

        audio_for_extraction_fft_blocks = self.apply_fft_to_blocks(audio_for_extraction_blocks)

        return_message = self.extract_bits_from_blocks(audio_for_extraction_fft_blocks)

        return return_message