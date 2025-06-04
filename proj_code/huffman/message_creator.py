# TODO documentează

from bitstring import BitArray
from .huffman_encoding import Huffman

class MessageParser:

    @staticmethod
    def create_message_bits(message_bytes: bytes, character_size = 8):

        if (character_size >= 256):
            raise Exception('Maximum value accepted for Huffman Alphabet is 255!')

        message_bits = BitArray(message_bytes)
        encoder = Huffman(character_size)
        compressed_message, table = encoder.compress_string(message_bits)

        # the message is like this
        # 32 bits representing the length of the message (w/o the table) = N
        # 8 bits representing the character size
        # for the next N bits, we have the message
        # last, we concatenate the reversing table

        # we pad the left side of the message with the reverse MSB of the message
        # up to a dimension multiple of 8
        # (e.g. for 1001...., with padding the message will be 0001001...)
        # (likewise if we have 0110...., 110110....)
        # this way we know that the first bit that is different represents the start of our message

        message_size = len(compressed_message)
        # print(message_size)
        table_size = len(table)

        message_size_bin = BitArray(bin=bin(message_size))
        pad = 32 - len(message_size_bin)
        message_size_bin = BitArray(pad) + message_size_bin

        character_size_bin = BitArray(bin(character_size))
        character_size_bin_len = len(character_size_bin)

        if (character_size_bin_len < 8):
            pad = 8 - len(character_size_bin)
            character_size_bin = BitArray(pad) + character_size_bin

        message_MSB = message_size_bin[0]

        # if the message is perfect 8 length, I add the padding nevertheless
        left_pad_size = 8 - ((message_size + 32 + 8 + table_size) % 8)

        padding = BitArray(left_pad_size)
        if (not message_MSB):
            padding.invert()

        padded_result = padding + message_size_bin + character_size_bin + compressed_message + table

        len_padded_result = len(padded_result)

        padded_result_size_bin = BitArray(bin=bin(len_padded_result))
        pad = 32 - len(padded_result_size_bin)
        padded_result_size_bin = BitArray(pad) + padded_result_size_bin

        return padded_result_size_bin + padded_result
        
    @staticmethod
    def retrieve_message_bytes(message_bits: BitArray):

        len_message = message_bits[:32].unpack('uint32')[0]
        message_bits = message_bits[32:32 + len_message]

        first_bit = message_bits[0]

        for i in range(1,9):
            if message_bits[i] != first_bit: break

        message_bits = message_bits[i:]

        message_size = message_bits[:32].unpack('uint32')[0]
        character_size = message_bits[32:40].unpack('uint8')[0]

        compressed_message_bits = message_bits[40:40+message_size]
        table_bits = message_bits[40+message_size:]

        decoder = Huffman(character_size)

        original_message = decoder.decompress_string(compressed_message_bits, table_bits)

        return bytes.fromhex(hex(int(original_message.bin, 2))[2:])