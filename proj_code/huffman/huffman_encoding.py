from bitstring import BitArray
import heapq

class Huffman:

    class Node:
        def __init__(self, letter: int = -1, frequency: int = 0):
            # -1 reprezintă un nod, altă valoare este implicit frunză, reprezintă un caracter din alfabetul meu
            self.letter = letter
            self.frequency = frequency
            self.left_child = None
            self.right_child = None
            self.code = None

        def __lt__(self, other):
            return self.frequency < other.frequency
        
        def print_graph(self, tabs = 0):
            print("  " * tabs + str(self))
            if self.left_child:
                self.left_child.print_graph(tabs + 1)
            if self.right_child:            
                self.right_child.print_graph(tabs + 1)
        
        def __str__(self):

            return f"{str(self.letter)}: {str(self.frequency)} {self.code.bin if self.code else ''}"

    def __init__(self, CHARACTER_SIZE: int = 8):

        # CHARACTER_SIZE reprezintă numărul de biți pe care îl are o literă în alfabetul nostru
        # e.g. pentru CHARACTER_SIZE = 4, o să avem 16 litere în alfabetul nostru
        self.CHARACTER_SIZE = CHARACTER_SIZE

    # pentru un dicționar de forma {literă: frecvență}, întoarce un min-heap sortat după frecvență
    def __generate_huffman_tree(self, appearance_dict: dict):

        total_letters = sum(list(appearance_dict.values()))

        node_list = [self.Node(key, value / total_letters) for (key, value) in appearance_dict.items()]

        heapq.heapify(node_list)

        return node_list

    # metoda aceasta generează alfabetul pentru codare Huffman și numără
    # numărul de apariții al fiecărei litere
    def __generate_dictionary(self, message_string: BitArray):

        # generează un dicționar de permutări pentru fiecare literă
        dict_keys = [x for x in range(2**self.CHARACTER_SIZE)]
        char_dict = {key: 0 for key in dict_keys}

        # obțin numărul de apariții al fiecărei litere și o dadaug în dicționar
        for i in range(0, len(message_string), self.CHARACTER_SIZE):
            character = message_string[i:i+self.CHARACTER_SIZE]
            # iau valoarea în format int
            character_int = int(character.bin, base=2)

            char_dict[character_int] += 1

        return char_dict
    
    # funcție recursivă care asignează un cod Huffman fiecărei litere din alfabet
    # dacă nodul este o frunză, asignez codul
    def __assign_encodings(self, node: Node, code: str):
        # dacă un nod în graf are litera diferit de -1, înseamnă că este nod frunză
        # îi atribui codul Huffman obținut

        if (node.letter != -1):
            node.code = BitArray(bin = code)

        # dacă merg pe fiul stâng, adaug '0' la cod, altfel '1'
        if (node.left_child != None):
            self.__assign_encodings(node.left_child, code + '0')
        if (node.right_child != None):
            self.__assign_encodings(node.right_child, code + '1')

    # pot deduce care este CHARACTER_SIZE după numărul de apariții de '1'
    # în reprezentarea bitstring a encodării grafului de coduri
    # https://oeis.org/A001787
    def __determine_character_length(self, bitstring: BitArray):
        num_ones = bitstring.count(1)
        for n in range(1,64):
            num = 2**(n - 1) * (n + 2)
            if num_ones == num:
                return n

    # metodă care reproduce graful folosit pentru a genera codurile Huffman
    def decode_graph_bitstring(self, bitstring: BitArray):

        local_character_size = self.__determine_character_length(bitstring)

        if(self.CHARACTER_SIZE != local_character_size):
            print(f"Warning! The given bitstring maps a code with {local_character_size} bits!")
            print(f"This class was instantiated for {self.CHARACTER_SIZE} bits!")
            self.CHARACTER_SIZE = local_character_size

        def dfs(current_string: BitArray):
            # dacă MSB este 0, fac un nod cu 2 fii
            # dacă MSB este 1, citesc local_character_size biți și aflu litera aferentă frunzei
            MSB = current_string[0]
            if MSB == 1:
                letter = current_string.bin[1:1+local_character_size]
                letter = int(letter, base = 2)
                return self.Node(letter), current_string[1+local_character_size:]
            else:
                node = self.Node()
                root_remaining_bitstring = current_string[1:]
                node.left_child, remaining_bitstring = dfs(root_remaining_bitstring)
                node.right_child, remaining_bitstring = dfs(remaining_bitstring)
                return node, remaining_bitstring

        tree, _ = dfs(bitstring)
        self.__assign_encodings(tree, '')

        return tree

    # din graful de coduri Huffman, vreau să generez o reprezentare bitstring 
    # pe care să îl trimit alături de mesajul comprimat, astfel încât să 
    # pot reproduce mesajul original
    def create_graph_bitstring(self, node: Node):
        
        return_value = BitArray()

        def dfs(node, current_string: BitArray):
            if node.letter == -1:
                current_string += BitArray(bin='0')
            else:
                padded_letter = "{0:b}".format(node.letter).rjust(self.CHARACTER_SIZE, '0')
                current_string += BitArray(bin=f'1{padded_letter}')

            if (node.left_child != None):
                dfs(node.left_child, current_string)
            if (node.right_child != None):
                dfs(node.right_child, current_string)

        dfs(node, return_value)

        return return_value

    # dintr-un arbore iau codurile fiecărei litere al alfabetului pe care îl folosim
    # întorc un dicționar {literă: cod} sau {cod: literă}, după caz
    def get_codes(self, tree: Node, codes_as_keys: bool = False):

        return_dict = {}

        def dfs(node):
            if node.letter != -1:
                bit_letter = "{0:b}".format(node.letter).rjust(self.CHARACTER_SIZE, '0')
                if codes_as_keys:
                    return_dict.update({node.code.bin : bit_letter})
                else:
                    return_dict.update({bit_letter : node.code.bin})
            else:
                if (node.left_child != None):
                    dfs(node.left_child)
                if (node.right_child != None):
                    dfs(node.right_child)

        dfs(tree)

        return return_dict

    def __encode_message(self, message_string: BitArray, dict):

        encoded_message = ''

        for x in range(0, len(message_string), self.CHARACTER_SIZE):
            message_part = message_string[x: x + self.CHARACTER_SIZE].bin
            message_part_encoded = dict[message_part]
            encoded_message += message_part_encoded

        return BitArray(bin=encoded_message)
    
    def __decode_message(self, message_string: BitArray, dict):
        decoded_message = ''

        while message_string:
            code_len = 0
            while True:
                code_len += 1
                message_part = message_string[:code_len].bin

                if message_part in dict.keys():
                    message_part_decoded = dict[message_part]
                    decoded_message += message_part_decoded
                    break

            message_string = message_string[code_len:]
        return BitArray(bin=decoded_message)

    def compress_string(self, message_string: BitArray):
        
        # generez dicționarul și arborele de coduri Huffman
        char_dict = self.__generate_dictionary(message_string)
        tree = self.__generate_huffman_tree(char_dict)

        # pentru fiecare 2 noduri din min-heap, creez un nod "merged",
        # cu frecvența sumă și îl introduc din nou în min-heap
        while(len(tree) > 1):
            node_1 = heapq.heappop(tree)
            node_2 = heapq.heappop(tree)
            merge_node = self.Node(frequency = node_1.frequency + node_2.frequency)
            merge_node.left_child = node_1
            merge_node.right_child = node_2
            heapq.heappush(tree, merge_node)

        # atribui frunzelor codurile Huffman
        self.__assign_encodings(tree[0], '')

        # fac bitstring din graf
        tree_codebook = self.create_graph_bitstring(tree[0])

        # generez dicționarul literă - cod Huffman
        dict = self.get_codes(tree[0])

        # comprim mesajul folosindu-mă de codurile obținute
        encoded_message = self.__encode_message(message_string, dict)

        return encoded_message, tree_codebook

    def decompress_string(self, message_string: BitArray, encoding_table: BitArray):

        # decodez bitstringul în graf
        graph = self.decode_graph_bitstring(encoding_table)

        # asignez codurile Huffman
        self.__assign_encodings(graph, '')

        # generez dicționarul cod Huffman - literă
        dict = self.get_codes(graph, True)

        # decodez mesajul comprimat
        decoded_message = self.__decode_message(message_string, dict)

        return decoded_message