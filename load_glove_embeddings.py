"""Takes pre-trained glove model and makes it torch compatible,
Taken entirely from: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
"""
import pickle
import CONFIG
import bcolz as bcolz
import numpy as np

def generate_word2idx_and_idx_to_vector():
	words = []
	idx = 0
	word2idx = {}
	vectors = bcolz.carray(np.zeros(1), rootdir=f'{CONFIG.GLOVE_PATH}/6B.300.dat', mode='w')

	with open(f'{CONFIG.GLOVE_PATH}/glove.6B.300d.txt', 'rb') as f:
		for l in f:
			line = l.decode().split()
			word = line[0]
			words.append(word)
			word2idx[word] = idx
			idx += 1
			vect = np.array(line[1:]).astype(np.float)
			vectors.append(vect)

	vectors = bcolz.carray(vectors[1:].reshape((400001, 300)), rootdir=f'{CONFIG.GLOVE_PATH}/6B.300.dat', mode='w')
	vectors.flush()
	pickle.dump(words, open(f'{CONFIG.GLOVE_PATH}/6B.300_words.pkl', 'wb'))
	pickle.dump(word2idx, open(f'{CONFIG.GLOVE_PATH}/6B.300_idx.pkl', 'wb'))


if __name__ == "__main__":
	if True:  # TODO Check if files don't exist
		generate_word2idx_and_idx_to_vector()
