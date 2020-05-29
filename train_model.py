import os
import pickle

import bcolz

import CONFIG
from datetime import datetime

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from data_loaders import QADataset, data_load_fn
from models import BiAttnGRUEncoder, AttnGruDecoder, GloVeEmbedder
from pre_processing_BERT import check_and_gen_squad

BATCH_SIZE = 1
INPUT_SIZE = 768
BERT_VOCAB_SIZE = 28996

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

import logging

# create logger with 'spam_application'
log = logging.getLogger('spam_application')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

logging.info("loading pickle rick")
vectors = bcolz.open(f'{CONFIG.GLOVE_PATH}/6B.300.dat')[:]
words = pickle.load(open(f'{CONFIG.GLOVE_PATH}/6B.300_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{CONFIG.GLOVE_PATH}/6B.300_idx.pkl', 'rb'))
logging.info("loaded")

glove = {w: vectors[word2idx[w]] for w in words}
dataset = QADataset("data/squad_train_set")


def train(encoder, decoder, encoder_optim, deocder_optim, criterion, data, epochs, padding_idx):
	"""Trains a given encoder and decoder for the number of epcohs provided
	:param encoder: A model that encodes sentences
	:param decoder: A model that decodes from a previous encoder hidden state
	:param encoder_optim: Optimizer
	:param deocder_optim: Optimizer
	:param criterion: CrossEntopyLoss()
	:param data: QADataset
	:param epochs: Number of iterations for training
	:return: None
	"""
	global words

	encoder.train()
	decoder.train()
	cum_loss = 0
	start = datetime.now()
	for i in range(0, epochs):

		encoder_optim.zero_grad()
		deocder_optim.zero_grad()
		batch = next(iter(data))
		target_labels = torch.tensor(batch['original_target_indexes'])

		# Gets word vectors that encode the meaning of the word (from BERT model)
		# for more information on word vectors see: https://dzone.com/articles/introduction-to-word-vectors
		context_vec = batch['context']
		answer_tags = batch['answer_tags']
		answer_tags[answer_tags == padding_idx] = 3
		output_vec = batch['target']

		encoder.to("cuda")

		x, attn = encoder(context_vec, answer_tags)
		x = decoder(output_vec, x, attn)

		# Saves the model every 1000 iterations
		# prints the current sample and prediction for it
		# It also prints the loss but that is later in the code
		if i % 1000 == 0:
			print(f"TARGET: {target_labels[0]}")
			for b in range(0, x.shape[0]):
				print("=====")
				# print(f"ORIGINAL: {CONFIG.BERT_ENCODER.decode(target_labels[b].view(-1))}")
				pred = []
				for w in range(0, x[b].shape[0]):
					try:
						pred.append(words[torch.argmax(torch.softmax(x[b][w], 0), dim=0)])
					except IndexError:
						pred.append("UNK")
				print(f"PRED: {pred}")
				print("=====")
			torch.save(encoder.state_dict(), f'pre_trained/weight_saves/encoder_{i}')
			torch.save(decoder.state_dict(), f'pre_trained/weight_saves/decoder_{i}')

		x = x.view(-1, x.shape[2])
		target_labels = target_labels.view(-1).long()
		loss = criterion(x, target_labels)
		figure_shit_out = target_labels.clone()
		figure_shit_out[figure_shit_out == padding_idx] = 0
		figure_shit_out[figure_shit_out != padding_idx] = 1
		loss = loss / sum(figure_shit_out)
		# This calculates the gradients for all parameters in the encoder and decoder
		loss.backward()

		# This applies all the gradients for the encoder and decoder
		encoder_optim.step()
		deocder_optim.step()

		# This adds the numerical loss (adding loss objects fills up GPU memory very quickly)
		cum_loss += loss.item() / BATCH_SIZE

		del loss

		# if i % 1000 == 0 and i != 0:
		# 	end = datetime.now()
		# 	with open("log.txt", "a") as f:
		# 		f.write(f"Reached iteration {i} with loss {cum_loss / 990}\n")
		# 	print(i, cum_loss / 999)
		# 	print(f"Took {end - start}")
		# 	cum_loss = 0
		#
		# for n, w in encoder.named_parameters():
		# 	if w.grad is None:
		# 		print(i)
		# 		print("Detected None Gradient")
		# 		print(n)
		# 		continue
		# 	else:
		# 		pass
		# 	if torch.sum(w.grad) == 0:
		# 		print("0 gradient detected")
		# 		print(i)
		# 		print(n)
		#
		# for n, w in decoder.named_parameters():
		# 	if w.grad is None:
		# 		print("Detected None Gradient")
		# 		print(n)
		# 		print(i)
		# 		continue
		# 	else:
		# 		pass
		# 	if torch.sum(w.grad) == 0:
		# 		print("0 gradient detected")
		# 		print(n)
		# 		print(i)

		torch.cuda.empty_cache()


def test(encoder, decoder, input_data):
	for i in range(0, 10):
		data = next(iter(input_data))

		context_vec = data['context']
		answer_tags = data['answer_tags']
		output_vec = data['target']

		encoder.train(False)
		decoder.train(False)

		x, attn = encoder(context_vec, answer_tags)
		x = decoder(output_vec, x, attn)


# print(f"PRED: {CONFIG.BERT_ENCODER.decode(torch.argmax(torch.softmax(x[0], 1), dim=1))}")


# Init encoder and decoder models
# The input size is the size of the BERT embeddings (for a single word) plus 3 for the BIO embeddings
# The hidden size is a parameter of any RNN, it can be thought of the space that BERT words are projected into
# That's a bit abstract, but, it is essentially where the model learns to represent the sentence at a particular
# word.
# If this space is very large, it's possible for the model not to learn well as it won't find important details
# and instead just encapsulate everything as-is. If this space is very small the model might be unable to learn
# as it simply can't find what is important in the data. The size 600 here comes from the original paper and is
# what they found to be best.
weight_matrix = torch.zeros([max(dataset.used_idxes), 300])
for word_idx in range(0, max(dataset.used_idxes)):
	try:
		weight_matrix[word_idx, :] = torch.tensor(glove[words[word_idx]])
	except IndexError:
		weight_matrix[word_idx, :] = torch.rand([300])

embedder = GloVeEmbedder(weight_matrix)
encoder = BiAttnGRUEncoder(input_size=CONFIG.INPUT_SIZE + 3, hidden_size=600, embedder=embedder)
encoder.init_weights()

# The hidden size is notably doubled here due to the encoder being bi-directional. The decoder also doesn't take
# BIO tags as input. Instead it takes the previously predicted word, or in the case of teacher forcing, the ground
# truth.
decoder = AttnGruDecoder(input_size=CONFIG.INPUT_SIZE, hidden_size=1200, teacher_ratio=.5, embedder=embedder,
                         vocab_size=max(dataset.used_idxes))
decoder.init_weights()

# TODO CHANGE
if not os.path.exists("pre_trained"): os.mkdir("pre_trained")
if not os.path.exists("pre_trained/weight_saves"): os.mkdir("pre_trained/weight_saves")

# This line loads weights if they are already present
# if os.path.exists("pre_trained/weight_saves/encoder"):
# 	print("loaded weights")
# 	encoder.load_state_dict(torch.load("pre_trained/weight_saves/encoder"))
# if os.path.exists("pre_trained/weight_saves/decoder"):
# 	print("loaded weights")
# 	decoder.load_state_dict(torch.load("pre_trained/weight_saves/decoder"))

# These optimizers take care of adjusting learning rate according to gradient size
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=.001)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=.001)

# Words are treated as classes and the output of the model is a probability distribution of these classes for
# each word in the output.
criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=max(dataset.used_idxes))

# This creates a dataset compatible with pytorch that auto-shuffles and we don't have to worry about
# indexing errors
# check_and_gen_squad()


data_loader = DataLoader(dataset, shuffle=True, batch_size=2, collate_fn=data_load_fn)

train(encoder, decoder, encoder_optim, decoder_optim, criterion, data_loader, 250000, max(dataset.used_idxes))
# test(encoder, decoder, data)
