import json
import math
import os
import pickle
from datetime import datetime

import bcolz
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import CONFIG
from data_loaders import QADataset, data_load_fn
from models import BiAttnGRUEncoder, AttnGruDecoder, GloVeEmbedder

BATCH_SIZE = 1
INPUT_SIZE = 768
BERT_VOCAB_SIZE = 28996

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

vectors = bcolz.open(f'{CONFIG.GLOVE_PATH}/6B.300.dat')[:]
words = pickle.load(open(f'{CONFIG.GLOVE_PATH}/6B.300_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{CONFIG.GLOVE_PATH}/6B.300_idx.pkl', 'rb'))
with open(f"{CONFIG.DATA_PATH}/vocab.json") as f:
	vocab = json.load(f)

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
		output_vec = batch['target']

		encoder.to("cuda")

		x, attn = encoder(context_vec, answer_tags)
		x = decoder(context_vec, output_vec, x, attn)

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
					pred.append(vocab[str(torch.argmax(torch.softmax(x[b][w], 0), dim=0).item())])

				print(f"PRED: {pred}")
				print("=====")
			torch.save(encoder.state_dict(), f'pre_trained/weight_saves/encoder_{i}')
			torch.save(decoder.state_dict(), f'pre_trained/weight_saves/decoder_{i}')

		x = x.view(-1, x.shape[2])
		target_labels = target_labels.view(-1).long()
		loss = criterion(x, target_labels)
		figure_shit_out = target_labels.clone()
		figure_shit_out[figure_shit_out != padding_idx] = 1
		figure_shit_out[figure_shit_out == padding_idx] = 0
		loss = loss / sum(figure_shit_out)
		# This calculates the gradients for all parameters in the encoder and decoder
		loss.backward()

		# This applies all the gradients for the encoder and decoder
		encoder_optim.step()
		deocder_optim.step()

		# This adds the numerical loss (adding loss objects fills up GPU memory very quickly)
		cum_loss += loss.item() / BATCH_SIZE

		if i % 1000 == 0:
			print(f"avg loss from previous iterations: {cum_loss / 1000}")

		del loss

		torch.cuda.empty_cache()


def test(encoder, decoder, input_data):
	for i in range(0, 20):
		data = next(iter(input_data))

		context_vec = data['context']
		answer_tags = data['answer_tags']
		output_vec = data['target']

		ctx = []
		for idx, word in enumerate(context_vec[0].long()):
			word = str(word.item())
			prependable = ''
			appendable = ''
			if answer_tags[0][idx].item() == 1:
				prependable = "|"
				if idx == answer_tags[0].shape[0] - 1:
					appendable = "|"
				elif answer_tags[0][idx + 1].item() != 2:
					appendable = "|"
			if answer_tags[0][idx].item() == 2:
				if idx == answer_tags[0].shape[0] - 1:
					appendable = "|"
				else:
					if answer_tags[0][idx + 1] != 2:
						appendable = "|"

			ctx.append(prependable + vocab[word] + appendable)
		print("=========== NEW SEQUENCE ============")
		print(ctx)

		encoder.train(False)
		decoder.train(False)

		x, attn = encoder(context_vec, answer_tags)
		x = decoder(context_vec, output_vec, x, attn)
		for output in x:
			seq = []
			for idx in output:
				seq.append(vocab[str(idx)])
			print(seq)
		truth = []
		for word_idx in output_vec[0].long():
			word = str(word_idx.item())
			truth.append(vocab[word])
		print(truth)


torch.autograd.set_detect_anomaly(True)
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
weight_matrix = torch.zeros([CONFIG.MAX_VOCAB_SIZE, 300])
for word_idx in range(0, CONFIG.MAX_VOCAB_SIZE):
	if word_idx <= CONFIG.USABLE_VOCAB:
		weight_matrix[word_idx, :] = torch.tensor(glove[vocab[str(word_idx)]])
	else:
		weight_matrix[word_idx, :] = torch.rand([300])

embedder = GloVeEmbedder(weight_matrix)
encoder = BiAttnGRUEncoder(input_size=CONFIG.INPUT_SIZE + 3, hidden_size=600, embedder=embedder)
encoder.init_weights()

# The hidden size is notably doubled here due to the encoder being bi-directional. The decoder also doesn't take
# BIO tags as input. Instead it takes   the previously predicted word, or in the case of teacher forcing, the ground
# truth.
decoder = AttnGruDecoder(input_size=CONFIG.INPUT_SIZE, hidden_size=1200, teacher_ratio=1, embedder=embedder,
                         vocab_size=CONFIG.MAX_VOCAB_SIZE)
decoder.init_weights()

# TODO CHANGE
if not os.path.exists("pre_trained"): os.mkdir("pre_trained")
if not os.path.exists("pre_trained/weight_saves"): os.mkdir("pre_trained/weight_saves")

# This line loads weights if they are already present
iteration = 48000
# # if os.path.exists("pre_trained/weight_saves/encoder_1000"):
print("loaded weights")
encoder.load_state_dict(torch.load(f"pre_trained/weight_saves/encoder_fixed_attn{iteration}"))
# if os.path.exists("pre_trained/weight_saves/decoder_1000"):
print("loaded weights")
decoder.load_state_dict(torch.load(f"pre_trained/weight_saves/decoder_fixed_attn{iteration}"))

# These optimizers take care of adjusting learning rate according to gradient size
encoder_optim = torch.optim.Adam(filter(lambda x: x.requires_grad, encoder.parameters()))
decoder_optim = torch.optim.Adam(filter(lambda x: x.requires_grad, decoder.parameters()))


# Words are treated as classes and the output of the model is a probability distribution of these classes for
# each word in the output.
criterion = nn.NLLLoss(size_average=False, ignore_index=CONFIG.PADD_TOKEN_IDX)

# This creates a dataset compatible with pytorch that auto-shuffles and we don't have to worry about
# indexing errors
# check_and_gen_squad()
data_loader = DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=data_load_fn)
test(encoder, decoder, data_loader)

# data_loader = DataLoader(dataset, shuffle=True, batch_size=2, collate_fn=data_load_fn)
# train(encoder, decoder, encoder_optim, decoder_optim, criterion, data_loader, math.floor(len(data_loader) * 20 / 30),
#       CONFIG.PADD_TOKEN_IDX)
