import json
import pickle

import torch
import torch.nn as nn

# from CONFIG import BERT_MODEL, BERT_ENCODER
from numpy.random import choice

import CONFIG

BERT_VOCAB_SIZE = 28996
MAX_OUTPUT = 20
INPUT_SIZE = 768
with open(f"{CONFIG.DATA_PATH}/vocab.json") as f:
	vocab = json.load(f)


if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')


class GloVeEmbedder(nn.Module):
	def __init__(self, weight_matrix):
		super(GloVeEmbedder, self).__init__()
		self.embedder = nn.Embedding(weight_matrix.shape[0], weight_matrix.shape[1])
		self.embedder.from_pretrained(weight_matrix)

	def forward(self, x):
		return self.embedder(x)


class BiAttnGRUEncoder(nn.Module):
	def __init__(self, input_size, hidden_size, embedder):
		"""
		Note: This model expects input to be in the form (1, input size) not (input size, 1)
		:param input_size: Size of embeddings
		:param hidden_size: Size of hidden space (where input it projected into and represented)
		"""
		super(BiAttnGRUEncoder, self).__init__()
		bi_dir_hidden_size = hidden_size * 2
		self.hidden_size = hidden_size

		self.bio_tag_embedding = nn.Embedding(4, 3)
		self.GLoVE_embedding_layer = embedder
		# self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
		self.gru_module = nn.GRUCell(input_size, hidden_size)
		self.backwards_gru_module = nn.GRUCell(input_size, hidden_size)
		self.encoder_att_g_gate = nn.Linear(bi_dir_hidden_size * 2, bi_dir_hidden_size)
		self.encoder_att_f_gate = nn.Linear(bi_dir_hidden_size * 2, bi_dir_hidden_size)
		self.encoder_att_linear = nn.Linear(bi_dir_hidden_size, bi_dir_hidden_size)
		self.softmax = nn.Softmax(dim=2)
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		# TODO add dropout layer
		self.dropout_layer = nn.Dropout(p=.3)

	def init_weights(self):
		for n, w in self.named_parameters():
			if "bias" in n or "GLoVE" in n:
				continue
			nn.init.xavier_uniform_(w)

	def _attention_layer(self, hidden_state, all_hidden_states):
		"""Computes the model's attention state for a single time step
		:param hidden_state: The current hidden state for which attention is being computed
		:param all_hidden_states: Matrix containing all other hidden states (bi_dir_hidden_size, no_input_words)
		:return: the current attention vector of size (1, bi_dir_hidden_size)
		"""
		attn_layer = self.encoder_att_linear(hidden_state)
		# TODO: It might be important to not consider 0's from padding the batch. (probably not though?)
		attn_layer = self.softmax(torch.matmul(attn_layer, torch.transpose(all_hidden_states, 1, 2)))
		attn_layer = torch.matmul(attn_layer, all_hidden_states)
		attn_layer = torch.cat((attn_layer, hidden_state), dim=2)
		g_gate = self.sigmoid(self.encoder_att_g_gate(attn_layer))
		f_gate = self.tanh(self.encoder_att_f_gate(attn_layer))
		return g_gate * f_gate + (1 - g_gate) * hidden_state

	def calc_attention(self, all_hidden_states):
		"""Calculates attention across the entire input
		:param all_hidden_states: Matrix containing all other hidden states (bi_dir_hidden_size, no_input_words)
		:return: all attention vectors of size (bi_dir_hidden_size, no_input_words) (I think)
		"""
		batch_size = all_hidden_states.shape[0]
		no_states = all_hidden_states.shape[1]
		attn = torch.zeros([batch_size, no_states, self.hidden_size * 2])
		for state_idx in range(0, no_states):
			current_state = all_hidden_states[:, state_idx: state_idx + 1, :]
			attn[:, :, :] = self._attention_layer(current_state, all_hidden_states)
		return attn

	def forward(self, context, answer_tags):
		"""
		:param context: The original paragraph that is being embedded
		:param answer_tags: The answer tags as represented by BIO (B = 1, I = 2, O = 0)
		:return:
		"""
		# initialize holder variables
		batch_size = context.shape[0]
		no_words = context.shape[1]
		answer_tags = answer_tags.type(torch.long)
		hidden_state_forward = torch.zeros([batch_size, self.hidden_size])
		context = self.GLoVE_embedding_layer(context.long()).squeeze(2)
		context = torch.cat((context, self.bio_tag_embedding(answer_tags)), dim=2)

		# This computes the forward stage of the GRU.
		forward_states = torch.zeros([batch_size, no_words, self.hidden_size])
		for word_idx in range(0, no_words):
			# Loop over input and compute hidden states
			current_words = context[:, word_idx, :]
			hidden_state_forward = self.gru_module(current_words, hidden_state_forward)
			hidden_state_forward = self.dropout_layer(hidden_state_forward)
			forward_states[:, word_idx, :] = hidden_state_forward

		# This computes the backwards stage of the GRU.
		backward_states = torch.zeros([batch_size, no_words, self.hidden_size])
		hidden_state_backward = torch.zeros([batch_size, self.hidden_size])
		for word_idx in range(0, no_words):
			current_word = context[:, no_words - word_idx - 1, :]
			hidden_state_backward = self.backwards_gru_module(current_word, hidden_state_backward)
			hidden_state_backward = self.dropout_layer(hidden_state_backward)
			backward_states[:, word_idx, :] = hidden_state_backward

		# last_hidden_state = torch.zeros([batch_size, self.hidden_size * 2])
		all_hidden_states = torch.cat((forward_states, backward_states), dim=2)
		# last_hidden_state = all_hidden_states[:, -1, :]

		# Finally we compute the attention
		attn = self.calc_attention(all_hidden_states)

		# and return the last hidden state as well as the attention
		return torch.cat((hidden_state_forward, hidden_state_backward), dim=1), attn


class AttnGruDecoder(nn.Module):
	def __init__(self, input_size, hidden_size, teacher_ratio, embedder, vocab_size):
		"""
		Note: This model expects input to be (1, input size) not (input size, 1)
		:param input_size: Size of row vector inputs
		:param hidden_size: Dimensionality of hidden space
		"""
		super(AttnGruDecoder, self).__init__()
		self.gru_module = nn.GRUCell(input_size, hidden_size)
		self.vocab_size = vocab_size
		self.GLoVE_embedder = embedder
		self.prediction_layer = nn.Linear(hidden_size, vocab_size)
		self.decoder_att_linear = nn.Linear(hidden_size, hidden_size)
		self.decoder_attn_weighted_ctx = nn.Linear(hidden_size * 2, hidden_size)
		self.softmax = nn.Softmax(dim=2)
		self.tanh = nn.Tanh()
		# TODO: add dropout parameter
		self.dropout_layer = nn.Dropout(p=.3)
		self.teacher_forcing_ratio = teacher_ratio

	def init_weights(self):
		for n, w in self.named_parameters():
			if "bias" in n or "GLoVE" in n:
				continue
			nn.init.xavier_uniform_(w)

	def greedy_search(self, hidden_state, encoder_attention):
		preds = torch.zeros([MAX_OUTPUT, 1]).long()
		generated_sequence = []
		no_outputted = 0
		current_words = self.GLoVE_embedder(torch.tensor([[CONFIG.START_TOKEN_IDX]])).squeeze(1)
		while True:
			if no_outputted == MAX_OUTPUT:
				break

			hidden_state = self.gru_module(current_words, hidden_state)

			attn_layer = self.decoder_att_linear(hidden_state)
			attn_layer = torch.nn.functional.softmax(torch.matmul(attn_layer, torch.t(encoder_attention[0])), dim=1)
			attn_layer = torch.matmul(attn_layer, encoder_attention[0])
			attn_layer = torch.cat((attn_layer, hidden_state), dim=1)
			hidden_state = self.tanh(self.decoder_attn_weighted_ctx(attn_layer))

			preds[no_outputted, :] = torch.argmax(
				torch.nn.functional.softmax(self.prediction_layer(hidden_state), dim=1))

			current_words = self.GLoVE_embedder(
				torch.tensor(preds[no_outputted, :].unsqueeze(0).type(torch.LongTensor), device='cuda')).squeeze(1)
			pred_idx = str(preds[no_outputted, 0].item())
			generated_sequence.append(vocab[pred_idx])
			if pred_idx == str(CONFIG.END_TOKEN_IDX): break
			no_outputted += 1
		print(generated_sequence)
		return generated_sequence

	def forward(self, x, hidden_state, encoder_attention):
		"""
		:param x: The ground truth that the model is trying to predict
		:param hidden_state: The last hidden state of the encoder
		:param encoder_attention: The attention as calculated by the encoder
		:return:
		"""

		if self.training:
			batch_size = x.shape[0]
			no_words = x.shape[1]
			teacher_forcing_decisions = choice([0, 1], no_words,
			                                   p=[1 - self.teacher_forcing_ratio, self.teacher_forcing_ratio])
			preds = torch.zeros([batch_size, no_words, self.vocab_size])
			x = x.long()
			for word_idx in range(0, no_words):
				if teacher_forcing_decisions[word_idx] == 1 and word_idx != 0:
					current_words = self.GLoVE_embedder(x[:, word_idx - 1, :]).squeeze(1)
				elif word_idx == 0:
					current_words = self.GLoVE_embedder(torch.tensor([[CONFIG.START_TOKEN_IDX] for _ in range(0, batch_size)])).squeeze(1)
				else:
					current_words = self.GLoVE_embedder(torch.tensor(
						torch.argmax(preds[:, word_idx - 1, :], dim=1).long(), device='cuda'))

				hidden_state = self.gru_module(current_words, hidden_state)
				hidden_state = hidden_state.unsqueeze(1)
				attn_layer = self.decoder_att_linear(hidden_state)
				attn_layer = self.softmax(torch.matmul(attn_layer, torch.transpose(encoder_attention, 1, 2)))
				attn_layer = torch.matmul(attn_layer, encoder_attention)
				attn_layer = torch.cat((attn_layer, hidden_state), dim=2)
				hidden_state = self.tanh(self.decoder_attn_weighted_ctx(attn_layer)).squeeze()
				preds[:, word_idx, :] = self.prediction_layer(hidden_state)

			return preds
		else:
			return self.greedy_search(hidden_state, encoder_attention)


class Seq2SeqModel(nn.Module):
	def encoder(self, *inputs):
		"""Encodes inputs into a hidden state representation, implemented in subclasses
		"""

	def decoder(self, *inputs):
		"""Takes hidden representation along with other priors and predicts output
		"""

	def forward(self, *inputs):
		"""The entire forward pass, should call encoder and decoder functions
		"""


class ParagraphLevelGeneration(Seq2SeqModel):
	def __init__(self, hidden_size, input_sizes):
		super(ParagraphLevelGeneration, self).__init__()
		self.encoder_module = BiAttnGRUEncoder(hidden_size=hidden_size, input_size=input_sizes[0])
		self.decoder_module = AttnGruDecoder(hidden_size=hidden_size * 2, input_size=input_sizes[1])

	def encoder(self, x):
		return self.encoder_module(x)

	def decoder(self, y, last_hidden_state, attn):
		return self.decoder_module(y, last_hidden_state, attn)

	def forward(self, x, y):
		if self.training:
			x, attn = self.encoder(x)
			x = self.decoder(y, x, attn)
			return x
		else:
			x, attn = self.encoder(x)
			x = self.decoder(None, x, attn)
			return x
