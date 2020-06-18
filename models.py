import copy
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
		self.embedder.weight.requires_grad = False

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
		self.gru_module_layer_2 = nn.GRUCell(hidden_size, hidden_size)
		self.backwards_gru_module_layer_2 = nn.GRUCell(hidden_size, hidden_size)
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
			attn[:, state_idx: state_idx + 1, :] = self._attention_layer(current_state, all_hidden_states)
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
		hidden_state_forward_layer_2 = torch.zeros([batch_size, self.hidden_size])
		context = self.GLoVE_embedding_layer(context.long()).squeeze(2)
		context = torch.cat((context, self.bio_tag_embedding(answer_tags)), dim=2)

		# This computes the forward stage of the GRU.
		forward_states = torch.zeros([batch_size, no_words, self.hidden_size])
		for word_idx in range(0, no_words):
			# Loop over input and compute hidden states
			current_words = context[:, word_idx, :]
			hidden_state_forward = self.gru_module(current_words, hidden_state_forward)
			hidden_state_forward = self.dropout_layer(hidden_state_forward)
			hidden_state_forward_layer_2 = self.gru_module_layer_2(hidden_state_forward, hidden_state_forward_layer_2)
			forward_states[:, word_idx, :] = hidden_state_forward_layer_2

		# This computes the backwards stage of the GRU.
		backward_states = torch.zeros([batch_size, no_words, self.hidden_size])
		hidden_state_backward = torch.zeros([batch_size, self.hidden_size])
		hidden_state_backward_layer_2 = torch.zeros([batch_size, self.hidden_size])
		for word_idx in range(0, no_words):
			current_word = context[:, no_words - word_idx - 1, :]
			hidden_state_backward = self.backwards_gru_module(current_word, hidden_state_backward)
			hidden_state_backward = self.dropout_layer(hidden_state_backward)
			hidden_state_backward_layer_2 = self.backwards_gru_module_layer_2(hidden_state_backward,
			                                                                  hidden_state_backward_layer_2)
			backward_states[:, word_idx, :] = hidden_state_backward_layer_2

		# last_hidden_state = torch.zeros([batch_size, self.hidden_size * 2])
		all_hidden_states = torch.cat((forward_states, backward_states), dim=2)
		# last_hidden_state = all_hidden_states[:, -1, :]

		# Finally we compute the attention
		attn = self.calc_attention(all_hidden_states)

		concatenation = torch.cat((hidden_state_forward, hidden_state_backward), dim=1)

		# and return the last hidden state as well as the attention
		return concatenation, attn


class DecoderBeam:
	def __init__(self, prob, current_state, current_word, prev_output):
		self.prob = prob
		self.current_state = current_state
		self.current_words = current_word
		self.prev_output = prev_output
		self.children = []

	def add_child(self, node):
		self.children.append(node)


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
		self.dropout_layer = nn.Dropout(p=.6)
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

	def self_attention(self, hidden_state, encoder_attention):
		attn_layer = self.decoder_att_linear(hidden_state)
		raw_scores = torch.matmul(attn_layer, torch.transpose(encoder_attention, 1, 2))
		attn_layer = self.softmax(raw_scores)
		attn_layer = torch.matmul(attn_layer, encoder_attention)
		attn_layer = torch.cat((attn_layer, hidden_state), dim=2)
		hidden_state = self.tanh(self.decoder_attn_weighted_ctx(attn_layer)).squeeze()
		return torch.transpose(raw_scores, 1, 2), hidden_state

	def maxout_pointer(self, prediction_dist, original_sequence, raw_scores):
		batch_size = original_sequence.shape[0]
		no_input_words = original_sequence.shape[1]
		pred_idx = torch.argmax(prediction_dist, dim=1, keepdim=True)
		sequence_copy = torch.zeros([batch_size, no_input_words, 1])
		for i in range(0, batch_size):
			sequence_copy[i][original_sequence[i, :, :] == pred_idx[i]] = -1
		attn_indicies = (sequence_copy == -1).nonzero()
		copy_scores = torch.zeros([batch_size, no_input_words, 1])
		if attn_indicies.shape[0] != 0:
			attn_scores = torch.zeros([batch_size, attn_indicies.shape[0], 1])
			attn_scores[attn_scores == 0] = -10000  # Basically padding
			for idx, attn_indx in enumerate(attn_indicies):
				attn_scores[attn_indx[0], idx, :] = raw_scores[tuple(attn_indx)]
			max_scores = torch.max(attn_scores, keepdim=True, dim=1)[0]
			for attn_indx in attn_indicies:
				copy_scores[attn_indx[0], attn_indx[1], :] = max_scores[attn_indx[0]]
			copy_scores = copy_scores.squeeze()
			concated = torch.cat([prediction_dist, copy_scores], dim=1)
			dist = nn.functional.log_softmax(concated, dim=1).clone()
			copy_dist = dist[:, CONFIG.MAX_VOCAB_SIZE:]
			output_dist = dist[:, 0: CONFIG.MAX_VOCAB_SIZE]
			for attn_indx in attn_indicies:
				output_dist[attn_indx[0], attn_indx[1]] += copy_dist[attn_indx[0], attn_indx[1]]
		else:
			concated = torch.cat([prediction_dist, copy_scores.squeeze()], dim=1)
			dist = nn.functional.log_softmax(concated, dim=1)
			output_dist = dist[:, 0: CONFIG.MAX_VOCAB_SIZE]
		return output_dist

	def beam_decoder(self, hidden_state, encoder_attention):
		preds = torch.zeros([MAX_OUTPUT, 1]).long()
		k = 5
		generated_sequences = []
		no_outputted = 0
		current_words = self.GLoVE_embedder(torch.tensor([[CONFIG.START_TOKEN_IDX]])).squeeze(1)
		root_node = []
		i = 0
		can_break_tracker = [False for i in range(k)]
		while True:
			if not root_node:
				hidden_state = self.gru_module(current_words, hidden_state)

				attn_layer = self.decoder_att_linear(hidden_state)
				attn_layer = torch.nn.functional.softmax(torch.matmul(attn_layer, torch.t(encoder_attention[0])), dim=1)
				attn_layer = torch.matmul(attn_layer, encoder_attention[0])
				attn_layer = torch.cat((attn_layer, hidden_state), dim=1)
				hidden_state = self.tanh(self.decoder_attn_weighted_ctx(attn_layer))
				prob_dist = torch.nn.functional.softmax(self.prediction_layer(hidden_state), dim=1)
				index = torch.argmax(prob_dist)
				token_prob = prob_dist[:, index]
				root_node.extend(
					[DecoderBeam(token_prob, hidden_state, index, [torch.argmax(prob_dist).item()]) for _ in range(k)])
			else:
				top_probs_from_beams = []
				top_index_from_beams = []
				index_to_beam = {}
				for beam_idx, beam in enumerate(root_node):
					hidden_state = beam.current_state
					current_words = self.GLoVE_embedder(beam.current_words).unsqueeze(0)
					hidden_state = self.gru_module(current_words, hidden_state)

					attn_layer = self.decoder_att_linear(hidden_state)
					attn_layer = torch.nn.functional.softmax(torch.matmul(attn_layer, torch.t(encoder_attention[0])),
					                                         dim=1)
					attn_layer = torch.matmul(attn_layer, encoder_attention[0])
					attn_layer = torch.cat((attn_layer, hidden_state), dim=1)
					hidden_state = self.tanh(self.decoder_attn_weighted_ctx(attn_layer))
					prob_dist = torch.nn.functional.softmax(self.prediction_layer(hidden_state), dim=1)
					combined_prob_dist = beam.prob * prob_dist
					top_k_prob, top_k_index = torch.sort(combined_prob_dist, descending=True)
					top_k_prob = top_k_prob[0, 0:k].tolist()
					top_k_index = [(i, beam_idx, hidden_state) for i in top_k_index[0, 0:k].tolist()]
					top_index_from_beams.extend(top_k_index)
					top_probs_from_beams.extend(top_k_prob)
					# top_from_beams.extend(top_k)
					index_to_beam[len(top_probs_from_beams)] = beam_idx
				paried = list(zip(top_probs_from_beams, top_index_from_beams))
				paried.sort(key=lambda x: x[0], reverse=True)
				if i == 1:
					paried = [paried[i * k + 1] for i in range(0, k)]  # Make sure that a different start word is used
				paried = paried[0: k]
				sequences = [bruh.prev_output.copy() for bruh in root_node]
				for idx, beams in enumerate(paried):
					new_seq = sequences[beams[1][1]].copy()
					new_seq.append(beams[1][0])
					root_node[idx] = DecoderBeam(current_state=beams[1][2], current_word=torch.tensor(beams[1][0]),
					                             prev_output=new_seq,
					                             prob=beams[0])

				for idx in range(k):
					if root_node[idx].prev_output[-1] == CONFIG.END_TOKEN_IDX:
						can_break_tracker[idx] = True
					elif len(root_node[idx].prev_output) > 3000:
						can_break_tracker[idx] = True
						print("GOT WAY TO BIG SEQUENCE")
				can_break = True
				for breakable in can_break_tracker:
					if not breakable:
						can_break = False
				if can_break:
					root_node.sort(key=lambda x: x.prob, reverse=True)
					return [node.prev_output for node in root_node]
			i += 1

	def forward(self, original_sequence, target_sequence, hidden_state, encoder_attention):
		"""
		:param target_sequence: The ground truth that the model is trying to predict
		:param hidden_state: The last hidden state of the encoder
		:param encoder_attention: The attention as calculated by the encoder
		:return:
		"""
		if self.training:
			batch_size = target_sequence.shape[0]
			no_words = target_sequence.shape[1]
			no_input_words = original_sequence.shape[1]
			teacher_forcing_decisions = choice([0, 1], no_words,
			                                   p=[1 - self.teacher_forcing_ratio, self.teacher_forcing_ratio])
			inital_preds = torch.zeros([batch_size, no_words, self.vocab_size])
			preds = torch.zeros([batch_size, no_words, self.vocab_size])
			target_sequence = target_sequence.long()
			for word_idx in range(0, no_words):
				if teacher_forcing_decisions[word_idx] == 1 and word_idx != 0:
					current_words = self.GLoVE_embedder(target_sequence[:, word_idx - 1, :]).squeeze(1)
				elif word_idx == 0:
					current_words = self.GLoVE_embedder(
						torch.tensor([[CONFIG.START_TOKEN_IDX] for _ in range(0, batch_size)])).squeeze(1)
				else:
					current_words = self.GLoVE_embedder(torch.tensor(
						torch.argmax(inital_preds[:, word_idx - 1, :], dim=1).long(), device='cuda'))

				hidden_state = self.gru_module(current_words, hidden_state)
				hidden_state = hidden_state.unsqueeze(1)
				raw_scores, hidden_state = self.self_attention(hidden_state, encoder_attention)
				preds[:, word_idx, :] = self.prediction_layer(hidden_state)

			# Copy Mechanism
			# final_dist = self.maxout_pointer(inital_preds[:, word_idx, :], original_sequence, raw_scores)

			# preds[:, word_idx, :] = final_dist

			return preds
		else:
			return self.beam_decoder(hidden_state, encoder_attention)


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

#####
# Generate the word and if the word matches the index on
