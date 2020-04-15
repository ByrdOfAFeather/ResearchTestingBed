import torch

import torch.nn as nn

BERT_VOCAB_SIZE = 28996


class BidirectionalGRUEncoder(nn.Module):
	def __init__(self, input_size, hidden_size):
		"""pass
		:param input_size:
		:param hidden_size:
		"""
		super(BidirectionalGRUEncoder, self).__init__()
		self.bio_tag_embedding = nn.Embedding(3, 3)
		self.gru_module = nn.GRUCell(input_size, hidden_size)
		self.encoder_att_g_gate = nn.Linear(hidden_size * 4, hidden_size * 2)
		self.encoder_att_f_gate = nn.Linear(hidden_size * 4, hidden_size * 2)
		self.encoder_att_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()

		self.hidden_size = hidden_size

	def init_weights(self):
		for n, w in self.named_parameters():
			if "bias" in n:
				continue
			nn.init.xavier_uniform_(w)

	def _attention_layer(self, hidden_state, all_hidden_states):
		attn_layer = torch.t(self.encoder_att_linear(hidden_state))
		attn_layer = self.softmax(torch.matmul(torch.t(attn_layer), torch.t(all_hidden_states)))
		attn_layer = torch.matmul(attn_layer, all_hidden_states)
		attn_layer = torch.cat((attn_layer, hidden_state), dim=1)
		g_gate = self.sigmoid(self.encoder_att_g_gate(attn_layer))
		f_gate = self.tanh(self.encoder_att_f_gate(attn_layer))
		return g_gate * f_gate + (1 - g_gate) * hidden_state

	def calc_attention(self, all_hidden_states):
		batch_size = all_hidden_states.shape[0]
		no_states = all_hidden_states.shape[1]
		attn = torch.zeros([batch_size, no_states, self.hidden_size * 2])
		for batch_idx in range(0, batch_size):
			for state_idx in range(0, no_states):
				current_state = all_hidden_states[batch_idx, state_idx: state_idx + 1, :]
				attn[batch_idx, :, :] = self._attention_layer(current_state, all_hidden_states[batch_idx, :, :])
		return attn

	def forward(self, context, answer_tags):
		"""TODO: I'm actually building the answering embedding concat with context twice here :[
		:param context:
		:param answer_tags:
		:return:
		"""
		batch_size = context.shape[0]
		no_words = context.shape[1]
		hidden_state_forward = torch.zeros([1, 600])

		# TODO: even though it looks like I'm supporting batches, I'm not :-[
		forward_states = torch.zeros([batch_size, no_words, self.hidden_size])
		for batch_idx in range(0, batch_size):
			current_answer_tags = self.bio_tag_embedding(answer_tags[batch_idx, 0, :])
			current_context = context[batch_idx, :, :]
			current_embedding = torch.cat((current_answer_tags, current_context), dim=1)
			for word_idx in range(0, no_words):
				current_word = current_embedding[word_idx: word_idx + 1, :]
				hidden_state_forward = self.gru_module(current_word, hidden_state_forward)
				forward_states[batch_idx, word_idx, :] = hidden_state_forward

		backward_states = torch.zeros([batch_size, no_words, self.hidden_size])
		hidden_state_backward = torch.zeros([1, 600])
		for batch_idx in range(0, batch_size):
			current_answer_tags = self.bio_tag_embedding(answer_tags[batch_idx, 0, :])
			current_context = context[batch_idx, :, :]
			current_embedding = torch.cat((current_answer_tags, current_context), dim=1)
			for word_idx in range(0, no_words):
				current_word = current_embedding[no_words - word_idx - 1: no_words - word_idx, :]
				hidden_state_backward = self.gru_module(current_word, hidden_state_backward)
				backward_states[batch_idx, word_idx, :] = hidden_state_backward

		all_hidden_states = torch.cat((forward_states, backward_states), dim=2)
		attn = self.calc_attention(all_hidden_states)

		return torch.cat((hidden_state_forward, hidden_state_backward), dim=1), attn


class GRUDecoder(nn.Module):
	def __init__(self, input_size, hidden_size):
		"""The most basic decoder you can possibly make that returns predictions
		:param input_size: Size of row vector inputs
		:param hidden_size: Dimensionality of hidden space
		"""
		super(GRUDecoder, self).__init__()
		self.gru_module = nn.GRUCell(input_size, hidden_size)
		self.prediction_layer = nn.Linear(hidden_size, BERT_VOCAB_SIZE)
		self.decoder_att_linear = nn.Linear(hidden_size, hidden_size)
		self.decoder_attn_weighted_ctx = nn.Linear(hidden_size * 2, hidden_size)
		self.softmax = nn.Softmax()
		self.tanh = nn.Tanh()

	def init_weights(self):
		for n, w in self.named_parameters():
			if "bias" in n:
				continue
			nn.init.xavier_uniform_(w)

	def forward(self, x, hidden_state, encoder_attention):
		"""
		:param x: 
		:param hidden_state:
		:param encoder_attention:
		:return: 
		"""
		batch_size = x.shape[0]
		no_words = x.shape[1]
		preds = torch.zeros([batch_size, no_words, BERT_VOCAB_SIZE])
		for batch_idx in range(0, batch_size):
			for word_idx in range(0, no_words):
				current_word = x[batch_idx, word_idx: word_idx + 1, :]
				if word_idx == 0:
					hidden_state = self.gru_module(current_word, hidden_state)
				else:
					hidden_state = self.gru_module(current_word, hidden_state)

				preds[batch_idx, word_idx, :] = self.prediction_layer(hidden_state)

				attn_layer = self.decoder_att_linear(hidden_state)
				attn_layer = self.softmax(torch.matmul(attn_layer, torch.t(encoder_attention[batch_idx])))
				attn_layer = torch.matmul(attn_layer, encoder_attention[batch_idx])
				attn_layer = torch.cat((attn_layer, hidden_state), dim=1)
				hidden_state = self.tanh(self.decoder_attn_weighted_ctx(attn_layer))
		return preds
