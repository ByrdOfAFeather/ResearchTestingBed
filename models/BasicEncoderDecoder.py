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
		nn.init.xavier_uniform_(self.gru_module.weight_hh)
		nn.init.xavier_uniform_(self.gru_module.weight_ih)

	def forward(self, context, answer_tags):
		"""TODO: I'm actually building the answering embedding concat with context twice here :[
		:param context:
		:param answer_tags:
		:return:
		"""
		batch_size = context.shape[0]
		no_words = context.shape[1]
		hidden_state_forward = torch.zeros([1, 600])
		for batch_idx in range(0, batch_size):
			current_answer_tags = self.bio_tag_embedding(answer_tags[batch_idx, :, :])
			current_context = context[batch_idx, :, :]
			current_embedding = torch.cat((current_answer_tags, current_context), dim=1)
			for word_idx in range(0, no_words):
				current_word = current_embedding[word_idx: word_idx + 1, :]
				hidden_state_forward = self.gru_module(current_word, hidden_state_forward)

		hidden_state_backward = torch.zeros([1, 600])
		for batch_idx in range(0, batch_size):
			current_answer_tags = self.bio_tag_embedding(answer_tags[batch_idx, :, :])
			current_context = context[batch_idx, :, :]
			current_embedding = torch.cat((current_answer_tags, current_context), dim=1)
			for word_idx in range(0, no_words):
				current_word = current_embedding[batch_idx, no_words - word_idx - 1: no_words - word_idx, :]
				hidden_state_backward = self.gru_module(current_word, hidden_state_backward)

		return torch.cat((hidden_state_forward, hidden_state_backward), dim=1)


class GRUDecoder(nn.Module):
	def __init__(self, input_size, hidden_size):
		"""The most basic decoder you can possibly make that returns predictions
		:param input_size: Size of row vector inputs
		:param hidden_size: Dimensionality of hidden space
		"""
		super(GRUDecoder, self).__init__()
		self.gru_module = nn.GRUCell(input_size, hidden_size)
		self.prediction_layer = nn.Linear(hidden_size, BERT_VOCAB_SIZE)

	def forward(self, x, last_encoder_hidden_state):
		"""
		:param x: 
		:param last_encoder_hidden_state: 
		:return: 
		"""
		batch_size = x.shape[0]
		no_words = x.shape[1]
		preds = torch.zeros([batch_size, no_words, BERT_VOCAB_SIZE])
		for batch_idx in range(0, batch_size):
			for word_idx in range(0, no_words):
				current_word = x[batch_idx, word_idx: word_idx + 1, :]
				if word_idx == 0:
					last_encoder_hidden_state = self.gru_module(current_word, last_encoder_hidden_state)
				else:
					last_encoder_hidden_state = self.gru_module(current_word, last_encoder_hidden_state)

				preds[batch_idx, word_idx, :] = self.prediction_layer(last_encoder_hidden_state)
		return preds
