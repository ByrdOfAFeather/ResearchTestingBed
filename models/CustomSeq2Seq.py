from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

BERT_MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
BERT_VOCAB_SIZE = 28996

# TODO: Xavier normalization on layers


class GatedQuestionAnswering(nn.Module):
	"""
	"""

	def __init__(self, input_size, hidden_size, learning_rate=.001):
		super(GatedQuestionAnswering, self).__init__()
		bidirectional_hidden_size = hidden_size * 2  # TODO: Maybe more intuitive if this is what you input?

		self.input_size = input_size
		self.hidden_size = hidden_size

		self.forward_gate = nn.GRUCell(input_size, hidden_size)
		torch.nn.init.xavier_uniform_(self.forward_gate.weight_hh)
		torch.nn.init.xavier_uniform_(self.forward_gate.weight_ih)

		self.backward_gate = nn.GRUCell(input_size, hidden_size)
		torch.nn.init.xavier_uniform_(self.backward_gate.weight_hh)
		torch.nn.init.xavier_uniform_(self.backward_gate.weight_ih)

		self.decoder_gate = nn.GRUCell(input_size, bidirectional_hidden_size)
		torch.nn.init.xavier_uniform_(self.decoder_gate.weight_hh)
		torch.nn.init.xavier_uniform_(self.decoder_gate.weight_ih)

		# TODO: Naming Conventions (hard to tell which part this is resonsible for calculating)
		self.attention_mask_weights = torch.nn.Parameter(
			torch.rand([bidirectional_hidden_size, bidirectional_hidden_size]),
			requires_grad=True
		)
		torch.nn.init.xavier_uniform_(self.attention_mask_weights)

		self.softmax = nn.Softmax(dim=1)

		self.tanh_weights = torch.nn.Parameter(
			torch.rand([bidirectional_hidden_size * 2, bidirectional_hidden_size]),
			requires_grad=True
		)
		torch.nn.init.xavier_uniform_(self.tanh_weights)

		self.tanh = nn.Tanh()

		self.sigmoid_weights = torch.nn.Parameter(
			torch.rand([bidirectional_hidden_size * 2, bidirectional_hidden_size]),
			requires_grad=True
		)
		torch.nn.init.xavier_uniform_(self.sigmoid_weights)

		self.sigmoid = nn.Sigmoid()

		self.decode_attention_mask_weights = torch.nn.Parameter(
			torch.rand([bidirectional_hidden_size, bidirectional_hidden_size]),
			requires_grad=True
		)
		torch.nn.init.xavier_uniform_(self.decode_attention_mask_weights)

		self.decode_tanh_weights = torch.nn.Parameter(
			torch.rand([bidirectional_hidden_size * 2, bidirectional_hidden_size]),
			requires_grad=True
		)
		torch.nn.init.xavier_uniform_(self.decode_tanh_weights)

		self.word_prediction_weights = torch.nn.Parameter(
			torch.rand([bidirectional_hidden_size, BERT_VOCAB_SIZE]),
			requires_grad=True
		)
		torch.nn.init.xavier_uniform_(self.word_prediction_weights)

		self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
		self.criterion = nn.NLLLoss()

	def attention_mask(self, all_hidden_states):
		return torch.matmul(self.attention_mask_weights, all_hidden_states)

	def attention_unit(self, all_hidden_states, current_hidden_state):
		"""Calculates the attention for a single step
		UNTESTED BUT SHOULD WORK
		"""
		self_attention_mask = self.attention_mask(all_hidden_states)
		self_attention_vector = self.softmax(torch.matmul(current_hidden_state, self_attention_mask))
		attention_vector = torch.matmul(self_attention_vector, torch.t(all_hidden_states))

		concat = torch.cat((current_hidden_state, attention_vector), dim=1)
		assert concat.shape[1] == self.hidden_size * 4, "The concatenation is not correct!"

		self_match_representation = self.tanh(torch.matmul(concat, self.tanh_weights))
		gate = self.sigmoid(torch.matmul(concat, self.sigmoid_weights))

		assert gate.shape[1] == self.hidden_size * 2, "The gate is not the size of the hidden state! " \
		                                              f"In fact it is {gate.shape[1]} instead of {self.hidden_size * 2}"

		attention = torch.mul(gate, self_match_representation) + torch.mul((1 - gate), current_hidden_state)
		return attention

	def forward(self, input_context: torch.Tensor, output_context: torch.Tensor) -> Any:
		number_of_input_words = input_context.shape[0]
		forward_state, backward_state = torch.zeros([1, self.hidden_size]), torch.zeros([1, self.hidden_size])

		forward_states = torch.zeros((self.hidden_size, number_of_input_words))
		backward_states = torch.zeros((self.hidden_size, number_of_input_words))

		# # Encoding section
		# Calculates forward and backward states
		# TODO: oddness here with what is actually being concated
		for i in range(0, number_of_input_words):
			current_forward_embedding = torch.reshape(input_context[i], (1, self.input_size))
			forward_state = self.forward_gate(current_forward_embedding, forward_state)
			forward_states[:, i] = forward_state

			current_backward_embedding = torch.reshape(input_context[number_of_input_words - 1 - i],
			                                           (1, self.input_size))

			backward_state = self.backward_gate(current_backward_embedding, backward_state)
			backward_states[:, number_of_input_words - 1 - i] = backward_state

		# Calculates attentions
		all_hidden_states = torch.cat((forward_states, backward_states))
		# attentions = torch.zeros(number_of_input_words, self.hidden_size * 2)
		# for i in range(0, number_of_input_words):
		# 	index = torch.tensor([i])
		# 	current_forward = torch.t(torch.index_select(forward_states, dim=1, index=index))
		# 	current_backward = torch.t(torch.index_select(backward_states, dim=1, index=index))
		# 	current_state = torch.cat((current_forward, current_backward), dim=1)
		# 	attentions[i, :] = self.attention_unit(all_hidden_states, current_state)

		# # Decoding
		# TODO
		# 1: Calculate a layer, it's output, the teacher forcing lookup, and then a second layer
		# 2: Attention And Maxout Pointer
		last_encoder_state = torch.tensor([number_of_input_words - 1])
		current_decode_state = torch.t(torch.index_select(all_hidden_states, dim=1, index=last_encoder_state))

		number_of_output_words = output_context.shape[0]
		preds = torch.zeros([number_of_output_words, BERT_VOCAB_SIZE])

		# TODO: Find a better way of indexing that retains dimension
		idx = torch.tensor([0])
		start_token = torch.index_select(output_context, dim=0, index=idx)
		generated_words = torch.zeros([number_of_output_words, self.input_size])
		for i in range(0, number_of_output_words):
			if i == 0:
				current_decode_state = self.decoder_gate(start_token, current_decode_state)
			else:
				idx = torch.tensor([i - 1])
				current_output_word = torch.index_select(output_context, dim=0, index=idx)
				current_decode_state = self.decoder_gate(current_output_word, current_decode_state)

			softmax_decode = torch.matmul(current_decode_state, self.word_prediction_weights)
			preds[i, :] = softmax_decode
			# current_word = torch.argmax(softmax_decode, dim=1)
			# current_word = torch.tensor([[current_word]])  # Stupid BERT format
			# print(current_word)
			generated_words[i, :] = BERT_MODEL(torch.tensor([[torch.argmax(self.softmax(softmax_decode))]]))[0][0]

			# r_t = torch.matmul(torch.matmul(current_decode_state, self.decode_attention_mask_weights),
			#                    torch.t(attentions))
			# a_d_t = self.softmax(r_t)
			# c_t = torch.matmul(a_d_t, attentions)
			# current_decode_state = self.tanh(
			# 	torch.matmul(torch.cat((current_decode_state, c_t), dim=1), self.decode_tanh_weights))
			# print(current_decode_state[0, 1:10])
		return preds