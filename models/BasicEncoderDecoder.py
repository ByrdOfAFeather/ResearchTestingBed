import torch
import random as rand
import torch.nn as nn

BERT_VOCAB_SIZE = 28996
MAX_OUTPUT = 20
torch.set_default_tensor_type('torch.cuda.FloatTensor')
BERT_MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
START_TOKEN = BERT_MODEL(torch.tensor([[101]]))[0]


class BiAttnGRUEncoder(nn.Module):
	def __init__(self, input_size, hidden_size):
		"""
		Note: This model expects input to be in the form (1, input size) not (input size, 1)
		:param input_size: Size of embeddings
		:param hidden_size: Size of hidden space (where input it projected into and represented)
		"""
		super(BiAttnGRUEncoder, self).__init__()
		bi_dir_hidden_size = hidden_size * 2
		self.hidden_size = hidden_size

		self.bio_tag_embedding = nn.Embedding(3, 3)
		self.gru_module = nn.GRUCell(input_size, hidden_size)
		self.encoder_att_g_gate = nn.Linear(bi_dir_hidden_size * 2, bi_dir_hidden_size)
		self.encoder_att_f_gate = nn.Linear(bi_dir_hidden_size * 2, bi_dir_hidden_size)
		self.encoder_att_linear = nn.Linear(bi_dir_hidden_size, bi_dir_hidden_size)
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()

	def init_weights(self):
		for n, w in self.named_parameters():
			if "bias" in n:
				continue
			nn.init.xavier_uniform_(w)

	def _attention_layer(self, hidden_state, all_hidden_states):
		"""Computes the model's attention state for a single time step
		:param hidden_state: The current hidden state for which attention is being computed
		:param all_hidden_states: Matrix containing all other hidden states (bi_dir_hidden_size, no_input_words)
		:return: the current attention vector of size (1, bi_dir_hidden_size)
		"""
		attn_layer = torch.t(self.encoder_att_linear(hidden_state))
		attn_layer = self.softmax(torch.matmul(torch.t(attn_layer), torch.t(all_hidden_states)))
		attn_layer = torch.matmul(attn_layer, all_hidden_states)
		attn_layer = torch.cat((attn_layer, hidden_state), dim=1)
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
		for batch_idx in range(0, batch_size):
			for state_idx in range(0, no_states):
				current_state = all_hidden_states[batch_idx, state_idx: state_idx + 1, :]
				attn[batch_idx, :, :] = self._attention_layer(current_state, all_hidden_states[batch_idx, :, :])
		return attn

	def forward(self, context, answer_tags):
		"""TODO: I'm actually building the answering embedding concat with context twice here :[
		:param context: The original paragraph that is being embedded
		:param answer_tags: The answer tags as represented by BIO (B = 1, I = 2, O = 0)
		:return:
		"""
		# initialize holder variables
		batch_size = context.shape[0]
		no_words = context.shape[1]
		hidden_state_forward = torch.zeros([1, 600])

		# TODO: even though it looks like I'm supporting batches, I'm not :-[
		# This computes the forward stage of the GRU.
		forward_states = torch.zeros([batch_size, no_words, self.hidden_size])
		for batch_idx in range(0, batch_size):
			# Get inputs from current batch
			current_answer_tags = self.bio_tag_embedding(answer_tags[batch_idx, 0, :])
			current_context = context[batch_idx, :, :]
			current_embedding = torch.cat((current_answer_tags, current_context), dim=1)
			for word_idx in range(0, no_words):
				# Loop over input and compute hidden states
				current_word = current_embedding[word_idx: word_idx + 1, :]
				hidden_state_forward = self.gru_module(current_word, hidden_state_forward)
				forward_states[batch_idx, word_idx, :] = hidden_state_forward

		# This computes the backwards stage of the GRU.
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

		# Finally we compute the attention
		all_hidden_states = torch.cat((forward_states, backward_states), dim=2)
		attn = self.calc_attention(all_hidden_states)

		# and return the last hidden state as well as the attention
		return torch.cat((hidden_state_forward, hidden_state_backward), dim=1), attn


class AttnGruDecoder(nn.Module):
	def __init__(self, input_size, hidden_size, teacher_ratio):
		"""
		Note: This model expects input to be (1, input size) not (input size, 1)
		:param input_size: Size of row vector inputs
		:param hidden_size: Dimensionality of hidden space
		"""
		super(AttnGruDecoder, self).__init__()
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

	def greedy_search(self, hidden_state, encoder_attention, BERT_ENCODER=None):
		preds = torch.zeros([MAX_OUTPUT, 1])
		generated_sequence = []
		no_outputted = 0
		current_word = START_TOKEN[0]
		while True:
			if no_outputted == 0:
				hidden_state = self.gru_module(current_word, hidden_state)
			else:
				hidden_state = self.gru_module(current_word, hidden_state)

			preds[no_outputted, :] = torch.argmax(self.softmax(self.prediction_layer(hidden_state)))

			current_word = BERT_MODEL(torch.tensor(preds[no_outputted, :].unsqueeze(0).type(torch.LongTensor), device='cuda'))[0][0]
			generated_sequence.append(BERT_ENCODER.decode(torch.tensor([preds[no_outputted: no_outputted + 1, 0]])))
			if generated_sequence[no_outputted] == 102: break

			attn_layer = self.decoder_att_linear(hidden_state)
			attn_layer = self.softmax(torch.matmul(attn_layer, torch.t(encoder_attention[0])))
			attn_layer = torch.matmul(attn_layer, encoder_attention[0])
			attn_layer = torch.cat((attn_layer, hidden_state), dim=1)
			hidden_state = self.tanh(self.decoder_attn_weighted_ctx(attn_layer))
			if no_outputted == MAX_OUTPUT:
				break
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
			preds = torch.zeros([batch_size, no_words, BERT_VOCAB_SIZE])
			for batch_idx in range(0, batch_size):
				for word_idx in range(0, no_words):
					# TODO: Use Teacherforcing parameter (for now always 50% chance)
					if rand.randint(0, 1) == 1 or word_idx == 0:
						current_word = x[batch_idx, word_idx: word_idx + 1, :]
					else:
						current_word = BERT_MODEL(torch.tensor(torch.argmax(preds[batch_idx, word_idx - 1, :]).unsqueeze(0).unsqueeze(0).type(torch.LongTensor), device='cuda'))[0][0]
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
		else:
			BERT_ENCODER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
			return self.greedy_search(hidden_state, encoder_attention, BERT_ENCODER)
