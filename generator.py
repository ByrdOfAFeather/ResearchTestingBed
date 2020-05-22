from pre_trained_embeddings import BERT_ENCODER, BERT_MODEL
from models import ParagraphLevelGeneration


def parse_contexts():
	pass


def generate_question(paragraph, answer):
	"""Generates a question based on a paragraph and a question using a pre-trained model
	:param paragraph:
	:param answer:
	:return:
	"""
	# TODO
	context = parse_contexts()

	model = ParagraphLevelGeneration(input_sizes=768 + 3, hidden_size=600)

	# TODO
	model.load_state_dict('None')

	output = model(context)
