import pickle

import torch

GLOVE_PATH = "data/glove_embeddings/glove_6B"


# Model Information
INPUT_SIZE = 300
MAX_INPUT_SEQ_LENGTH = 50
USES_BERT = False
WRD_TO_IDX = pickle.load(open(f'{GLOVE_PATH}/6B.300_idx.pkl', 'rb'))


if USES_BERT:
	if torch.cuda.is_available():
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	BERT_ENCODER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
	BERT_MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')


# Console Helpers
class CONSOLE_COLORS:
	"""From: https://stackoverflow.com/a/287944/8448827
	"""
	HEADER = '\033[95m'
	OK_BLUE = '\033[94m'
	OK_GREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
