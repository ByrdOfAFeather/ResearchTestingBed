import json
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import CONFIG


class QADataset(Dataset):
	"""Generic dataset that can be used in question answering tasks
	In general, the files should be formatted as so:
		context: the paragraph that supports the answer or question (BERT indexes expected but not required)
		answer: the BIO tags for a particular answer
		question: the question (BERT indexes or equivalent expected)
	"""

	def __init__(self, data_path):
		self.data_path = data_path

	# with open(f"{self.data_path}/used_idx.json", 'r') as f:
	# 	self.used_idxes = list(json.load(f)["used"].keys())
	# 	self.used_idxes = [int(idx) for idx in self.used_idxes]
	# 	self.used_idxes.append(max(self.used_idxes) + 1)
	# 	PADDING_IDX = max(self.used_idxes)

	def __len__(self):
		return len(os.listdir(self.data_path)) - 1  # -1 comes from a generation file

	def __getitem__(self, idx):
		with open(f"{self.data_path}/item_{idx}.json", 'r') as f:
			sample = json.load(f)

		return sample


def data_load_fn(batch):
	"""TODO: Lots of code clean up
	"""
	max_length_target = max([len(x['target']) for x in batch])

	original_contexts = torch.zeros([len(batch), 1, CONFIG.MAX_INPUT_SEQ_LENGTH], )
	answer_tag = torch.zeros([len(batch), CONFIG.MAX_INPUT_SEQ_LENGTH])
	original_targets = torch.zeros([len(batch), 1, max_length_target], )

	contexts = torch.zeros([len(batch), CONFIG.MAX_INPUT_SEQ_LENGTH, 1])
	targets = torch.zeros([len(batch), max_length_target, 1])

	for batch_idx, batch_itm in enumerate(batch):
		indexing_start = batch[batch_idx]['answer_tags'][0].index(1)
		assert batch_itm['context'][0][0] == CONFIG.START_TOKEN_IDX, f"start token not present, {batch_itm['context']}"
		padd = math.ceil(CONFIG.MAX_INPUT_SEQ_LENGTH / 2)
		start_index = max(indexing_start - padd, 0)
		batch_itm['context'] = batch_itm['context'][start_index: indexing_start + padd]
		batch_itm['context'][0] = [CONFIG.START_TOKEN_IDX]
		batch_itm['context'][-1] = [CONFIG.END_TOKEN_IDX]
		batch_itm['answer_tags'] = batch_itm['answer_tags'][0][start_index: indexing_start + padd]

		original_contexts[batch_idx, :, :] = torch.t(nn.functional.pad(torch.tensor(batch_itm['context']),
		                                                               [0, 0, 0, CONFIG.MAX_INPUT_SEQ_LENGTH - len(
			                                                               batch_itm['context'])],
		                                                               value=CONFIG.PADD_TOKEN_IDX))

		contexts[batch_idx, :, :] = \
			nn.functional.pad(torch.tensor([batch_itm['context']]),
			                  [0, 0, 0, CONFIG.MAX_INPUT_SEQ_LENGTH - len(batch_itm['context'])],
			                  value=CONFIG.PADD_TOKEN_IDX)

		answer_tag[batch_idx, :] = torch.t(nn.functional.pad(torch.tensor(batch_itm['answer_tags']),
		                                                     [0, CONFIG.MAX_INPUT_SEQ_LENGTH - len(
			                                                     batch_itm['answer_tags'])],
		                                                     value=CONFIG.ANSWER_PADD_IDX))

		original_targets[batch_idx, :, :] = torch.t(nn.functional.pad(torch.tensor(batch_itm['target']),
		                                                              [0,
		                                                               0,
		                                                               0,
		                                                               max_length_target - len(batch_itm["target"])],
		                                                              value=CONFIG.PADD_TOKEN_IDX))

		targets[batch_idx, :, :] = nn.functional.pad(torch.tensor(batch_itm['target']),
		                                             [0,
		                                              0,
		                                              0,
		                                              max_length_target - len(batch_itm["target"])],
		                                             value=CONFIG.PADD_TOKEN_IDX)

	return {"context": contexts,
	        "target": targets,
	        "answer_tags": answer_tag,
	        "original_context_indexes": original_contexts,
	        "original_target_indexes": original_targets,
	        }
