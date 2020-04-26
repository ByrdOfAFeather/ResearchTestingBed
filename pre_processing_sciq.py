"""
Author: Matthew Byrd (ByrdOfAFeather)
Pre-processes SCIQ dataset such that it can be run through a seq2seq model.

Usage:
python pre_procesing_sciq.py
"""

import json
import string
import torch
import os

# Run on the GPU
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Load tokenizing processes from Transformers package
BERT_TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
BERT_MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

# Basic word embeddings that the model will later transform
EMBEDER = {
	"B": 1,
	"I": 2,
	"O": 0,
}


def parse_question(question):
	"""Parses a single example from the dataset
	:param question: A json containing the sample's context, answer, and question
	:return: BERT tokens for context and question as well as BIO tags for the correct answer
	"""
	context = question['support']
	answer = question['correct_answer']
	target = question['question']

	context_words = context.split(" ")[0: 510]
	target_words = target.split(" ")

	punc_filter = str.maketrans('', '', string.punctuation)

	context_words = [word.translate(punc_filter) for word in context_words]
	target_words = [word.translate(punc_filter) for word in target_words]
	answer_words = [word.translate(punc_filter) for word in answer.split(" ")]

	bio_embeddings = [EMBEDER['O']]
	inside_answer = False
	answer_index = 0
	can_be_inside_answer = True

	# The following loop and above code does:
	# -Find where the answer is and place a B tag
	# -While still in the answer (the answer is more than one word) put an I tag
	# -Outside of the answer place a O tag
	# -Start and end with an O tag for BERT's automatic
	# -start token and end token representing the start and end of a sentence.
	for word in context_words:
		if word.lower() == answer_words[0].lower() and can_be_inside_answer:
			bio_embeddings.append(EMBEDER["B"])
			answer_index += 1
			inside_answer = True
			can_be_inside_answer = False
		elif inside_answer:
			if len(answer_words) > 1:
				if word.lower() != answer_words[answer_index]:
					inside_answer = False
					bio_embeddings.append(EMBEDER["O"])
				else:
					bio_embeddings.append(EMBEDER["I"])
			else:
				inside_answer = False
				bio_embeddings.append(EMBEDER["O"])
		else:
			bio_embeddings.append(EMBEDER["O"])
	bio_embeddings.append(EMBEDER["O"])

	ground_truth = torch.tensor([BERT_TOKENIZER.encode(target_words)])
	context_words = torch.tensor([BERT_TOKENIZER.encode(context_words)])

	assert len(bio_embeddings) == len(context_words[0]), f'The BIO tags are not equal in length to the embeddings! ' \
	                                                     f'{None} & {len(bio_embeddings)} & {len(context_words[0])}'
	return context_words, bio_embeddings, ground_truth


if __name__ == "__main__":
	train_set = json.load(open("Sciq/train.json"))
	for idx, question in enumerate(train_set):
		embedding, tags, ground_truth = parse_question(question)
		if embedding is None: continue
		tags = [tags]
		embedding = embedding.cpu().detach().numpy().tolist()
		ground_truth = ground_truth.cpu().detach().numpy().tolist()

		json_for_ex = {"context": embedding, "answer_tags": tags, "target": ground_truth}

		if not os.path.exists("data"):
			os.mkdir("data")

		if not os.path.exists("data/sciq_train_set"):
			os.mkdir("data/sciq_train_set")

		with open(f"data/sciq_train_set/item_{idx}.json", 'w') as file:
			json.dump(json_for_ex, file)

	test_set = json.load(open("Sciq/test.json"))
	for idx, question in enumerate(test_set):
		embedding, tags, ground_truth = parse_question(question)
		if embedding is None: continue
		tags = [tags]
		embedding = embedding.cpu().detach().numpy().tolist()
		ground_truth = ground_truth.cpu().detach().numpy().tolist()

		json_for_ex = {"context": embedding, "answer_tags": tags, "target": ground_truth}

		if not os.path.exists("data"):
			os.mkdir("data")

		if not os.path.exists("data/sciq_test_set"):
			os.mkdir("data/sciq_test_set")

		with open(f"data/sciq_test_set/item_{idx}.json", 'w') as file:
			json.dump(json_for_ex, file)

	dev_set = json.load(open("Sciq/valid.json"))
	for idx, question in enumerate(test_set):
		embedding, tags, ground_truth = parse_question(question)
		if embedding is None: continue
		tags = [tags]
		embedding = embedding.cpu().detach().numpy().tolist()
		ground_truth = ground_truth.cpu().detach().numpy().tolist()

		json_for_ex = {"context": embedding, "answer_tags": tags, "target": ground_truth}

		if not os.path.exists("data"):
			os.mkdir("data")

		if not os.path.exists("data/sciq_dev_set"):
			os.mkdir("data/sciq_dev_set")

		with open(f"data/sciq_dev_set/item_{idx}.json", 'w') as file:
			json.dump(json_for_ex, file)
