import json
import shutil
import string

import CONFIG

# TODO:
# 1) Get the max vocab size: let's limit it to BERT -- DONE
# 2) Get Word idxs and save them to JSON file


PUNC_FILTER = str.maketrans('', '', string.punctuation)
EMBEDER = {
	"B": 1,
	"I": 2,
	"O": 0,
}
EXCEPTIONS = {"\"", "("}


def _index_tokens(tokens, wrd_to_idx):
	idxes = []
	for token in tokens:
		try:
			idxes.append([wrd_to_idx[token.lower()]["custom_vocab_idx"]])
		except KeyError:
			idxes.append([CONFIG.UNKNOWN_TOKEN_IDX])
	return idxes


def _parse_context(paragraph, current_question, wrd_to_idx, include_punc=False):
	punc_filter = str.maketrans('', '', string.punctuation)

	context_text = paragraph['context']
	answer_info = paragraph['qas'][current_question]['answers'][0]
	answer_start = answer_info['answer_start']
	answer_text = answer_info['text']

	context_words = context_text.split(" ")
	ground_truth = paragraph['qas'][current_question]['question'].split(" ")

	# Get rid of punctuation
	if not include_punc:
		context_words = [word.translate(punc_filter) for word in context_words]
		ground_truth = [word.translate(punc_filter) for word in ground_truth]

	_context_words = _index_tokens(context_words, wrd_to_idx)

	bio_base = []  # O to match with BERT's "CRT" Token  TODO: CRT? Or was it another shorten
	char_tracker = 0
	in_answer_section = False
	answer_words = answer_text.split(" ")
	answer_word_index = 0
	start_index = 0
	can_be_in_answer = True

	for idx, word in enumerate(context_text.split(" ")):
		# Answer_start is a bit inaccurate as it can start in the middle of words in the case of
		# arbitrary prefixes, some examples: $, (, top-, etc
		if (char_tracker == answer_start or (
				answer_words[0] in word and word.index(answer_words[0]) is not 0 or word == "50-50")) and \
				can_be_in_answer:
			start_index = idx
			bio_base.append(EMBEDER["B"])
			in_answer_section = True if len(answer_words) != 1 else False
			can_be_in_answer = False

		elif in_answer_section:
			if len(answer_words) != 1:
				bio_base.append(EMBEDER["I"])
				answer_word_index += 1
				if answer_word_index == len(answer_words):
					in_answer_section = False
			else:
				in_answer_section = False

		else:
			bio_base.append(EMBEDER["O"])
		char_tracker += len(word) + 1

	# Embed words
	_context_words = _context_words[max(0, start_index - 255): start_index + 255]
	context_words = [[CONFIG.START_TOKEN_IDX]]
	context_words.extend(_context_words)
	context_words.append([CONFIG.END_TOKEN_IDX])

	bio_base = bio_base[max(0, start_index - 255): start_index + 255]
	bio_base.insert(0, EMBEDER["O"])
	bio_base.append(EMBEDER["O"])  # End O for BERT's end token

	try:
		bio_base.index(1)
	except ValueError:
		print(context_text.split(" "))
		print(answer_words)
		shutil.rmtree("data/squad_train_set")
		import sys
		sys.exit(0)

	if len(ground_truth) > 1000:
		return None, None, None
	ground_truth = _index_tokens(ground_truth, wrd_to_idx)

	_ground_truth = ground_truth
	ground = [[CONFIG.START_TOKEN_IDX]]
	ground.extend(ground_truth)
	ground_truth = ground
	ground_truth.append([CONFIG.END_TOKEN_IDX])

	assert len(bio_base) == len(context_words), f'The BIO tags are not equal in length to the embeddings! ' \
	                                            f'{answer_info} & {len(bio_base)} & {len(context_words)}'
	return context_words, bio_base, ground_truth


def sample_most_used_words():
	"""Looks through all of the SQuAD dataset and samples the most used words that also appear in the GLoVE vocabulary
	:return:
	"""
	word_dict = {}
	data_json = json.load(open(f'{CONFIG.DATA_PATH}/train-v2.0.json', 'r'))
	for overall_idx, _ in enumerate(data_json['data']):
		for paragraphs in data_json['data'][overall_idx]['paragraphs']:
			for qas_idx, question_answer in enumerate(paragraphs['qas']):
				context_text = paragraphs['context']
				context_words = context_text.split(" ")
				ground_truth = paragraphs['qas'][qas_idx]['question'].split(" ")
				if len(ground_truth) > 1000: continue

				for word in context_words:
					word = word.lower().translate(PUNC_FILTER)
					try:
						word_dict[word] += 1
					except KeyError:
						word_dict[word] = 1
				for word in ground_truth:
					word = word.lower().translate(PUNC_FILTER)
					try:
						word_dict[word] += 1
					except KeyError:
						word_dict[word] = 1

	result = sorted(list(word_dict.items()), key=lambda x: -x[1])
	return_strings = []
	idx = 0
	wrd_to_idx = {}
	idx_to_idx = {}
	for wrd_res, _ in result:
		if len(return_strings) == CONFIG.USABLE_VOCAB + 1:
			break
		else:
			try:
				glove_idx = CONFIG.GLoVE_WRD_TO_IDX[wrd_res]
				wrd_to_idx[wrd_res] = {"custom_vocab_idx": idx, "glove_idx": glove_idx}
				idx_to_idx[idx] = wrd_res
				idx += 1
				return_strings.append(wrd_res)
			except KeyError:
				continue

	idx_to_idx[CONFIG.START_TOKEN_IDX] = "<START>"
	idx_to_idx[CONFIG.END_TOKEN_IDX] = "<END>"
	idx_to_idx[CONFIG.PADD_TOKEN_IDX] = "<PADD>"
	idx_to_idx[CONFIG.UNKNOWN_TOKEN_IDX] = "<UNK>"

	with open(f"{CONFIG.DATA_PATH}/vocab.json", 'w') as f:
		json.dump(idx_to_idx, f)
	return return_strings, wrd_to_idx


def pre_process():
	vocab, wrd_to_idx = sample_most_used_words()
	word_set = set(vocab)
	data_json = json.load(open(f'{CONFIG.DATA_PATH}/train-v2.0.json', 'r'))
	overall_qas_idx = 0
	for overall_idx, _ in enumerate(data_json['data']):
		for paragraphs in data_json['data'][overall_idx]['paragraphs']:
			for qas_idx, question_answer in enumerate(paragraphs['qas']):
				if question_answer["is_impossible"]:
					continue

				embedding, tags, ground_truth = _parse_context(paragraphs, qas_idx,
				                                               wrd_to_idx)
				if embedding is None: continue
				tags = [tags]
				embedding = embedding
				ground_truth = ground_truth

				json_for_ex = {"context": embedding, "answer_tags": tags, "target": ground_truth}
				with open(f"data/squad_train_set/item_{overall_qas_idx}.json", 'w') as file:
					json.dump(json_for_ex, file)

				overall_qas_idx += 1


sample_most_used_words()