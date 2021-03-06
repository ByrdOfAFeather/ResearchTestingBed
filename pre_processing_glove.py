import json
import shutil
import string

import CONFIG

BASE_VOCAB = 400000  # TODO: MOVE TO CONFIG
DATA_PATH = "data"

EMBEDER = {
	"B": 1,
	"I": 2,
	"O": 0,
}
EXCEPTIONS = {"\"", "("}
topics = []
topic_sets = {}
unknown_words_indx = 4
USED_INDEXES = {}


def _index_tokens(tokens):
	global unknown_words_indx, USED_INDEXES  # TODO: Ew globals
	idxes = []
	for token in tokens:
		try:
			idxes.append([CONFIG.GLoVE_WRD_TO_IDX[token.lower()]])
			USED_INDEXES[CONFIG.GLoVE_WRD_TO_IDX[token.lower()]] = token.lower()
		except KeyError:
			CONFIG.GLoVE_WRD_TO_IDX[token.lower()] = BASE_VOCAB + unknown_words_indx
			idxes.append([BASE_VOCAB + unknown_words_indx])
			USED_INDEXES[(BASE_VOCAB + unknown_words_indx)] = token.lower()
			unknown_words_indx += 1
	return idxes


def _parse_context(paragraph, current_question, include_punc=False):
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

	_context_words = _index_tokens(context_words)

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
	context_words = [[400002]]
	context_words.extend(_context_words)
	context_words.append([400003])

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
	ground_truth = _index_tokens(ground_truth)

	_ground_truth = ground_truth
	ground = [[400002]]
	ground.extend(ground_truth)
	ground_truth = ground
	ground_truth.append([400003])

	assert len(bio_base) == len(context_words), f'The BIO tags are not equal in length to the embeddings! ' \
	                                            f'{answer_info} & {len(bio_base)} & {len(context_words)}'
	return context_words, bio_base, ground_truth


def generate_squad():
	data_json = json.load(open(f'{DATA_PATH}/train-v2.0.json', 'r'))
	overall_qas_idx = 0
	for overall_idx, _ in enumerate(data_json['data']):
		for paragraphs in data_json['data'][overall_idx]['paragraphs']:
			for qas_idx, question_answer in enumerate(paragraphs['qas']):
				if question_answer["is_impossible"]:
					continue

				embedding, tags, ground_truth = _parse_context(paragraphs, qas_idx)  # TODO: Split input args up
				if embedding is None: continue
				tags = [tags]
				embedding = embedding
				ground_truth = ground_truth

				json_for_ex = {"context": embedding, "answer_tags": tags, "target": ground_truth}
				with open(f"data/squad_train_set/item_{overall_qas_idx}.json", 'w') as file:
					json.dump(json_for_ex, file)

				overall_qas_idx += 1
	with open(f"data/squad_train_set/used_idx.json", 'w') as file:
		json.dump({"used": USED_INDEXES}, file)


if __name__ == "__main__":
	generate_squad()
