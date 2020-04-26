import json
import string
import os
import datetime

import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

DATA_PATH = "stanford-question-answering-dataset"
BERT_TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
BERT_MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

data_json = json.load(open(f'{DATA_PATH}/train-v2.0.json', 'r'))
topics = []
topic_sets = {}

# # TODO
# Get a numeric index 1 - ..... (however many samples are in SQAuD)
# Samples are based on question and answer not context (question result, answer + context = input)
# Save samples to the disk
EMBEDER = {
    "B": 1,
    "I": 2,
    "O": 0,
}


def _parse_context(paragraph, current_question, include_punc=False):
    punc_filter = str.maketrans('', '', string.punctuation)

    context_text = paragraph['context']
    answer_info = paragraph['qas'][current_question]['answers'][0]
    answer_start = answer_info['answer_start']
    answer_text = answer_info['text']

    context_words = context_text.split(" ")[0: 510]
    ground_truth = paragraph['qas'][current_question]['question'].split(" ")

    # Get rid of punctuation
    if not include_punc:
        context_words = [word.translate(punc_filter) for word in context_words]
        ground_truth = [word.translate(punc_filter) for word in ground_truth]

    # Embed words
    context_words = torch.tensor([BERT_TOKENIZER.encode(context_words)])

    bio_base = [EMBEDER['O']]  # O to match with BERT's "CRT" Token  TODO: CRT? Or was it another shorten
    char_tracker = 0
    in_answer_section = False
    answer_words = answer_text.split(" ")
    answer_word_index = 1
    for word in context_text.split(" ")[0: 510]:
        if char_tracker == answer_start:
            bio_base.append(EMBEDER["B"])
            in_answer_section = True if len(answer_words) != 1 else False

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

    bio_base.append(EMBEDER["O"])  # End O for BERT's end token

    if len(ground_truth) > 1000:
        return None, None, None
    ground_truth = torch.tensor([BERT_TOKENIZER.encode(ground_truth)])

    assert len(bio_base) == len(context_words[0]), f'The BIO tags are not equal in length to the embeddings! ' \
                                                   f'{answer_info} & {len(bio_base)} & {len(context_words[0])}'
    return context_words, bio_base, ground_truth


def gen_data():
    overall_qas_idx = 0
    for overall_idx, _ in enumerate(data_json['data']):
        for paragraphs in data_json['data'][overall_idx]['paragraphs']:
            for qas_idx, question_answer in enumerate(paragraphs['qas']):
                if question_answer["is_impossible"]:
                    continue

                embedding, tags, ground_truth = _parse_context(paragraphs, qas_idx)  # TODO: Split input args up
                if embedding is None: continue
                tags = [tags]
                embedding = embedding.cpu().detach().numpy().tolist()
                ground_truth = ground_truth.cpu().detach().numpy().tolist()

                json_for_ex = {"context": embedding, "answer_tags": tags, "target": ground_truth}
                with open(f"train_set/item_{overall_qas_idx}.json", 'w') as file:
                    json.dump(json_for_ex, file)
                overall_qas_idx += 1


def start_generation():
    if os.path.exists("train_set"):
        print("Dataset already built!")
        return
        # try:
        #     with open("gen_info.json", 'r') as f:
        #         print("found a json!")  # TODO: Check if the date matches up
        # except FileNotFoundError:
        #     gen_data()
    else:
        os.mkdir("train_set")
        gen_data()
        gen_info = {"date_generated": str(datetime.datetime.now()), "punctuation": False, "impossible_questions": False}
        with open(f"train_set/gen_info.json", 'w') as file:
            json.dump(gen_info, file)


if __name__ == "__main__":
    start_generation()
