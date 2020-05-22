"""
Basic preprocessing script for the SQuAD dataset. Expects the SQuAD dataset to already exist in the test directory.
This is not automatically included in the package due to the size of the dataset.
"""
import shutil

import torch

import json
import string
import os
import datetime
import sys

# from CONFIG import CONSOLE_COLORS, BERT_ENCODER

DATA_PATH = "data"

topics = []
topic_sets = {}

EMBEDER = {
    "B": 1,
    "I": 2,
    "O": 0,
}

EXCEPTIONS = {"\"", "("}

TOTAL = 0


def _parse_context(paragraph, current_question, include_punc=False):
    global TOTAL
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

    bio_base = []  # O to match with BERT's "CRT" Token  TODO: CRT? Or was it another shorten
    char_tracker = 0
    in_answer_section = False
    answer_words = answer_text.split(" ")
    answer_word_index = 0
    start_index = 0

    for idx, word in enumerate(context_text.split(" ")):
        # Answer_start is a bit inaccurate as it can start in the middle of words in the case of
        # arbitrary prefixes, some examples: $, (, top-, etc
        if char_tracker == answer_start or (
                answer_words[0] in word and word.index(answer_words[0]) is not 0 or word == "50-50"):
            start_index = idx
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

    # Embed words
    context_words = context_words[max(0, start_index - 255): start_index + 255]
    context_words = torch.tensor([BERT_ENCODER.encode(context_words)])
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
    ground_truth = torch.tensor([BERT_ENCODER.encode(ground_truth)])

    assert len(bio_base) == len(context_words[0]), f'The BIO tags are not equal in length to the embeddings! ' \
                                                   f'{answer_info} & {len(bio_base)} & {len(context_words[0])}'
    return context_words, bio_base, ground_truth


def generate_squad():
    global TOTAL
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
                embedding = embedding.cpu().detach().numpy().tolist()
                ground_truth = ground_truth.cpu().detach().numpy().tolist()

                json_for_ex = {"context": embedding, "answer_tags": tags, "target": ground_truth}
                with open(f"data/squad_train_set/item_{overall_qas_idx}.json", 'w') as file:
                    json.dump(json_for_ex, file)
                overall_qas_idx += 1
    print(TOTAL)


def check_and_gen_squad():
    if torch.cuda.is_available():
        print(f"{CONSOLE_COLORS.OK_GREEN}Found GPU: {torch.cuda.get_device_name(0)} {CONSOLE_COLORS.ENDC}")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print(f"{CONSOLE_COLORS.WARNING}No GPU found, "
              f"this could significantly increase computation time!{CONSOLE_COLORS.ENDC}")

    if not os.path.exists("data"):
        print(
            f"{CONSOLE_COLORS.FAIL}SQuAD DATASET NOT FOUND, PLEASE DOWNLOAD THE DATA AND PUT IT IN "
            f"data/train-v2.0.json{CONSOLE_COLORS.ENDC}")
        sys.exit(0)
    if not os.path.exists("data/train-v2.0.json"):
        print(
            f"{CONSOLE_COLORS.FAIL}SQuAD DATASET NOT FOUND, PLEASE DOWNLOAD THE DATA AND PUT IT IN "
            f"data/train-v2.0.json{CONSOLE_COLORS.ENDC}")
        sys.exit(0)
    print(f"{CONSOLE_COLORS.OK_GREEN}Found data, starting generation, please be patient!{CONSOLE_COLORS.ENDC}")
    if os.path.exists("data/squad_train_set"):
        print("Dataset already built!")
        return
    else:
        os.mkdir("data/squad_train_set")
        generate_squad()
        gen_info = {"date_generated": str(datetime.datetime.now()),
                    "punctuation": False,
                    "impossible_questions": False,
                    "additional_notes": "Inputs larger than 510 tokens were discarded."}
        with open(f"data/squad_train_set/gen_info.json", 'w') as file:
            json.dump(gen_info, file)


if __name__ == "__main__":
    check_and_gen_squad()
