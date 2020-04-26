import json
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/teacher_forcing_havled")

from models.BasicEncoderDecoder import AttnGruDecoder, BiAttnGRUEncoder
import os

torch.set_default_tensor_type('torch.cuda.FloatTensor')

BERT_MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
BERT_TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

BATCH_SIZE = 5


def train(encoder, decoder, encoder_optim, deocder_optim, criterion, data, epochs):
    """Trains a given encoder and decoder for the number of epcohs provided
	:param encoder: A model that encodes sentences
	:param decoder: A model that decodes from a previous encoder hidden state
	:param encoder_optim: Optimizer
	:param deocder_optim: Optimizer
	:param criterion: CrossEntopyLoss()
	:param data: QADataset
	:param epochs: Number of iterations for training
	:return: None
	"""
    encoder.train()
    decoder.train()
    cum_loss = 0
    for i in range(0, epochs):
        try:
            encoder_optim.zero_grad()
            deocder_optim.zero_grad()
            loss = None
            for j in range(0, BATCH_SIZE):
                batch = next(iter(data))
                target_labels = torch.tensor(batch['target'])

                # Gets word vectors that encode the meaning of the word (from BERT model)
                # for more information on word vectors see: https://dzone.com/articles/introduction-to-word-vectors
                context_vec = BERT_MODEL(torch.tensor(batch['context']))[0]
                answer_tags = torch.tensor([batch['answer_tags']])
                output_vec = BERT_MODEL(target_labels)[0]

                x, attn = encoder(context_vec, answer_tags)
                x = decoder(output_vec, x, attn)

                if i % 1000 == 0:
                    print("=====")
                    print(f"TARGET: {target_labels}")
                    print(f"ORIGINAL: {BERT_TOKENIZER.decode(target_labels[0])}")
                    print(f"PRED: {BERT_TOKENIZER.decode(torch.argmax(torch.softmax(x[0], 1), dim=1))}")
                    print("=====")
                    # Saves the model every 1000 iterations
                    # prints the current sample and prediction for it
                    # It also prints the loss but that is later in the code
                    torch.save(encoder.state_dict(), f'pre_trained/weight_saves/encoder_{i}')
                    torch.save(decoder.state_dict(), f'pre_trained/weight_saves/decoder_{i}')

                target_labels.contiguous().view(-1)
                if loss is None:
                    loss = criterion(x[0], target_labels[0])
                else:
                    loss += criterion(x[0], target_labels[0])

            # This calculates the gradients for all parameters in the encoder and decoder
            loss.backward()

            # This applies all the gradients for the encoder and decoder
            encoder_optim.step()
            deocder_optim.step()

            # This adds the numerical loss (adding loss objects fills up GPU memory very quickly)
            cum_loss += loss.item() / BATCH_SIZE

            if i % 1000 == 0 and i != 0:
                print(i, cum_loss / 999)
                writer.add_scalar('training loss',
                                  cum_loss / 1000,
                                  i * len(data) + i)

                cum_loss = 0
        except Exception as e:
            # Saves the model if any error occurs
            print(f"found error {e} saving model")
            torch.save(encoder.state_dict(), 'pre_trained/weight_saves/encoder')
            torch.save(decoder.state_dict(), 'pre_trained/weight_saves/decoder')
            break


def test(encoder, decoder, input_data):
    target_labels = torch.tensor(input_data['target'])

    context_vec = BERT_MODEL(torch.tensor(input_data['context']))[0]
    answer_tags = torch.tensor([input_data['answer_tags']])
    output_vec = BERT_MODEL(target_labels)[0]

    encoder.train(False)
    decoder.train(False)

    x, attn = encoder(context_vec, answer_tags)
    x = decoder(output_vec, x, attn)
    return x


class QADataset(Dataset):
    """Generic dataset that can be used in question answering tasks
	In general, the files should be formatted as so:
		context: the paragraph that supports the answer or question (BERT indexes expected but not required)
		answer: the BIO tags for a particular answer
		question: the question (BERT indexes or equivalent expected)
	"""

    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(os.listdir(self.data_path)) - 1  # -1 comes from a generation file

    def __getitem__(self, idx):
        with open(f"{self.data_path}/item_{idx}.json", 'r') as f:
            sample = json.load(f)

        return sample


def build_model_and_train():
    # TODO: Make it clear that this 3 comes from embedding layer

    # Init encoder and decoder models
    # The input size is the size of the BERT embeddings (for a single word) plus 3 for the BIO embeddings
    # The hidden size is a parameter of any RNN, it can be thought of the space that BERT words are projected into
    # That's a bit abstract, but, it is essentially where the model learns to represent the sentence at a particular
    # word.
    # If this space is very large, it's possible for the model not to learn well as it won't find important details
    # and instead just encapsulate everything as-is. If this space is very small the model might be unable to learn
    # as it simply can't find what is important in the data. The size 600 here comes from the original paper and is
    # what they found to be best.
    encoder = BiAttnGRUEncoder(input_size=768 + 3, hidden_size=600)
    encoder.init_weights()

    # The hidden size is notably doubled here due to the encoder being bi-directional. The decoder also doesn't take
    # BIO tags as input. Instead it takes the previously predicted word, or in the case of teacher forcing, the ground
    # truth.
    decoder = AttnGruDecoder(input_size=768, hidden_size=1200, teacher_ratio=.5)
    decoder.init_weights()

    # This line loads weights if they are already present
    if os.path.exists("pre_trained/sciq_weights/encoder"):
        encoder.load_state_dict(torch.load("pre_trained/sciq_weights/encoder_249000"))
    if os.path.exists("pre_trained/sciq_weights/decoder"):
        decoder.load_state_dict(torch.load("pre_trained/sciq_weights/decoder_249000"))

    # These optimizers take care of adjusting learning rate according to gradient size
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=.001)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=.001)

    # Words are treated as classes and the output of the model is a probability distribution of these classes for
    # each word in the output.
    criterion = nn.CrossEntropyLoss()

    # This creates a dataset compatible with pytorch that auto-shuffles and we don't have to worry about
    # indexing errors
    data = DataLoader(QADataset("data/sciq_train_set"), shuffle=True)

    if not os.path.exists("pre_trained/weight_saves"): os.mkdir("pre_trained/weight_saves")

    # print("EXAMPLE OUTPUT USAGE")
    # x = test(encoder, decoder, next(iter(data)))
    # print(x)

    train(encoder, decoder, encoder_optim, decoder_optim, criterion, data, 250000)


if __name__ == "__main__":
    build_model_and_train()
