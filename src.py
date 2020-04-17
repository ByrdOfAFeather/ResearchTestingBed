import json
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.BasicEncoderDecoder import GRUDecoder, BidirectionalGRUEncoder
import os

torch.set_default_tensor_type('torch.cuda.FloatTensor')

BERT_MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
BERT_TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

BATCH_SIZE = 1


def train(encoder, decoder, encoder_optim, deocder_optim, criterion, data, epochs):
    encoder.train()
    decoder.train()
    # encoder.cuda()
    # decoder.cuda()
    cum_loss = 0
    index = 0
    for i in range(0, epochs):
        # try:
        encoder_optim.zero_grad()
        deocder_optim.zero_grad()
        loss = None
        for j in range(0, BATCH_SIZE):
            # try:
            #     batch = data[index]
            # except Exception:
            #     index = 0
            #     batch = data[index]
            # index += 1
            batch = next(iter(data))
            target_labels = torch.tensor(batch['target'])

            context_vec = BERT_MODEL(torch.tensor(batch['context']))[0]
            answer_tags = torch.tensor([batch['answer_tags']])
            output_vec = BERT_MODEL(target_labels)[0]

            max_out_idxs = batch['maxout']
            input_info = batch['input_info']

            x, attn = encoder(context_vec, answer_tags)
            x = decoder(output_vec, x, attn, max_out_idxs, input_info)

            if i % 1000 == 0:
                print("=====")
                print(f"TARGET: {target_labels}")
                print(f"ORIGINAL: {BERT_TOKENIZER.decode(target_labels[0])}")
                print(f"PRED: {BERT_TOKENIZER.decode(torch.argmax(torch.softmax(x[0], 1), dim=1))}")
                print("=====")
                torch.save(encoder.state_dict(), '_error_saves/encoder')
                torch.save(decoder.state_dict(), '_error_saves/decoder')

            target_labels.contiguous().view(-1)
            if loss is None:
                loss = criterion(x[0], target_labels[0])
            else:
                loss += criterion(x[0], target_labels[0])

        loss.backward()
        # for n, w in encoder.named_parameters():
        #     if w.grad is None:
        #         print("Detected None Gradient")
        #         print(n)
        #         continue
        #     else:
        #         pass
        #     if torch.sum(w.grad) == 0:
        #         print("0 gradient detected")
        #         print(n)
        #
        # for n, w in decoder.named_parameters():
        #     if w.grad is None:
        #         print("Detected None Gradient")
        #         print(n)
        #         continue
        #     else:
        #         pass
        #     if torch.sum(w.grad) == 0:
        #         print("0 gradient detected")
        #         print(n)

        encoder_optim.step()
        deocder_optim.step()

        cum_loss += loss.item()

        if i % 1000 == 0:
            print(i, cum_loss / 999)
            cum_loss = 0
        # except Exception as e:
        #     print(f"found error {e} saving model")
        #     torch.save(encoder.state_dict(), '_error_saves/encoder')
        #     torch.save(decoder.state_dict(), '_error_saves/decoder')
        #     break


class SQuADSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(os.listdir(self.data_path)) - 1  # 3-1 due to get info file

    def __getitem__(self, idx):
        with open(f"{self.data_path}/item_{idx}.json", 'r') as f:
            sample = json.load(f)

        return sample


def build_model_and_train():
    # TODO: Make it clear that this 3 comes from embedding layer
    encoder = BidirectionalGRUEncoder(input_size=768 + 3, hidden_size=600)
    encoder.init_weights()
    decoder = GRUDecoder(input_size=768, hidden_size=1200)
    decoder.init_weights()

    if os.path.exists("error_saves/encoder"):
        encoder.load_state_dict(torch.load("error_saves/encoder"))
    if os.path.exists("error_saves/decoder"):
        decoder.load_state_dict(torch.load("error_saves/decoder"))
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=.001)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=.001)
    criterion = nn.NLLLoss()

    # data = DataLoader(SQuADSet("train_set"), shuffle=True)
    data = DataLoader(SQuADSet("train_set"), shuffle=True)
    train(encoder, decoder, encoder_optim, decoder_optim, criterion, data, 250000)


if __name__ == "__main__":
    build_model_and_train()
