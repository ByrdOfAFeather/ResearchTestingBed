import json
import os

import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from models.BasicEncoderDecoder import GRUDecoder, BidirectionalGRUEncoder
from models.CustomSeq2Seq import GatedQuestionAnswering

BERT_MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
BERT_TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

BATCH_SIZE = 1


def train(encoder, decoder, encoder_optim, deocder_optim, criterion, data, epochs):
    encoder.train()
    decoder.train()
    cum_loss = 0
    index = 0
    for i in range(0, epochs):
        try:
            encoder_optim.zero_grad()
            deocder_optim.zero_grad()
            loss = None
            for j in range(0, BATCH_SIZE):
                try:
                    batch = data[index]
                except Exception:
                    index = 0
                    batch = data[index]
                index += 1
                # batch = next(iter(data))
                target_labels = torch.tensor(batch['target'])

                context_vec = BERT_MODEL(torch.tensor([batch['context'][0][0:250]]))[0]
                answer_tags = torch.tensor(batch['answer_tags'])
                output_vec = BERT_MODEL(target_labels)[0]

                x = encoder(context_vec, answer_tags)
                x = decoder(output_vec, x)

                if i % 100 == 0:
                    print("=====")
                    print(f"TARGET: {target_labels}")

                    print(f"ORIGINAL: {BERT_TOKENIZER.decode(target_labels[0])}")
                    print(f"PRED: {BERT_TOKENIZER.decode(torch.argmax(torch.softmax(x[0], 1), dim=1))}")
                    print("=====")

                target_labels.contiguous().view(-1)
                if loss is None:
                    loss = criterion(x[0], target_labels[0])
                else:
                    loss += criterion(x[0], target_labels[0])

            loss.backward()
            for n, w in encoder.named_parameters():
                if w.grad is None:
                    print("Detected None Gradient")
                    print(n)
                    continue
                else:
                    pass
                if torch.sum(w.grad) == 0:
                    print("0 gradient detected")
                    print(n)

            encoder_optim.step()
            deocder_optim.step()

            cum_loss += loss.item()

            if i % 100 == 0:
                print(i, cum_loss / 99)
                cum_loss = 0
        except Exception as e:
            print(i)
            print(index)
            print("ERROR")
            print(e)
            print([batch['context'][0][0:250]])
            break


def train_seq2seq(model, optimizer, criterion, data, epochs):
    model.train()
    for i in range(0, epochs):
        print("------------------")
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss = None
        for j in range(0, BATCH_SIZE):
            batch = next(iter(data))
            target_labels = torch.tensor(batch['target'])

            input_vec = BERT_MODEL(torch.tensor(batch['context']))[0][:, 0:20, :]
            output_vec = BERT_MODEL(target_labels)[0][0]
            output_vec = output_vec.detach()
            x = model(output_vec, output_vec)

            print("=====")
            print(f"TARGET: {target_labels}")
            print(f"ORIGINAL: {BERT_TOKENIZER.decode(target_labels[0])}")
            print(f"PRED: {BERT_TOKENIZER.decode(torch.argmax(torch.softmax(x, 1), dim=1))}")
            print("=====")

            target_labels.contiguous().view(-1)
            if loss is None:
                loss = criterion(x, target_labels[0])
            else:
                loss += criterion(x, target_labels[0])

        loss.backward()
        for n, w in model.named_parameters():
            if w.grad is not None:
                if torch.sum(w.grad) == 0:
                    print("0 gradient detected")
                    print(n)

        optimizer.step()

        print(i, loss)
        print("------------------")


class SQuADSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(os.listdir(self.data_path)) - 1  # 3-1 due to get info file

    def __getitem__(self, idx):
        with open(f"{self.data_path}/item_{idx}.json", 'r') as f:
            sample = json.load(f)

        return sample


if __name__ == "__main__":
    encoder = BidirectionalGRUEncoder(input_size=768, hidden_size=600)

    decoder = GRUDecoder(input_size=768, hidden_size=1200)
    encoder_optim = torch.optim.Adam(encoder.parameters())
    decoder_optim = torch.optim.Adam(decoder.parameters())
    criterion = nn.CrossEntropyLoss()

    # data = DataLoader(SQuADSet("train_set"), shuffle=True)
    data = SQuADSet("train_set")
    train(encoder, decoder, encoder_optim, decoder_optim, criterion, data, 250000)
