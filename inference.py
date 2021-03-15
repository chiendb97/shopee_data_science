import argparse
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaConfig

from models import RobertaForTokenClassification
from utils import format_punctuatation, seed_everything


def read_csv(path):
    df = pd.read_csv(path).head(100)
    data = []
    for idx, text in df.values.tolist():
        data.append(text)

    return data


def text_to_index(data, tokenizer, max_sequence_length):
    cls_id = 0
    eos_id = 2
    pad_id = 1

    subwords = []
    index = np.zeros((len(data), max_sequence_length))
    for idx, address in tqdm(enumerate(data), total=len(data)):
        address_word = format_punctuatation(address)
        address = " " + " ".join(address_word)
        input_ids = tokenizer(address)['input_ids']
        subwords = tokenizer.convert_ids_to_tokens(input_ids)

        if len(input_ids) > max_sequence_length:
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ] * (max_sequence_length - len(input_ids))

        index[idx, :] = np.array(input_ids, dtype=np.long)
        subwords.append(subwords)

    return index, subwords


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_path', type=str, default='./data/test.csv')
    parser.add_argument('--model_name', type=str, default='cahya/roberta-base-indonesian-522M')
    parser.add_argument('--max_sequence_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--ckpt_path', type=str, default='./models')

    args = parser.parse_args()

    seed_everything(69)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_bert = torch.load(os.path.join(args.ckpt_path, f"model.bin"))
    data = read_csv(args.test_path)
    index, subwords = text_to_index(data, tokenizer, args.max_sequence_length)

    test_dataset = torch.utils.data.TensorDataset((torch.tensor(index, dtype=torch.long)))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model_bert.eval()

    pred = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
    for i, (x_batch,) in pbar:
        mask = (x_batch != 1)
        with torch.no_grad():
            y_hat = model_bert(x_batch.to(device), attention_mask=mask.to(device))

        y_pred = torch.argmax(y_hat, 2)
        pred += y_pred.detach().cpu().numpy().tolist()


if __name__ == '__main__':
    main()
