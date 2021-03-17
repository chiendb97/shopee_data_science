import json

from transformers import RobertaTokenizer
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import RobertaConfig, get_constant_schedule, get_linear_schedule_with_warmup
from transformers import AdamW

from inference import sufprocess
from models import RobertaForTokenClassification
from utils import convert_lines, seed_everything, read_data, accuracy_score


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_path', type=str, default='./data/train.csv')
    parser.add_argument('--outout_path', type=str, default='./data/output.csv')
    parser.add_argument('--dict_acronyms_path', type=str, default='./data/dict_acronyms.json')
    parser.add_argument('--model_name', type=str, default='cahya/roberta-base-indonesian-522M')
    parser.add_argument('--max_sequence_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--ckpt_path', type=str, default='./models')

    args = parser.parse_args()

    seed_everything(69)

    # Load model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    config = RobertaConfig.from_pretrained(
        args.model_name,
        output_hidden_states=True,
        num_labels=5
    )

    model_bert = RobertaForTokenClassification.from_pretrained(args.model_name, config=config)
    model_bert.cuda()

    if torch.cuda.device_count():
        print(f"Training using {torch.cuda.device_count()} gpus")
        model_bert = nn.DataParallel(model_bert)
        tsfm = model_bert.module.roberta
    else:
        tsfm = model_bert.roberta

    print("Read data ...")
    data_train, raw_data, raw_label, dict_acronyms = read_data(args.train_path)

    with open(args.dict_acronyms_path, "w") as f:
        json.dump(dict_acronyms, f)

    print("Convert line ...")
    x_train, y_train, subwords = convert_lines(data_train, tokenizer, args.max_sequence_length)

    x_train, x_valid, y_train, y_valid, subwords_train, subwords_valid, data_train, data_valid, label_train, label_valid \
        = train_test_split(x_train, y_train, subwords, raw_data, raw_label, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.long),
                                                   torch.tensor(y_train, dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(x_valid, dtype=torch.long),
                                                   torch.tensor(y_valid, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Creating optimizer and lr schedulers
    param_optimizer = list(model_bert.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = int(args.epochs * len(data_train) / args.batch_size / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
    scheduler0 = get_constant_schedule(optimizer)  # PyTorch scheduler

    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)

    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = False

    frozen = True
    best_score = 0.

    for epoch in range(args.epochs):
        if epoch > 0 and frozen:
            for child in tsfm.children():
                for param in child.parameters():
                    param.requires_grad = True

            frozen = False
            del scheduler0
            torch.cuda.empty_cache()

        avg_loss = 0.

        optimizer.zero_grad()
        model_bert.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i, (x_batch, y_batch) in pbar:
            mask = (x_batch != 1)
            y_hat, loss = model_bert(x_batch.cuda(), attention_mask=mask.cuda(), labels=y_batch)
            loss.backward()
            if i % args.accumulation_steps == 0 or i == len(pbar) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if not frozen:
                    scheduler.step()
                else:
                    scheduler0.step()
            lossf = loss.item()
            pbar.set_postfix(loss=lossf)
            avg_loss += loss.item() / len(train_loader)

        print("------------------------------- Training epoch {} ----------------------------".format(epoch + 1))
        print(f"\nTrain avg loss = {avg_loss:.4f}")

        model_bert.eval()
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
        output = []
        preds = []
        matrix_pred = []
        avg_loss = 0.

        for i, (x_batch, y_batch) in pbar:
            mask = (x_batch != 1)
            with torch.no_grad():
                y_hat, loss = model_bert(x_batch.cuda(), attention_mask=mask.cuda(), labels=y_batch)

            y_pred = torch.argmax(y_hat, 2)
            matrix_pred += y_pred.detach().cpu().numpy().tolist()
            output += y_batch[mask].detach().cpu().numpy().tolist()
            preds += y_pred[mask].detach().cpu().numpy().tolist()
            lossf = loss.item()
            pbar.set_postfix(loss=lossf)
            avg_loss += loss.item() / len(valid_loader)

        index, label_pred = sufprocess(dict_acronyms, data_valid, subwords_valid, matrix_pred)
        score = accuracy_score(label_valid, label_pred)
        precision, recall, f1_score, support = precision_recall_fscore_support(output, preds)
        print(f"\nValid avg loss = {avg_loss:.4f}")
        print(f"\nValid accuracy score = {score:.4f}")
        print(f"\nPrecision:", precision)
        print(f"\nRecall:", recall)
        print(f"\nF1 score:", f1_score)
        print(f"\nSupport:", support)
        if score >= best_score:
            torch.save(model_bert, os.path.join(args.ckpt_path, "model.pt"))
            best_score = score
            df = pd.DataFrame({"address": data_valid, "label": label_valid, "pred": label_pred})
            df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()
