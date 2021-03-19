import json
from itertools import chain
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
from utils import convert_lines, seed_everything, read_data, accuracy_score, read_csv
from augment import augment_punct, augment_replace_address


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_path', type=str, default='./data/train.csv')
    parser.add_argument('--dict_acronyms_path', type=str, default='./data/dict_acronyms.json')
    parser.add_argument('--model_name', type=str, default='cahya/roberta-base-indonesian-522M')
    parser.add_argument('--activation_function', type=str, default='softmax')
    parser.add_argument('--loss_type', type=str, default='lsr')
    parser.add_argument('--max_sequence_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--num_multiply', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--ckpt_path', type=str, default='./models')

    args = parser.parse_args()
    assert args.activation_function in ['softmax', 'crf']
    seed_everything(69)

    # Load model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    config = RobertaConfig.from_pretrained(
        args.model_name,
        num_labels=5,
        output_hidden_states=True
    )

    model_bert = RobertaForTokenClassification.from_pretrained(args.model_name, config=config,
                                                               activation_function=args.activation_function,
                                                               loss_type=args.loss_type)

    model_bert.cuda()
    if torch.cuda.device_count():
        print(f"Training using {torch.cuda.device_count()} gpus")
        model_bert = nn.DataParallel(model_bert)
        tsfm = model_bert.module.roberta
    else:
        tsfm = model_bert.roberta

    print("\nRead data ...")

    data_train, label_train = read_csv(args.train_path)
    data_train, data_valid, label_train, label_valid = train_test_split(data_train, label_train, test_size=0.2,
                                                                        random_state=42)

    data_train_ap, label_train_ap = augment_punct(data_train, label_train)

    data_train, text_train, label_train, dict_acronyms = read_data(data_train, label_train)
    data_train_ap, text_train_ap, label_train_ap = read_data(data_train_ap, label_train_ap, da=False)
    data_train_ar, label_train_ar = augment_replace_address(data_train, label_train, num_multiply=args.num_multiply)

    data_train = data_train + data_train_ap + data_train_ar
    label_train = label_train + label_train_ap + label_train_ar

    data_valid, text_valid, label_valid = read_data(data_valid, label_valid, da=False)

    with open(args.dict_acronyms_path, "w") as f:
        json.dump(dict_acronyms, f)

    print("\nConvert line ...")
    x_train, y_ner_train, y_cf_train, subwords_train = convert_lines(data_train, tokenizer, args.max_sequence_length)
    x_valid, y_ner_valid, y_cf_valid, subwords_valid = convert_lines(data_valid, tokenizer, args.max_sequence_length)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.long),
                                                   torch.tensor(y_ner_train, dtype=torch.long),
                                                   torch.tensor(y_cf_train, dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(x_valid, dtype=torch.long),
                                                   torch.tensor(y_ner_valid, dtype=torch.long),
                                                   torch.tensor(y_cf_valid, dtype=torch.long))

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
        for i, (x_batch, y_ner_batch, y_cf_batch) in pbar:
            mask = (x_batch != 1)
            y_hat_ner, loss_ner = model_bert(x_batch.cuda(), attention_mask=mask.cuda(), labels_ner=y_ner_batch.cuda())

            loss_ner.backward()
            if i % args.accumulation_steps == 0 or i == len(pbar) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if not frozen:
                    scheduler.step()
                else:
                    scheduler0.step()

            pbar.set_postfix(loss_ner=loss_ner.item())
            avg_loss += loss_ner.item() / len(train_loader)

        print("------------------------------- Training epoch {} -------------------------------".format(epoch + 1))
        print(f"\nTrain avg loss = {avg_loss:.4f}")

        model_bert.eval()
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
        output_ner = []
        pred_ner = []
        matrix_pred = []
        avg_loss = 0.

        for i, (x_batch, y_ner_batch) in pbar:
            mask = (x_batch != 1)
            with torch.no_grad():
                y_hat_ner, loss_ner = model_bert(x_batch.cuda(), attention_mask=mask.cuda(),
                                                 labels_ner=y_ner_batch.cuda())

            if args.activation_function == 'softmax':
                y_pred_ner = torch.argmax(y_hat_ner, 2)
                matrix_pred += y_pred_ner.detach().cpu().numpy().tolist()
                output_ner += y_ner_batch[mask].detach().cpu().numpy().tolist()
                pred_ner += y_pred_ner[mask].detach().cpu().numpy().tolist()

            else:
                y_pred_ner = model_bert.module.crf.decode(y_hat_ner, mask.cuda())
                matrix_pred += y_pred_ner
                output_ner += y_ner_batch[mask].detach().cpu().numpy().tolist()
                pred_ner += list(chain.from_iterable(y_pred_ner))

            pbar.set_postfix(loss_ner=loss_ner.item())
            avg_loss += loss_ner.item() / len(valid_loader)

        index, label_pred = sufprocess(dict_acronyms, text_valid, subwords_valid, matrix_pred)
        score_ner = accuracy_score(label_valid, label_pred)
        precision, recall, f1_score, support = precision_recall_fscore_support(output_ner, pred_ner)
        print(f"\nValid avg loss = {avg_loss:.4f}")
        print(f"\nValid accuracy score = {score_ner:.4f}")
        print(f"\nPrecision:", precision)
        print(f"\nRecall:", recall)
        print(f"\nF1 score:", f1_score)
        print(f"\nSupport:", support)
        if score_ner >= best_score:
            torch.save(model_bert, os.path.join(args.ckpt_path, args.activation_function + "_" + "model_ner.pt"))
            best_score = score_ner
            df = pd.DataFrame({"address": text_valid, "label": label_valid, "pred": label_pred})
            df.to_csv("./data/output_{}.csv".format(args.activation_function), index=False)


if __name__ == '__main__':
    main()
