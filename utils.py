import re
import string
import pandas as pd
import numpy as np
import torch
from difflib import SequenceMatcher
from tqdm import tqdm


def format_punctuatation(text):
    punct = string.punctuation.translate(str.maketrans('', '', "-'"))
    for ch in punct:
        text = text.replace(ch, " " + ch + " ")

    return text.strip().split()


def remove_punct(words):
    new_words = []
    new2old = []
    for i, word in enumerate(words):
        if word not in string.punctuation:
            new_words.append(word)
            new2old.append(i)

    return new_words, new2old


def check_semilar(source, target):
    for i in range(len(source)):
        # source_rm_punct = source[i].translate(str.maketrans('', '', '.,'))
        # target_rm_punct = target[i].translate(str.maketrans('', '', '.,'))

        if not target[i].startswith(source[i]):
            return False

    return True


def preprocess(text_word, label_word, pre_start=-1, pre_end=-1):
    new_text_word, new2old_text = remove_punct(text_word)
    new_label_word, new2old_label = remove_punct(label_word)
    if len(new_text_word) * len(new_label_word) == 0:
        return -1, -1

    start, end = -1, -1
    max_ratio = 0
    for i in range(0, len(new_text_word) - len(new_label_word) + 1):

        if check_semilar(new_text_word[i: i + len(new_label_word)], new_label_word):
            ratio = SequenceMatcher(None, " ".join(new_text_word[i: i + len(new_label_word)]),
                                    " ".join(new_label_word)).ratio()

            if not (pre_end <= new2old_text[i] or new2old_text[i + len(new_label_word) - 1] + 1 <= pre_start):
                continue

            if ratio >= max_ratio:
                max_ratio = ratio
                start = new2old_text[i]
                end = new2old_text[i + len(new_label_word) - 1] + 1

    return start, end


def convert_lines(data, tokenizer, max_sequence_length):
    index = np.zeros((len(data), max_sequence_length))
    label = np.zeros((len(data), max_sequence_length))
    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, info in tqdm(enumerate(data), total=len(data)):
        address_word = info['raw_address']
        poi_start, poi_end = info['poi']
        street_start, street_end = info['street']

        address = " " + " ".join(address_word)
        input_ids = tokenizer(address)['input_ids']
        subwords = tokenizer.convert_ids_to_tokens(input_ids)
        lbl_raw = [0] * len(address_word)

        if poi_start >= 0:
            lbl_raw[poi_start] = 1
            for i in range(poi_start + 1, poi_end):
                lbl_raw[i] = 2

        if street_start >= 0:
            lbl_raw[street_start] = 3
            for i in range(street_start + 1, street_end):
                lbl_raw[i] = 4

        k = -1
        lbl = [0] * (len(subwords) - 2)
        for i, word in enumerate(subwords[1:-1]):
            if word.startswith("Ġ"):
                k += 1

            if (lbl_raw[k] == 1 or lbl_raw[k] == 3) and not word.startswith("Ġ"):
                lbl[i] = lbl_raw[k] + 1
            else:
                lbl[i] = lbl_raw[k]

        lbl = [0] + lbl + [0]

        assert len(input_ids) == len(lbl)

        if len(input_ids) > max_sequence_length:
            input_ids = input_ids[:max_sequence_length]
            lbl = lbl[:max_sequence_length]
            input_ids[-1] = eos_id
            lbl[-1] = 0
        else:
            input_ids = input_ids + [pad_id, ] * (max_sequence_length - len(input_ids))
            lbl = lbl + [0, ] * (max_sequence_length - len(lbl))

        index[idx, :] = np.array(input_ids, dtype=np.long)
        label[idx, :] = np.array(lbl, dtype=np.long)

    return index, label


def read_data(path):
    format_data = []
    data = pd.read_csv(path)
    for ide, raw_address, poi_street in data.values.tolist():
        poi, street = poi_street.split("/")
        raw_address = format_punctuatation(raw_address)
        poi = format_punctuatation(poi)
        street = format_punctuatation(street)

        poi_start, poi_end = -1, -1
        street_start, street_end = -1, -1

        if len(street) >= len(poi):
            if street:
                street_start, street_end = preprocess(raw_address, street)
                if street_start == -1:
                    continue

            if poi:
                poi_start, poi_end = preprocess(raw_address, poi, street_start, street_end)
                if poi_start == -1:
                    continue
        else:
            if poi:
                poi_start, poi_end = preprocess(raw_address, poi)
                if poi_start == -1:
                    continue

            if street:
                street_start, street_end = preprocess(raw_address, street, poi_start, poi_end)
                if street_start == -1:
                    continue

        if street_start >= 0 and poi_start >= 0 and not (poi_end <= street_start or street_end <= poi_start):
            continue

        format_data.append(
            {"raw_address": raw_address, "poi": (poi_start, poi_end), "street": (street_start, street_end)})

    return format_data


def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
