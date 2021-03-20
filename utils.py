import re
import string
import pandas as pd
import numpy as np
import torch
from difflib import SequenceMatcher
from tqdm import tqdm


def split_text(text):
    result = []
    punct = string.punctuation.translate(str.maketrans('', '', "-'"))
    word = ""
    space = ""
    for ch in text:
        if ch in punct:
            if word != "":
                result.append(word)
                word = ""

            if space != "":
                result.append(space)
                space = ""
            result.append(ch)

        elif ch == " ":
            space += ch
            if word != "":
                result.append(word)
                word = ""
        else:
            word += ch
            if space != "":
                result.append(space)
                space = ""

    if word != "":
        result.append(word)

    if space != "":
        result.append(space)

    return result


def add_to_label(address, start, end):
    punct = string.punctuation.translate(str.maketrans('', '', "-'")) + " "
    while start > 0 and address[start - 1] not in punct:
        start -= 1

    while end < len(address) and address[end] not in punct:
        end += 1

    return address[start: end]


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


def preprocess(dict_acronyms, text_word, label_word, pre_start=-1, pre_end=-1):
    new_text_word, new2old_text = remove_punct(text_word)
    new_label_word, new2old_label = remove_punct(label_word)
    if len(new_text_word) * len(new_label_word) == 0:
        return -1, -1

    start, end = -1, -1
    pos = -1
    max_ratio = 0
    for i in range(0, len(new_text_word) - len(new_label_word) + 1):
        if check_semilar(new_text_word[i: i + len(new_label_word)], new_label_word):
            ratio = SequenceMatcher(None, " ".join(new_text_word[i: i + len(new_label_word)]),
                                    " ".join(new_label_word)).ratio()

            if not (pre_end <= new2old_text[i] or new2old_text[i + len(new_label_word) - 1] + 1 <= pre_start):
                continue

            if ratio >= max_ratio:
                max_ratio = ratio
                pos = i
                start = new2old_text[i]
                end = new2old_text[i + len(new_label_word) - 1] + 1

    if pos >= 0:
        for i in range(pos, pos + len(new_label_word)):
            if new_label_word[i - pos] != new_text_word[i]:
                dict_acronyms[new_text_word[i]] = new_label_word[i - pos]

    return start, end


def convert_lines(data, tokenizer, max_sequence_length):
    index = np.zeros((len(data), max_sequence_length))
    label_ner = np.zeros((len(data), max_sequence_length))
    label_cf = np.zeros(len(data))

    subwords = []
    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, info in tqdm(enumerate(data), total=len(data)):
        address_word = info['raw_address']
        poi_start, poi_end = info['poi']
        street_start, street_end = info['street']

        address = " " + " ".join(address_word)
        input_ids = tokenizer(address)['input_ids']
        subword = tokenizer.convert_ids_to_tokens(input_ids)
        subwords.append(subword)
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
        lbl = [0] * (len(subword) - 2)
        for i, word in enumerate(subword[1:-1]):
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
        label_ner[idx, :] = np.array(lbl, dtype=np.long)

        if poi_start == -1 and street_start == -1:
            label_cf[idx] = 0

        else:
            label_cf[idx] = 1

    return index, label_ner, label_cf, subwords


def read_csv(path, test=False):
    df = pd.read_csv(path)
    data = []
    label = []

    if test:
        for idx, address in df.values.tolist():
            data.append(address)

        return data
    else:
        for idx, address, lbl in df.values.tolist():
            data.append(address)
            label.append(lbl)

        return data, label


def read_data(text, label, da=True):
    format_data = []
    dict_acronyms = {}
    raw_data, raw_label = [], []

    for address, poi_street in list(zip(text, label)):
        poi, street = poi_street.split("/")
        poi, street = poi.strip(), street.strip()
        raw_address = format_punctuatation(address)
        poi = format_punctuatation(poi)
        street = format_punctuatation(street)

        poi_start, poi_end = -1, -1
        street_start, street_end = -1, -1

        if len(street) >= len(poi):
            if street:
                street_start, street_end = preprocess(dict_acronyms, raw_address, street)
                if street_start == -1:
                    continue

            if poi:
                poi_start, poi_end = preprocess(dict_acronyms, raw_address, poi, street_start, street_end)
                if poi_start == -1:
                    continue
        else:
            if poi:
                poi_start, poi_end = preprocess(dict_acronyms, raw_address, poi)
                if poi_start == -1:
                    continue

            if street:
                street_start, street_end = preprocess(dict_acronyms, raw_address, street, poi_start, poi_end)
                if street_start == -1:
                    continue

        if street_start >= 0 and poi_start >= 0 and not (poi_end <= street_start or street_end <= poi_start):
            continue

        raw_data.append(address.strip())
        raw_label.append(poi_street.strip())

        format_data.append(
            {"raw_address": raw_address, "poi": (poi_start, poi_end), "street": (street_start, street_end)})

    if da:
        return format_data, raw_data, raw_label, dict_acronyms

    return format_data, raw_data, raw_label


def convert_lines_cf(data, tokenizer, label=None, max_sequence_length=64):
    index = np.zeros((len(data), max_sequence_length))
    label_cf = np.zeros(len(data))

    subwords = []
    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, address in tqdm(enumerate(data), total=len(data)):
        address = address.replace(",", "")
        address_word = format_punctuatation(address)
        address = " " + " ".join(address_word)
        input_ids = tokenizer(address)['input_ids']
        subword = tokenizer.convert_ids_to_tokens(input_ids)
        subwords.append(subword)

        if len(input_ids) > max_sequence_length:
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ] * (max_sequence_length - len(input_ids))

        index[idx, :] = np.array(input_ids, dtype=np.long)
        if label is not None:
            if label[idx].strip() == "/":
                label_cf[idx] = 0
            else:
                label_cf[idx] = 1

    if label is not None:
        return index, label_cf

    return index


def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_true)
    count = 0.
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            count += 1

    return count / len(y_true)
