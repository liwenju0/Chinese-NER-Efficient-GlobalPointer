import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, PreTrainedTokenizerFast, BertTokenizerFast
import json
from utils.tools import search
from utils.tools import token_rematch
import configparser
from os.path import abspath, join, dirname
file = join(abspath(dirname(__file__)), '../train_config/config.ini')
con = configparser.ConfigParser()
con.read(file, encoding='utf8')
items = con.items('path')
path = dict(items)
items = con.items('model_superparameter')
model_sp = dict(items)
model_path = path['model_path']
maxlen = eval(model_sp['maxlen'])
batch_size = eval(model_sp['batch_size'])
from torch.utils.data.distributed import DistributedSampler

tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)

def load_data(filename, is_train=True):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    categories = set()
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = [l['text']]
            for k, v in l['label'].items():
                categories.add(k)
                for spans in v.values():
                    for start, end in spans:
                        d.append((start, end, k))
            D.append(d)
    categories = list(sorted(categories))
    return D, categories if is_train else D


class NerDataset(Dataset):
    def __init__(self, data, tokenizer, categories_size, categories2id):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.categories_size = categories_size
        self.categories2id = categories2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        label = torch.zeros((self.categories_size, self.maxlen, self.maxlen))
        context = tokenizer(d[0], return_offsets_mapping=True, max_length=self.maxlen, truncation=True,
                            padding='max_length', return_tensors='pt')
        tokens = tokenizer.tokenize(d[0], max_length=self.maxlen, add_special_tokens=True)
        mapping = token_rematch().rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        for entity_input in d[1:]:
            start, end = entity_input[0], entity_input[1]
            if start in start_mapping and end in end_mapping and start < self.maxlen and end < self.maxlen:
                start = start_mapping[start]
                end = end_mapping[end]
                label[self.categories2id[entity_input[2]], start, end] = 1
        # label = 
        return context, label

def yeild_data(train_file_data, is_train, categories_size=None, categories2id=None, DDP=True):
    if is_train:
        train_data, categories = load_data(train_file_data, is_train=is_train)
        categories_size = len(categories)
        categories2id = {c: idx for idx, c in enumerate(categories)}
        id2categories = {idx: c for idx, c in enumerate(categories)}
        train_data = NerDataset(train_data, tokenizer, categories_size, categories2id)
        if DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, shuffle=False)
        else:
            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_dataloader, categories_size, categories2id, id2categories
    else:
        train_data = load_data(train_file_data, is_train=is_train)
        train_data = NerDataset(train_data, tokenizer, categories_size, categories2id)
        if DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        else:
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
        return train_dataloader

