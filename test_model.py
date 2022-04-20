import re
import json
import numpy as np
import os
import sys
from os.path import abspath, join, dirname
sys.path.extend([abspath(dirname(__file__)),
                 join(abspath(dirname(__file__)), 'data_processing'),
                 join(abspath(dirname(__file__)), 'inference_model'),
                 join(abspath(dirname(__file__)), 'loss_function'),
                 join(abspath(dirname(__file__)), 'utils'),
                 join(abspath(dirname(__file__)), 'model'),
                 join(abspath(dirname(__file__)), 'metrics'),
                 join(abspath(dirname(__file__)), 'train_config'),
                 join(abspath(dirname(__file__)), 'data')])

from inference_model.inference import NER
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW, get_linear_schedule_with_warmup
from data_process import yeild_data
from model import EfficientGlobalPointerNet as GlobalPointerNet
from loss_fun import global_pointer_crossentropy
from metrics import global_pointer_f1_score
from tqdm import tqdm
import configparser

from tools import setup_seed
from data_process import load_data

setup_seed(1234)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

file = join(abspath(dirname(__file__)), 'train_config/config.ini')
con = configparser.ConfigParser()
con.read(file, encoding='utf8')
items = con.items('path')
path = dict(items)
items = con.items('model_superparameter')
model_sp = dict(items)
model_path = path['model_path']
train_file_data = path['train_file_data']
val_file_data = path['val_file_data']
model_save_path = path['model_save_path']
head_size = eval(model_sp['head_size'])
hidden_size = eval(model_sp['hidden_size'])
learning_rate = eval(model_sp['learning_rate'])
clip_norm = eval(model_sp['clip_norm'])
re_maxlen = eval(model_sp['re_maxlen'])
train_dataloader, categories_size, categories2id, id2categories = yeild_data(train_file_data, is_train=True, DDP=False)
val_data = load_data(train_file_data, is_train=False)
val_dataloader = yeild_data(val_file_data, categories_size=categories_size, categories2id=categories2id, is_train=False,
                            DDP=False)
model = GlobalPointerNet(model_path, categories_size, head_size, hidden_size).to(device)

model.load_state_dict(torch.load(model_save_path))
while True:
    sent = input("请输入要识别的文本：")
    entities = NER.recognize(sent, id2categories, model)
    for e in entities:
        start_idx= e[0]
        end_idx = e[1]
        e_type = e[2]
        sent = sent[:start_idx] + ["<"] + sent[start_idx:end_idx+1] +[e_type+">"] +sent[end_idx+1:]
        print(sent)

