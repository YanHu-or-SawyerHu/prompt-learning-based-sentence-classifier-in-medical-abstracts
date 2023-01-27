bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
percentage_of_training_data = 0.05
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '10'

import json, requests
from bs4 import BeautifulSoup
import glob,random
from shutil import copyfile
classes = ['OBJECTIVE','METHODS','BACKGROUND','CONCLUSIONS','RESULTS']
# read data
data_dir = '/collab/yhu5/PICO_sentence_classification/HSLN-Joint-Sentence-Classification/data/RCT_with_title/'
train =  data_dir + 'PICO_train.txt'
valid = data_dir+'PICO_dev.txt'
test = data_dir+'PICO_test.txt'

def get_train_lenth(file):
    PMID = []
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('###'):
                PMID.append(line[3:].strip())
    return PMID

def read_whole(file):
    file_dict = {'text':[],'label':[]}
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('###') and line != '\n' and line.split('\t')[0]!='Title':
                file_dict['text'].append(line.split('\t')[1])
                file_dict['label'].append(classes.index(line.split('\t')[0]))

    return file_dict
    
def read_part(file,stop_PMID):
    PMID = '0'
    file_dict = {'text':[],'label':[]}
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            if PMID == stop_PMID:
                break
            else:
                if not line.startswith('###') and line != '\n' and line.split('\t')[0]!='Title':
                    file_dict['text'].append(line.split('\t')[1])
                    file_dict['label'].append(classes.index(line.split('\t')[0]))
                elif line.startswith('###'):
                    PMID = line[3:].strip()
    return file_dict
    
total_train_PMID = get_train_lenth(train)
length = round(len(total_train_PMID)*percentage_of_training_data)
stop_PMID = total_train_PMID[length]

train_dict = read_part(train,stop_PMID)
test_dict = read_whole(test)
valid_dict = read_whole(valid)

from datasets import Dataset
train = Dataset.from_dict(train_dict)
test = Dataset.from_dict(test_dict)
valid = Dataset.from_dict(valid_dict)

from datasets import load_dataset

imdb = load_dataset("imdb")

all_dataset = imdb
all_dataset['train']=train
all_dataset['valid']=valid
all_dataset['test']=test
all_dataset.pop('unsupervised')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(bert_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length = 512)
    
tokenized_dataset = all_dataset.map(preprocess_function, batched=True)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
device = torch.device("cuda")
model = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=5).to(device)
#model = AutoModelForSequenceClassification.from_pretrained('/collab/yhu5/PICO_sentence_classification/BERT-based-text-classification/results/roberta-base/checkpoint-10500/').to(device)

from datasets import load_metric
from sklearn.metrics import f1_score
import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {'f1':f1_score(labels, predictions, average='weighted')}
    
training_args = TrainingArguments(
    output_dir="./results/"+bert_name,
    learning_rate=1e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=5000,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_steps = 5000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['valid'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics = compute_metrics 
)

trainer.train()

trainer.save_model()
trainer.evaluate()
result = trainer.predict(tokenized_dataset['test'])
print (result.metrics)