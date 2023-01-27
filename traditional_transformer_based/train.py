bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

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

def read_whole(file):
    file_dict = {'text':[],'label':[]}
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('###') and line != '\n' and line.split('\t')[0]!='Title':
                file_dict['text'].append(line.split('\t')[1])
                file_dict['label'].append(classes.index(line.split('\t')[0]))

    return file_dict
    
train_dict = read_whole(train)
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