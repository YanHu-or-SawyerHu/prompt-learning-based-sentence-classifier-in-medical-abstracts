import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
percentage_of_training_data = 0.05
import json, requests
from bs4 import BeautifulSoup
import glob,random
from shutil import copyfile
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
classes = ['OBJECTIVE','METHODS','BACKGROUND','CONCLUSIONS','RESULTS']
# read data
data_dir = '../data/pubmed-rct/PubMed_200k_RCT_with_title/'
train = data_dir+'train.txt'
valid = data_dir+'dev.txt'
test = data_dir+'test.txt'
def read_part(file):
    lenth=0
    file_dict = {}
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        print (len(lines))
        for line in lines:
            if line.startswith('###'):
                PMID = line.strip().strip('###').strip(':')
                file_dict.update({PMID:{'Title':[],'OBJECTIVE':[],'BACKGROUND':[],'METHODS':[],'RESULTS':[],'CONCLUSIONS':[],'Sequence':[]}}) 
            elif line !='\n':
                sent = line.strip().split('\t')[1]
                tag = line.split('\t')[0]
                file_dict[PMID][tag].append(sent)
                file_dict[PMID]['Sequence'].append(sent)

    return file_dict

def read_whole(file):
    lenth=0
    file_dict = {}
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('###'):
                PMID = line.strip().strip('###').strip(':')
                file_dict.update({PMID:{'Title':[],'OBJECTIVE':[],'BACKGROUND':[],'METHODS':[],'RESULTS':[],'CONCLUSIONS':[],'Sequence':[]}}) 
            elif line !='\n':
                sent = line.strip().split('\t')[1]
                tag = line.split('\t')[0]
                file_dict[PMID][tag].append(sent)
                file_dict[PMID]['Sequence'].append(sent)

    return file_dict
train_dict = read_part(train)
valid_dict = read_whole(valid)
test_dict = read_whole(test)
print ("total:",len(train_dict))
reduced_train_dict = {}
print (round(percentage_of_training_data*len(train_dict)))
for key,value in train_dict.items():
    
    if len(reduced_train_dict) == round(percentage_of_training_data*len(train_dict)):
        break
    else:
        reduced_train_dict.update({key:value})
print ("reduce:",len(reduced_train_dict))
print (list(reduced_train_dict.keys())[-1])







# mask 1: + sentence 1
def create_sent_to_tag_dict(content_dict):
    inv_map = {}
    for tag,sents in content_dict.items():
        if tag!='Sequence':
            for sent in sents:
                inv_map.update({sent:tag})
    return inv_map

from openprompt.data_utils import InputExample
dataset = {'train':[],'test':[],'validation':[]}

def data_loader(input_dict,set_name):
    for i1,(PMID, content) in enumerate(input_dict.items()):
        sent_to_tag_dict = create_sent_to_tag_dict(content)
        all_sents = content['Sequence']

        for i2,sent in enumerate(all_sents):
            if sent_to_tag_dict[sent]!= 'Title':
                
                if i2 != len(all_sents)-1:
                    dataset[set_name].append(InputExample(text_a=sent,label=classes.index(sent_to_tag_dict[sent])))
                else:
                    dataset[set_name].append(InputExample(text_a=sent,label=classes.index(sent_to_tag_dict[sent])))

data_loader(reduced_train_dict,'train')
data_loader(valid_dict,'validation')
data_loader(test_dict,'test')
template_text = '{"mask"}: {"placeholder":"text_a"}'










from openprompt.prompts import ManualTemplate
from openprompt.prompts import MixedTemplate

#mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)
# To better understand how does the template wrap the example, we visualize one instance.

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0]) 
print(wrapped_example)
# We provide a `PromptDataLoader` class to help you do all the above matters and wrap them into an `torch.DataLoader` style iterator.
from openprompt import PromptDataLoader
from openprompt.prompts import ManualTemplate
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=512, 
    batch_size=8,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=512, batch_size=8,shuffle=False, teacher_forcing=False, predict_eos_token=False,truncate_method="head")
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=512, batch_size=8,shuffle=False, teacher_forcing=False, predict_eos_token=False,truncate_method="head")
# Define the verbalizer
# In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:

from openprompt.prompts import ManualVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer, num_classes=5, 
                        label_words=[['OBJECTIVE'],['METHODS'],['BACKGROUND'],['CONCLUSIONS'],['RESULTS']])

logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm, and 
# Although you can manually combine the plm, template, verbalizer together, we provide a pipeline 
# model which take the batched data from the PromptDataLoader and produce a class-wise logits
from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()
# Now the training is standard
import torch
import pandas as pd
from sklearn.metrics import classification_report
from transformers import  AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)


best_f1 = 0
no_impro = 0

for epoch in range(100):
    tot_loss = 0 
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
            
    # Evaluate
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        
    report = classification_report(alllabels, allpreds, target_names = classes, digits=4)
    f1 = f1_score(alllabels, allpreds, average='weighted')
    f_noooooo = open('./200k_template1_1.txt','a+')
    f_noooooo.write(str(f1)+'\n\n')
    f_noooooo.close()
    print (f1)
    if float(f1)>best_f1:
        best_f1=float(f1)
        no_impro = 0
    else:
        no_impro+=1
    if no_impro==3:
        break
        
    # Evaluate
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        
    report = classification_report(alllabels, allpreds, target_names = classes, digits=4)
    f_noooooo = open('./200k_template1_1.txt','a+')
    f_noooooo.write(str(report)+'\n\n')
    f_noooooo.close()
    print (report)

