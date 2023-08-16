import torch
from transformers import BertForSequenceClassification

import os
import csv
import re
import time

import numpy as np
import torch.nn as nn
import pandas as pd

from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from transformers import AutoModel

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

from tqdm import tqdm
from scipy.special import softmax
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score
from transformers import BertLMHeadModel, AutoTokenizer, BertForSequenceClassification
from transformers.adapters import LoRAConfig, AutoAdapterModel, BertAdapterModel, RobertaAdapterModel

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-inputdata', type=str)
parser.add_argument('-run', type=str)

args = parser.parse_args()
num_label = 3
inputdata = args.inputdata
run = args.run
task = re.sub("\.tsv","",inputdata)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

model = AutoAdapterModel.from_pretrained("lkonle/fiction-gbert-large", num_labels=num_label)
tokenizer = AutoTokenizer.from_pretrained("lkonle/fiction-gbert-large", padding=True, 
                                          add_special_tokens=True,
                                          truncation="only_first", max_len=512, padding_side="left")

config = LoRAConfig(r=4, alpha=16)
model.add_adapter(task, config=config)
model.set_active_adapters(task)

model.add_classification_head(task, num_labels=num_label)

def create_input(ind, dataset):
    
    extext = ""
    while ind > ind-20 and ind > 0:
        try:
            
            ind-=1
            extext += [dataset.loc[ind,"Text"]+" [SEP]"]


            if dataset.loc[ind,"Scene"] != ids:
                break
                
        except:
            ind-=1
            
    return " ".join(extext[::-1])


def preprocess(inputdata, bsize, mode):
    
    sframe = pd.DataFrame()
    sframe["t"] = list(inputdata["Text"])
    sframe["l"] = list(inputdata["Label"])
    
    
    if mode == "train":
        
        sframe = sframe.sample(frac=1)
    
        l1frame = sframe[sframe["l"] == 1]
        l2frame = sframe[sframe["l"] == 0].sample(int(len(l1frame)))
        l3frame = sframe[sframe["l"] == 2].sample(int(len(l1frame)), replace=True)
        
        sframe = pd.concat([l1frame, l2frame, l3frame], axis=0).sample(frac=1)

    sequences = list(sframe["t"])
    labels = list(sframe["l"])
        
    #sequences = [create_input(x, inputdata) for x in sequences]

    tokenized = [tokenizer.encode_plus(x, max_length=512, truncation="only_first", 
                                       padding="max_length") for x in sequences]
    
    
    i = 0
    
    batches = []
    x_t = []
    y_t = []
    x_ids = []
    x_a = []
    
    while i < len(tokenized):
        
        x_t.append(torch.tensor(tokenized[i]['input_ids']))
        x_ids.append(torch.tensor(tokenized[i]['token_type_ids']))
        y_t.append(torch.tensor(labels[i]))
        x_a.append(torch.tensor(tokenized[i]['attention_mask']))
        
        if len(x_t) == bsize:
            
            batches.append([x_t, x_ids, y_t, x_a])
            x_t = []
            y_t = []
            x_ids = []
            x_a = []
        i+=1
        
    return batches

data = pd.read_csv(inputdata, sep="\t", index_col=0)

c2 = list(data[data.Label == 1].sample(frac=0.2).index)
if num_label == 3:
    
    c3 = list(data[data.Label == 2].sample(frac=0.2).index)
    
else:
    
    c3 = []
    
    
c1 = list(data[data.Label == 0].sample(n=len(c2)).index)
indexes = c1+c2+c3

test_data = data.loc[indexes,:]
t = [x for x in list(data.index) if x not in indexes]
train_data = data.loc[t,:]

bsize = 10
accum_aim = 2

train_batches = preprocess(train_data, bsize, "train")
test_batches = preprocess(test_data, bsize, "test")

epochs = 20
lr = 1e-4
optimizer = AdamW(model.parameters(), lr = lr)
total_steps = (len(train_batches) * epochs)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = int(total_steps/100*10),
                                            num_training_steps = total_steps)

device="cuda"
model.cuda()
model = nn.DataParallel(model)

def binas(x):
    
    if x >= 0.5:
        return 1
    else:
        return 0
    
mylog = ""
report = []
total_loss = 0
i = 0
accum_c = 0
logs = []

f1 = 0
rep_t = []
rep_p = []
accum_c = 0

for epoch_i in list(range(0, epochs)):
    
    # train lyrik
    model.module.train_adapter(task)
    train_batches = preprocess(train_data, bsize, "train")
    
    for step, batch in tqdm(enumerate(train_batches), total=len(train_batches)):
        i+=1
        b_input_ids = torch.stack(batch[0]).to(device)
        b_ids = torch.stack(batch[1]).to(device)
        b_labels = torch.stack(batch[2]).to(device)
        b_at = torch.stack(batch[3]).to(device)
        
        
        outputs = model(b_input_ids, token_type_ids=b_ids, labels=b_labels, attention_mask=b_at)
        
        loss = outputs[0].sum()/accum_aim
        loss.backward()
        
        accum_c+=1
        if accum_c == accum_aim:
            
            clip_grad_norm_(model.parameters(), 0.9)
            print(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            accum_c = 0
        
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        trues = label_ids.flatten()
        pred = np.argmax(logits, axis=1)

        rep_t.append(trues)
        rep_p.append(pred)

    pframe = pd.DataFrame()
    pframe["label"] = np.array(rep_t).flatten()
    pframe["prediction"] = np.array(rep_p).flatten()
    mylog+="\nTRAIN\n"+classification_report(pframe["label"], pframe["prediction"])
    print("\nTRAIN\n"+classification_report(pframe["label"], pframe["prediction"]))
    
    with open("logs/"+task+"_"+run+".tsv", "w") as f:
        f.write(mylog)
            
    rep_t = []
    rep_p = []
    
    # test lyrik
    
    if (epoch_i+1) % 5 == 0:
        
    
        model.eval()
        
        with torch.no_grad():
            
            scores = []
            labels = []
            for batch in tqdm(test_batches):

                b_input_ids = torch.stack(batch[0]).to(device)
                b_ids = torch.stack(batch[1]).to(device)
                b_at = torch.stack(batch[3]).to(device)
                b_labels = torch.stack(batch[2]).to(device)

                outputs = model(b_input_ids, token_type_ids=b_ids, attention_mask=b_at)

                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                trues = label_ids.flatten()
                pred = np.argmax(logits, axis=1)

                rep_t.append(trues)
                rep_p.append(pred)

            
        pframe = pd.DataFrame()
        pframe["label"] = np.array(rep_t).flatten()
        pframe["prediction"] = np.array(rep_p).flatten()
        mylog+="\nTEST\n"+classification_report(pframe["label"], pframe["prediction"])
        print("\nTEST\n"+classification_report(pframe["label"], pframe["prediction"]))

prediciton = []
for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    

    inputs = tokenizer.encode_plus(row["Text"], max_length=512, truncation="only_first", padding="max_length")
    
    b_input_ids = torch.tensor((inputs["input_ids"], inputs["input_ids"])).to(device)
    b_ids = torch.tensor((inputs["token_type_ids"], inputs["token_type_ids"])).to(device)
    b_at = torch.tensor((inputs["attention_mask"], inputs["attention_mask"])).to(device)
                      
    outputs = model(b_input_ids, token_type_ids=b_ids, attention_mask=b_at)

    
    logits = outputs[0][0]
    logits = logits.detach().cpu().numpy()
    pred = np.argmax(logits)
    prediciton.append(pred)
    
    
mylog+="\nFINAL TEST\n"+classification_report(test_data["Label"], prediciton)

with open("logs/"+task+"_"+run+".tsv", "w") as f:
    f.write(mylog)
          
model.module.save_adapter("models/"+task, task, with_head=True)
