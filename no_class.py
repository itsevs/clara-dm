import pickle
from pathlib import Path
import re
import torch
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
from transformers import RobertaForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch import nn
import numpy as np
from collections import Counter
from collections import defaultdict
import nltk
nltk.download('punkt')
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence
import spacy
import es_core_news_sm

print(torch.cuda.is_available())

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.system("nvidia-smi")

#from google.colab import drive
#drive.mount('/content/drive')
#dir = '/content/drive/MyDrive/Colab-Notebooks/TFM-HIPE-DHQ/10/'

dir = 'C:/Users/evasa/Downloads/10/' # no se pueden poner backslashes \
os.listdir(dir)
docs = [doc for doc in os.listdir(dir) if doc.endswith('iob')]

def asignacion(docs):
    val = docs[8]
    test = []
    test.append(docs[0])
    test.append(docs[3])
    test.append(docs[10])

    for i in [8, 3, 0]: # tienen que estar en orden inverso
        docs.pop(i)
    train = docs

    return train, test, val

train, test, val = asignacion(docs)

def encadenar_csvs(split): # encadenar los csv del train/test en uno solo
    columns = ['Token', 'Tag']
    # nota: solo se puede ejecutar una vez para que no se sobreescriba el dataframe final "df_fin"

    if isinstance(split, list):
        return pd.concat([pd.read_csv(dir + str(i), sep='\t', names=columns, encoding='latin1') for i in split])

    else:
        return pd.read_csv(dir + str(split), sep='\t', names=columns, encoding='latin1')
    

train = encadenar_csvs(train)
test = encadenar_csvs(test)
val = encadenar_csvs(val)

def attempt_decode(x):
    try:
        return x.encode('latin-1').decode('utf-8')
    except UnicodeDecodeError:
       return x
       # f'Unable to decode: %s' %

train['Token'] = train['Token'].apply(attempt_decode)
val['Token'] = val['Token'].apply(attempt_decode)
test['Token'] = test['Token'].apply(attempt_decode)

nlp = spacy.load('es_core_news_sm')

def parse_clara(dataframe):
    
    all_tokens = []
    all_tags = []

    for row in dataframe.itertuples():
      all_tokens.append(row.Token)
      all_tags.append(row.Tag)

    dataset_tokens_str = ' '.join(all_tokens)
    sentence_tokens = [str(i).split() for i in nlp(dataset_tokens_str).sents]

    sentence_tags = []
    counter = 0
    for sentence in sentence_tokens:
      length = len(sentence)
      sentence_tags.append(all_tags[counter:counter+length])
      counter = counter+length

    return sentence_tokens, sentence_tags

train_sents, train_tags = parse_clara(train)
val_sents, val_tags = parse_clara(val)
test_sents, test_tags = parse_clara(test)

#print(f'numero de frases en el train: {len(train_sents)}')
#print(f'numero de frases en el val: {len(val_sents)}')
#print(f'numero de frases en el test: {len(test_sents)}')

def estadisticas(split):
    # ver cuánto mide la frase más larga
    max_len = 50
    i = 0
    lista = []
    for sentence in split: # en train hay 6 frases que miden más de 50 caracteres; en val hay 4 y en test hay 5
        if len(sentence) > max_len:
            lista.append( (i, len(sentence)) )
            max_len = len(sentence)
        i += 1
    
    sum_tokens = len([token for sentence in split for token in sentence ])
    #sum_tags = len([token for sentence in train_tags for token in sentence ])

    return max_len, sum_tokens #, lista # lista con las posiciones de las frases que superan la longitud máxima 
                # en el test_texts del francés, la frase 64 mide 1158 tokens porque faltan las etiquetas de EndOfSentence
                # en el train_texts del francés, la frase 272 mide 529, la frase 319 mide 752 y la 3877 mide 2069 # esto era antes de solucionar las frases que empezaban por I-

print("la frase mas larga de train mide " + str(estadisticas(train_sents)[0]))
print("la frase mas larga de test mide " + str(estadisticas(test_sents)[0]))
print("la suma de tokens en train es " + str(estadisticas(train_sents)[1]))
print("la suma de tokens en test es " + str(estadisticas(test_sents)[1]))

# Importante: hay que guardar estos diccionarios junto con el modelo, ya que si se carga el modelo y se da otro diccionario de conversión se va a la porra todo
flat_tags1 = [tag for sublist in train_tags for tag in sublist]
flat_tags2 = [tag for sublist in test_tags for tag in sublist]
flat_tags3 = [tag for sublist in val_tags for tag in sublist]

unique_tags = sorted(list(set(flat_tags1+flat_tags2+flat_tags3)))
#tag2id = {tag:id for id, tag in enumerate(unique_tags)}
#id2tag = {id:tag for tag, id in tag2id.items()}
#class_names = sorted(list(id2tag.values()))
class_names = unique_tags
class_names.append('I-prod_ador')
print(class_names, len(class_names))


train_dataset_dict = {"tokens": train_sents, "ner_tags": train_tags} 
val_dataset_dict = {"tokens": val_sents, "ner_tags": val_tags} 
test_dataset_dict = {"tokens": test_sents, "ner_tags": test_tags} 

features = Features({'tokens': Sequence(Value("string")), 'ner_tags': Sequence(ClassLabel(names=class_names))})

train_dataset = Dataset.from_dict(train_dataset_dict, features=features)
val_dataset = Dataset.from_dict(val_dataset_dict, features=features)
test_dataset = Dataset.from_dict(test_dataset_dict, features=features)

ner_dataset = DatasetDict()
ner_dataset["train"] = train_dataset
ner_dataset["val"] = val_dataset
ner_dataset["test"] = test_dataset

tags = ner_dataset["test"].features["ner_tags"].feature

def create_tag_names(batch):
    return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}

ner_dataset = ner_dataset.map(create_tag_names)

def ver_ejemplo_dataset(split, elemento):
    ejemplo = ner_dataset[split][elemento]
    return pd.DataFrame([ejemplo['tokens'], ejemplo['ner_tags_str']], ['Tokens', 'Tags'])

ver_ejemplo_dataset("test", 28)

def split2freqs(ner_dataset):
    split2freqs = defaultdict(Counter)
    for split, dataset in ner_dataset.items():
        for row in dataset["ner_tags_str"]:
            for tag in row:
                if tag.startswith("B"):
                    tag_type = tag.split("-")[1]
                    split2freqs[split][tag_type] += 1
    return pd.DataFrame.from_dict(split2freqs, orient="index")

split2freqs(ner_dataset)

index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

