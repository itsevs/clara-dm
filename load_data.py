import pickle
from pathlib import Path
import re
import torch
import pandas as pd
import csv
#from sklearn.model_selection import train_test_split
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
import os
#print(torch.cuda.is_available())

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#os.system("nvidia-smi")

#from google.colab import drive
#drive.mount('/content/drive')
#dir = '/content/drive/MyDrive/Colab-Notebooks/TFM-HIPE-DHQ/10/'

class Data():
    def __init__(self, dir = '10/'): # no se pueden poner backslashes \
        self.dir = dir
        self.docs = [doc for doc in os.listdir(dir) if doc.endswith('iob')]

    def asignacion(self):
        val = self.docs[8]
        test = []
        test.append(self.docs[0])
        test.append(self.docs[3])
        test.append(self.docs[10])

        for i in [8, 3, 0]: # tienen que estar en orden inverso
            self.docs.pop(i)
        train = self.docs

        self.train, self.test, self.val = train, test, val

    def encadenar_csvs(self, split): # encadenar los csv del train/test en uno solo
        columns = ['Token', 'Tag']
        # nota: solo se puede ejecutar una vez para que no se sobreescriba el dataframe final "df_fin"

        if isinstance(split, list):
            return pd.concat([pd.read_csv(self.dir + str(i), sep='\t', names=columns, encoding='latin1') for i in split])

        else:
            return pd.read_csv(self.dir + str(split), sep='\t', names=columns, encoding='latin1')
    

    def attempt_decode(self, x):
        try:
            return x.encode('latin-1').decode('utf-8')
        except UnicodeDecodeError:
            return x
        # f'Unable to decode: %s' %

    def parse_clara(self, dataframe):
        
        all_tokens = []
        all_tags = []

        for row in dataframe.itertuples():
            all_tokens.append(row.Token)
            all_tags.append(row.Tag)

        dataset_tokens_str = ' '.join(all_tokens)
        sentence_tokens = [str(i).split() for i in self.nlp(dataset_tokens_str).sents]

        sentence_tags = []
        counter = 0
        for sentence in sentence_tokens:
            length = len(sentence)
            sentence_tags.append(all_tags[counter:counter+length])
            counter = counter+length

        return sentence_tokens, sentence_tags

    

#print(f'numero de frases en el train: {len(train_sents)}')
#print(f'numero de frases en el val: {len(val_sents)}')
#print(f'numero de frases en el test: {len(test_sents)}')

    def estadisticas(self, split):
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

    def print_estadisticas(self):
        print("la frase mas larga de train mide " + str(self.estadisticas(self.train_sents)[0]))
        print("la frase mas larga de test mide " + str(self.estadisticas(self.test_sents)[0]))
        print("la suma de tokens en train es " + str(self.estadisticas(self.train_sents)[1]))
        print("la suma de tokens en test es " + str(self.estadisticas(self.test_sents)[1]))

    def create_tag_names(self, batch):
        return {"ner_tags_str": [self.tags.int2str(idx) for idx in batch["ner_tags"]]}

    def split2freqs(self, ner_dataset):
        split2freqs = defaultdict(Counter)
        for split, dataset in ner_dataset.items():
            for row in dataset["ner_tags_str"]:
                for tag in row:
                    if tag.startswith("B"):
                        tag_type = tag.split("-")[1]
                        split2freqs[split][tag_type] += 1
        return pd.DataFrame.from_dict(split2freqs, orient="index")




    def processing(self):

            self.asignacion()

            self.train = self.encadenar_csvs(self.train)
            self.test = self.encadenar_csvs(self.test)
            self.val = self.encadenar_csvs(self.val)

            self.train['Token'] = self.train['Token'].apply(self.attempt_decode)
            self.val['Token'] = self.val['Token'].apply(self.attempt_decode)
            self.test['Token'] = self.test['Token'].apply(self.attempt_decode)

            self.nlp = spacy.load('es_core_news_sm')

            self.train_sents, self.train_tags = self.parse_clara(self.train)
            self.val_sents, self.val_tags = self.parse_clara(self.val)
            self.test_sents, self.test_tags = self.parse_clara(self.test)


            # Importante: hay que guardar estos diccionarios junto con el modelo, ya que si se carga el modelo y se da otro diccionario de conversión se va a la porra todo
            self.flat_tags1 = [tag for sublist in self.train_tags for tag in sublist]
            self.flat_tags2 = [tag for sublist in self.test_tags for tag in sublist]
            self.flat_tags3 = [tag for sublist in self.val_tags for tag in sublist]

            self.unique_tags = sorted(list(set(self.flat_tags1+self.flat_tags2+self.flat_tags3)))
            #tag2id = {tag:id for id, tag in enumerate(unique_tags)}
            #id2tag = {id:tag for tag, id in tag2id.items()}
            #class_names = sorted(list(id2tag.values()))
            self.class_names = self.unique_tags
            #class_names.append('I-prod_ador')
            #print(class_names, len(class_names))


            self.train_dataset_dict = {"tokens": self.train_sents, "ner_tags": self.train_tags} 
            self.val_dataset_dict = {"tokens": self.val_sents, "ner_tags": self.val_tags} 
            self.test_dataset_dict = {"tokens": self.test_sents, "ner_tags": self.test_tags} 

            self.features = Features({'tokens': Sequence(Value("string")), 'ner_tags': Sequence(ClassLabel(names=self.class_names))})

            self.train_dataset = Dataset.from_dict(self.train_dataset_dict, features=self.features)
            self.val_dataset = Dataset.from_dict(self.val_dataset_dict, features=self.features)
            self.test_dataset = Dataset.from_dict(self.test_dataset_dict, features=self.features)

            self.ner_dataset = DatasetDict()
            self.ner_dataset["train"] = self.train_dataset
            self.ner_dataset["val"] = self.val_dataset
            self.ner_dataset["test"] = self.test_dataset

            self.tags = self.ner_dataset["test"].features["ner_tags"].feature

    

            self.ner_dataset = self.ner_dataset.map(self.create_tag_names)

            self.split2freqs(self.ner_dataset) # para ver distribucion de etiquetas

            self.index2tag = {idx: tag for idx, tag in enumerate(self.tags.names)}
            self.tag2index = {tag: idx for idx, tag in enumerate(self.tags.names)}

'''
def ver_ejemplo_dataset(split, elemento):
    ejemplo = ner_dataset[split][elemento]
    return pd.DataFrame([ejemplo['tokens'], ejemplo['ner_tags_str']], ['Tokens', 'Tags'])

ver_ejemplo_dataset("test", 28)
'''