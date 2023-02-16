from transformers import AutoTokenizer
import pandas as pd
from transformers import AutoConfig
import torch
from transformers import AutoModelForTokenClassification
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score #, classification_report

from load_data import Data


class Tokenizer(Data):
    def __init__(self, dir='10/', model_name = "xlm-roberta-base"):
        super().__init__(dir)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_tensors="pt")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def ejemplo_tokenizado(self, text):
        xlmr_tokens = self.tokenizer(text).tokens()
        return pd.DataFrame([xlmr_tokens], index=["XLM-R"])
    
    def ver_input_ids(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        xlmr_tokens = self.tokenizer(text).tokens()
        return pd.DataFrame([xlmr_tokens, input_ids[0].numpy()], index=["Tokens", "Input IDs"])

    '''
    import torch.nn as nn
    from transformers import XLMRobertaConfig
    from transformers.modeling_outputs import TokenClassifierOutput
    from transformers.models.roberta.modeling_roberta import RobertaModel
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
    '''

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], max_length=100, padding='max_length', is_split_into_words=True) # padding='max_lenght', truncation=True, max_length=10, se atasca si meto esto
                                                                                                                            # con return_tensors="pt" no funciona y parece que con truncation tampoco
        labels = []
        for idx, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def encode_dataset(self, corpus):
        return corpus.map(self.tokenize_and_align_labels, batched=True, remove_columns=['ner_tags', 'tokens']) # 


    def align_predictions(self, predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []

        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                # Ignore label IDs = -100
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(self.index2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(self.index2tag[preds[batch_idx][seq_idx]])

            labels_list.append(example_labels)
            preds_list.append(example_preds)

        return preds_list, labels_list


    def compute_metrics(self, eval_pred):
        y_pred, y_true = self.align_predictions(eval_pred.predictions, eval_pred.label_ids)
        return {"f1": f1_score(y_true, y_pred), "precision": precision_score(y_true, y_pred), "recall": recall_score(y_true, y_pred)}


    def encode(self):

        self.config = AutoConfig.from_pretrained(self.model_name, 
                                                num_labels = self.tags.num_classes,
                                                id2label = self.index2tag,
                                                label2id = self.tag2index)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (AutoModelForTokenClassification #XLMRobertaForTokenClassification
                    .from_pretrained(self.model_name, config = self.config)
                    .to(self.device))
        
        self.clara_encoded = self.encode_dataset(self.ner_dataset)