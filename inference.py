from transformers import AutoModelForTokenClassification
import numpy as np
from transformers import DataCollatorForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("evs/xlm-roberta-hipe2020-fr-de")

def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list

from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score #, classification_report

def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions, 
                                       eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred), "precision": precision_score(y_true, y_pred), "recall": recall_score(y_true, y_pred)} 



data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)

test_args = TrainingArguments(
    output_dir = "./results",
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 8,   
    dataloader_drop_last = False    
)

# init trainer
trainer = Trainer(
              model = model, 
              data_collator=data_collator,
              args = test_args, 
              compute_metrics = compute_metrics)


test_results = trainer.predict(clara_encoded["test"])
test_results.metrics