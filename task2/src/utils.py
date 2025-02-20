import torch
from transformers import (DataCollatorForTokenClassification)
from torch.utils.data import Dataset
from sklearn.metrics import (precision_score,
                             recall_score, f1_score, accuracy_score)
import numpy as np
import json

from inference import label2id


class JSONDataset(Dataset):
    def __init__(self, json_file, tokenizer, label2id, max_len=256):
        self.data = self.load_data(json_file)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def load_data(self, json_file):
        """Load the JSON file."""
        with open(json_file, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized_inputs = self.tokenizer(
            item['tokens'],
            is_split_into_words=True,
            padding='max_length',  # padding to the max sequence length
            truncation=True,       # truncate if exceed max length
            max_length=self.max_len,  # maximum length for input
        )
        labels = []
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # special tokens
            elif word_idx != previous_word_idx:
                labels.append(int(item["labels"][word_idx]))
            else:
                labels.append(-100)
            previous_word_idx = word_idx
    
        # Adjust label padding to match max_len
        labels = labels[:self.max_len]
        labels += [-100] * (self.max_len - len(labels))
        tokenized_inputs['labels'] = torch.tensor(labels, dtype=torch.long)
    
        # Convert input_ids and attention_mask to tensors
        tokenized_inputs['input_ids'] = torch.tensor(
            tokenized_inputs['input_ids'],dtype=torch.long)
        tokenized_inputs['attention_mask'] = torch.tensor(
            tokenized_inputs['attention_mask'], dtype=torch.long)
    
        return tokenized_inputs
    

def compute_metrics(pred):
    """
    Compute evaluation metrics: precision, recall, F1, and accuracy.
    Args:
        pred: A PredictionOutput object containing predictions and labels.
    Returns:
        A dictionary with evaluation metrics.
    """
    # Flatten predictions and labels
    predictions = np.argmax(pred.predictions, axis=2)  # Get class with max logit
    labels = pred.label_ids

    # Flatten and filter out special tokens (label IDs == -100)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Convert lists of lists into flat lists
    flat_predictions = [p for preds in true_predictions for p in preds]
    flat_labels = [l for labels in true_labels for l in labels]

    # Compute metrics
    precision = precision_score(flat_labels, flat_predictions,
                                average="weighted", zero_division=0)
    
    recall = recall_score(flat_labels, flat_predictions,
                          average="weighted", zero_division=0)
    
    f1 = f1_score(flat_labels, flat_predictions,
                  average="weighted", zero_division=0)
    
    accuracy = accuracy_score(flat_labels, flat_predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_class_weights(dataset):
    """
    Compute class weights based on the class frequencies in the dataset.
    This will calculate weights for each class: ["O", "B-ANIMAL", "I-ANIMAL"].
    """
    label_counts = [0, 0, 0,]  # Assuming labels: ["O", "B-ANIMAL", "I-ANIMAL"]

    for example in dataset:  # Use the training set to calculate class frequencies
        labels = example['labels']
        for label in labels:
            if label != -100:  # Skip padded tokens
                label_counts[label] += 1

    total_labels = sum(label_counts)
    class_weights = [total_labels / count for count in label_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights


def get_data_collator(tokenizer):
    """
    Returns a data collator for token classification.
    
    Args:
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for token classification.

    Returns:
    transformers.DataCollatorForTokenClassification: The data collator.
    """
    return DataCollatorForTokenClassification(tokenizer=tokenizer)


def compute_loss_with_class_weights(outputs, labels, class_weights):
    """
    Custom loss function that applies class weights during training.
    """
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(outputs.device), ignore_index=-100)
    
    return loss_fct(outputs.view(-1, 5), labels.view(-1))


def tokenize_and_label_regular(sentence, tokens, labels, tokenizer):
    tokenized_sentence = tokenizer(sentence.strip().split(),
                                   return_tensors="pt",
                                   is_split_into_words=True)
    tokenized_sentence = tokenizer.convert_ids_to_tokens(
        tokenized_sentence["input_ids"].squeeze().tolist())

    tokenized_sentence = [tok for tok in tokenized_sentence if tok not in ['[CLS]', '[SEP]']]

    tokens.extend(tokenized_sentence)
    labels.extend([int(label2id["O"])] * len(tokenized_sentence))


def tokenize_and_label_elevation(string, tokens, labels,
                                 annot_label, tokenizer):
    
    tokenized_entity = tokenizer(string.strip().split(),
                                 return_tensors="pt", is_split_into_words=True)
    tokenized_entity = tokenizer.convert_ids_to_tokens(
       tokenized_entity["input_ids"].squeeze().tolist())
    for i, token in enumerate(tokenized_entity):
        if token not in ['[CLS]', '[SEP]']:
            if token in {"'s", ",", "."}:
                tokens.append(token)
                labels.append(int(label2id["O"]))
            else:
                tokens.append(token)
                labels.append(
                    int(label2id[annot_label]) if i == 0 else int(label2id[annot_label]))


def convert_to_bert_format(sentence, entities, tokenizer):
    start = 0
    tokens = []
    labels = []
    for entity in entities:
        entity_start = entity[0]
        entity_end = entity[1]
        entity_label = entity[2]

        tokenize_and_label_regular(sentence[start:entity_start],
                                   tokens, labels, tokenizer)
        
        tokenize_and_label_elevation(sentence[entity_start:entity_end], tokens,
                                     labels, entity_label, tokenizer)
        start = entity_end

    tokenize_and_label_regular(sentence[start:], tokens, labels, tokenizer)
    return (tokens, labels, sentence)

