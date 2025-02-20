import torch
# when trained using BERT pre-tuned for NER
# from transformers import (Trainer, TrainingArguments,
#                           AutoTokenizer, AutoModelForTokenClassification)
from transformers import (BertForTokenClassification, BertTokenizerFast,
                          Trainer, TrainingArguments)


from utils import (JSONDataset, compute_loss_with_class_weights,
                   compute_class_weights, get_data_collator,
                   compute_metrics)

labels = ["O", "B-ANIMAL", "I-ANIMAL"]
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

# when trained using BERT pre-tuned for NER
# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

model_name = "bert-base-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# File paths
train_file = "../dataset/processed/training_dataset.json"
val_file = "../dataset/processed/val_dataset.json"


# Dataset Class (defined earlier)
train_dataset = JSONDataset(train_file, tokenizer, label2id, max_len=256)
val_dataset = JSONDataset(val_file, tokenizer, label2id, max_len=256)

class_weights = compute_class_weights(train_dataset)

# when trained using BERT pre-tuned for NER
# model = AutoModelForTokenClassification.from_pretrained(
#     "dslim/bert-base-NER",

model = BertForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,   
    label2id=label2id,   
    ignore_mismatched_sizes=True)

# Set training arguments for multi-GPU
training_args = TrainingArguments(
    output_dir="./bert_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    gradient_accumulation_steps=1,
    fp16=False,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
)


data_collator = get_data_collator(tokenizer)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True)
        logits = outputs.get("logits")
        loss = compute_loss_with_class_weights(logits, labels,
                                               class_weights)
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  
    compute_metrics=compute_metrics
)

trainer.train()
metrics = trainer.evaluate()

model_save_path = "./bert_model"
tokenizer_save_path = "./bert_tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

model.push_to_hub("sskyisthelimit/animal-ner-model")
tokenizer.push_to_hub("sskyisthelimit/animal-ner-model")
