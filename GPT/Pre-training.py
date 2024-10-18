from datasets import load_dataset, concatenate_datasets
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Loading the dataset
dataset_dailydialog = load_dataset('roskoN/dailydialog')

# Merge training set
combined_dataset = concatenate_datasets([dataset_dailydialog['train']])

# Loading pre-trained models and tokenizers
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Adding a custom padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Segment the data set and generate labels
def tokenize_function(examples):
    tokenized_output = tokenizer([" ".join(utterance) for utterance in examples['utterances']],
                                 truncation=True, padding="max_length", max_length=128)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

# Segment the training set
tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

# Segment the validation set
eval_dataset = dataset_dailydialog['validation'].map(tokenize_function, batched=True)

# Remove irrelevant columns
eval_dataset = eval_dataset.remove_columns(['id', 'acts', 'emotions', 'utterances'])

# Setting data format
tokenized_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

# Setting training parameters
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    eval_strategy="epoch",  # Evaluation is performed at each epoch
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    logging_dir="./logs",
    prediction_loss_only=True,
    remove_unused_columns=False
)

# Init Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset  # 传递 eval_dataset
)

# Training
trainer.train()
