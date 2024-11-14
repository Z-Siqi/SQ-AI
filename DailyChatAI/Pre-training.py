from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
import torch

# Load datasets
print("DEBUG: Datasets init loading")
dataset_dailydialog = load_dataset('roskoN/dailydialog')
dataset_alexandreteles_mental_health = load_dataset("alexandreteles/mental-health-conversational-data")
dataset_mpingale_mental_health = load_dataset("mpingale/mental-health-chat-dataset")
dataset_casual_conversation = load_dataset("SohamGhadge/casual-conversation")
dataset_allenai_prosocial_dialog = load_dataset("allenai/prosocial-dialog")
# Split
split_dataset_mpingale_mental_health = dataset_mpingale_mental_health['train'].train_test_split(test_size=0.5)
split_dataset_casual_conversation = dataset_casual_conversation['train'].train_test_split(test_size=0.1)
init_split_allenai_prosocial_dialog = dataset_allenai_prosocial_dialog['train'].train_test_split(test_size=0.9)
init_validation_split_allenai_prosocial_dialog = dataset_allenai_prosocial_dialog['validation'].train_test_split(test_size=0.9)
print("DEBUG: Next")
# Merge training dataset
print("DEBUG: combined_dataset")
combined_dataset = concatenate_datasets([
    dataset_alexandreteles_mental_health['train'], 
    split_dataset_casual_conversation['train'],
    split_dataset_mpingale_mental_health['train'],
    init_split_allenai_prosocial_dialog['train'],
    dataset_dailydialog['train']
])
print("DEBUG: Next")

print("DEBUG: combined_eval_dataset")
combined_eval_dataset = concatenate_datasets([
    dataset_alexandreteles_mental_health['train'], 
    split_dataset_casual_conversation['train'], 
    split_dataset_mpingale_mental_health['test'],
    init_validation_split_allenai_prosocial_dialog['train'],
    dataset_dailydialog['validation']
])
print("DEBUG: Next")

# Loading pre-trained models and tokenizers
print("DEBUG: Loading pre-training model and tokenizer")
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
print("DEBUG: Next")

# Adding the custom padding token
print("DEBUG: Adding the custom padding token")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
print("DEBUG: Next")

# Segment the data set and generate labels
print("DEBUG: Segment the data set and generate labels")
def tokenize_function(examples):
    # Concatenate the contents of all fields into a string
    utterances = []
    num_examples = len(examples[next(iter(examples))])
    for i in range(num_examples):
        Context = " ".join(examples['Context'][i]) if isinstance(examples['Context'][i], list) else examples['Context'][i]
        Response = examples['Response'][i]
        questionText = " ".join(examples['questionText'][i]) if isinstance(examples['questionText'][i], list) else examples['questionText'][i]
        answerText = examples['answerText'][i]
        question = " ".join(examples['question'][i]) if isinstance(examples['question'][i], list) else examples['question'][i]
        answer = examples['answer'][i]
        context = " ".join(examples['context'][i]) if isinstance(examples['context'][i], list) else examples['context'][i]
        response = examples['response'][i]
        _utterances_ = " ".join(examples['utterances'][i]) if isinstance(examples['utterances'][i], list) else examples['utterances'][i]
        
        # Concatenate into an input string
        combined_utterance = f"{Context} {Response} {question} {answer} {questionText} {answerText} {context} {response} {_utterances_}"
        utterances.append(combined_utterance)

    # Perform word segmentation on the concatenated string
    tokenized_output = tokenizer(utterances, truncation=True, padding="max_length", max_length=128)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output
print("DEBUG: Next")

# Segment the training set
print("DEBUG: combined_dataset")
tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)
print("DEBUG: Next")

# Segment the validation set
print("DEBUG: Segment the validation set")
eval_dataset = combined_eval_dataset.map(tokenize_function, batched=True)
print("DEBUG: Next")

print("-------------------- Mapped Columns (eval) --------------------")
print(eval_dataset)
print("---------------------------------------------------------------")

# Set data format
print("DEBUG: Set data format")
tokenized_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
print("DEBUG: Next")

# Setting training arguments
print("DEBUG: Setting training arguments")
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    eval_strategy="steps",
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    label_smoothing_factor=0.1,
    save_steps=10_000,
    logging_dir="./logs",
    prediction_loss_only=True,
    remove_unused_columns=False,
    use_cpu=False,
    tpu_num_cores=1,
    no_cuda=False
)
print("DEBUG: Next")

# Save tokenizer
tokenizer.save_pretrained("./results/tokenizer")

# Customize loss function and add repeated penalty
print("DEBUG: Customize loss function and add repeated penalty")
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get model output
        outputs = model(**inputs)
        logits = outputs.logits

        # Get target labels
        labels = inputs.get("labels")

        # Calculate the standard cross entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()  # 调用 CrossEntropyLoss 类
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Add duplicate n-gram penalty to avoid generating duplicate content
        repeat_penalty = self.compute_repeat_penalty(logits, labels)
        total_loss = loss + 0.1 * repeat_penalty  # 0.1 is the coefficient for adjusting the penalty weight

        return (total_loss, outputs) if return_outputs else total_loss

    def compute_repeat_penalty(self, logits, labels):
        # Simply calculate the n-gram repetition rate as a penalty term (3-gram)
        ngram_size = 3
        penalty = 0.0
        for i in range(ngram_size, logits.size(1)):
            if torch.equal(labels[:, i - ngram_size:i], labels[:, i-ngram_size+1:i+1]):
                penalty += 1.0
        return penalty
print("DEBUG: Next")

# Init Trainer
print("DEBUG: Init Trainer")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset
)
print("DEBUG: Next")

# Start Training
print("DEBUG: Training...")
trainer.train()
