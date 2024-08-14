import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, LlamaConfig
from datasets import load_dataset
import evaluate
from copy import deepcopy

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Starting script execution...")

def load_and_prepare_dataset():
    print("Starting to load dataset...")
    dataset = load_dataset("Gokul-waterlabs/ICD-10-CM", split="train")
    print(f"Dataset loaded. Number of examples: {len(dataset)}")
    dataset = dataset.select(range(1000))
    print(f"Dataset trimmed to 1,000 examples.")
    return dataset

dataset = load_and_prepare_dataset()

print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-13b")
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer initialized.")

def tokenize_function(examples):
    inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=64)
    outputs = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = outputs["input_ids"]
    return inputs

print("Starting tokenization of dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
print("Dataset tokenization completed.")

print("Splitting dataset into train and test sets...")
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
print(f"Split complete. Train size: {len(train_dataset)}, Test size: {len(eval_dataset)}")

print("Loading teacher model...")
teacher_model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-13b", torch_dtype=torch.float16)
teacher_model.config.pad_token_id = tokenizer.pad_token_id
teacher_model = teacher_model.to(device)
teacher_config = teacher_model.config
print("Teacher model loaded and moved to GPU.")

# Define smaller student model configuration
student_config = LlamaConfig(
    vocab_size=teacher_config.vocab_size,
    hidden_size=256,  # Significantly reduced
    intermediate_size=512,  # Significantly reduced
    num_hidden_layers=2,  # Reduced to 2 layers
    num_attention_heads=8,  # Reduced number of attention heads
    max_position_embeddings=teacher_config.max_position_embeddings,
    rms_norm_eps=teacher_config.rms_norm_eps,
    pad_token_id=tokenizer.pad_token_id,
)

print(f"Creating student model with {student_config.num_hidden_layers} layers")

# Create smaller student model
student_model = AutoModelForCausalLM.from_config(student_config)
student_model = student_model.to(device)

print(f"Student model created with {sum(p.numel() for p in student_model.parameters())} parameters")

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs_student = model(**inputs)
        loss_ce = outputs_student.loss
        logits_student = outputs_student.logits

        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits

        if logits_student.size() != logits_teacher.size():
            logits_teacher = logits_teacher[:, :logits_student.size(1), :]

        loss_kd = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(logits_student / 2.0, dim=-1),
            F.softmax(logits_teacher / 2.0, dim=-1)
        ) * (2.0 ** 2)

        loss = 0.5 * loss_ce + 0.5 * loss_kd
        return (loss, outputs_student) if return_outputs else loss

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # Mask out padding tokens
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]
    
    # Ensure predictions and labels are 1D arrays
    predictions = predictions.ravel()
    labels = labels.ravel()
    
    return accuracy_metric.compute(predictions=predictions, references=labels)

print("Metric preparation complete.")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=50,
    num_train_epochs=1,
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=4,
)

print("Training arguments initialized successfully.")

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    teacher_model=teacher_model
)

print("Trainer setup complete.")

print("Starting training...")
trainer.train()
print("Training completed.")

print("Saving model...")
trainer.save_model("./small_medalpaca_model")
print("Model saved.")

student_params = sum(p.numel() for p in student_model.parameters())
teacher_params = sum(p.numel() for p in teacher_model.parameters())

print(f"Teacher Model Parameters: {teacher_params}")
print(f"Student Model Parameters: {student_params}")
print(f"Parameter Reduction: {(teacher_params - student_params) / teacher_params * 100:.2f}%")

print("Script execution completed.")