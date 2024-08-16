import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, LlamaConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Was asked to gather along dimension 0")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'  # Use all available GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_prepare_dataset(max_samples=None):
    print("Starting to load dataset...")
    dataset = load_dataset("Gokul-waterlabs/ICD-10-CM", split="train")
    print(f"Dataset loaded. Number of examples: {len(dataset)}")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Dataset trimmed to {len(dataset)} examples.")
    return dataset

def tokenize_function(examples):
    inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=256)
    outputs = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=256)
    inputs["labels"] = outputs["input_ids"]
    return inputs

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_student = model(**inputs, use_cache=False)
        loss_ce = outputs_student.loss
        logits_student = outputs_student.logits

        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs, use_cache=False)
            logits_teacher = outputs_teacher.logits

        if logits_student.size() != logits_teacher.size():
            logits_teacher = logits_teacher[:, :logits_student.size(1), :]

        loss_kd = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(logits_student / self.temperature, dim=-1),
            F.softmax(logits_teacher / self.temperature, dim=-1)
        ) * (self.temperature ** 2)

        loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd

        return (loss, outputs_student) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.fp16 and self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def train(self):
        return super().train()

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def generate_text(prompt, model, tokenizer, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        use_cache=True  # Enable caching for generation
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def comprehensive_evaluation(model, dataset):
    model.eval()
    all_predictions = []
    all_labels = []
    
    for batch in dataset:
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy().ravel())
        all_labels.extend(inputs['labels'].cpu().numpy().ravel())
    
    mask = np.array(all_labels) != -100
    filtered_predictions = np.array(all_predictions)[mask]
    filtered_labels = np.array(all_labels)[mask]
    
    accuracy = accuracy_score(filtered_labels, filtered_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(filtered_labels, filtered_predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def plot_confusion_matrix(model, dataset, tokenizer):
    model.eval()
    all_predictions = []
    all_labels = []
    
    for batch in dataset:
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy().ravel())
        all_labels.extend(inputs['labels'].cpu().numpy().ravel())
    
    mask = np.array(all_labels) != -100
    filtered_predictions = np.array(all_predictions)[mask]
    filtered_labels = np.array(all_labels)[mask]
    
    cm = confusion_matrix(filtered_labels, filtered_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_inference_time(model, dataset, num_samples=100):
    model.eval()
    total_time = 0
    num_tokens = 0
    
    for i, batch in enumerate(dataset):
        if i >= num_samples:
            break
        
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        
        total_time += end_time - start_time
        num_tokens += inputs['input_ids'].numel()
    
    avg_time_per_sample = total_time / num_samples
    tokens_per_second = num_tokens / total_time
    
    return {
        "avg_time_per_sample": avg_time_per_sample,
        "tokens_per_second": tokens_per_second
    }

if __name__ == "__main__":
    print("Starting script execution...")

    dataset = load_and_prepare_dataset(max_samples=5000)  # Reduced dataset size for quicker execution

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-13b")
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer initialized.")

    print("Starting tokenization of dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=4)
    print("Dataset tokenization completed.")

    print("Splitting dataset into train and validation sets...")
    splits = tokenized_datasets.train_test_split(test_size=0.1)
    train_dataset = splits['train']
    validation_dataset = splits['test']
    print(f"Split complete. Train size: {len(train_dataset)}, Validation size: {len(validation_dataset)}")

    print("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-13b", torch_dtype=torch.float16, device_map="auto")
    teacher_model.config.pad_token_id = tokenizer.pad_token_id
    teacher_model.gradient_checkpointing_enable()
    teacher_model.config.use_cache = False  # Disable caching during training
    print("Teacher model loaded and moved to GPU.")

    teacher_config = teacher_model.config

    student_config = LlamaConfig(
        vocab_size=teacher_config.vocab_size,
        hidden_size=768,  # Reduced from 1024
        intermediate_size=2048,  # Reduced from 2816
        num_hidden_layers=6,  # Reduced from 8
        num_attention_heads=12,  # Reduced from 16
        max_position_embeddings=teacher_config.max_position_embeddings,
        rms_norm_eps=teacher_config.rms_norm_eps,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False  # Disable caching during training
    )

    print(f"Creating student model with {student_config.num_hidden_layers} layers")
    student_model = AutoModelForCausalLM.from_config(student_config)
    student_model.gradient_checkpointing_enable()
    student_model = student_model.to(device)
    print(f"Student model created with {sum(p.numel() for p in student_model.parameters())} parameters")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=500,
        num_train_epochs=5,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        per_device_train_batch_size=1,  # Reduced from 2
        per_device_eval_batch_size=1,  # Reduced from 2
        weight_decay=0.01,
        save_total_limit=3,
        fp16=True,
        gradient_accumulation_steps=64,  # Increased from 16
        logging_steps=100,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    print("Training arguments initialized successfully.")

    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        alpha=0.7,
        temperature=4.0,
    )

    print("Trainer setup complete.")

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    print("Saving model...")
    trainer.save_model("./improved_small_medalpaca_model")
    print("Model saved.")

    print("Evaluating model on validation set...")
    validation_results = trainer.evaluate(eval_dataset=validation_dataset)
    print("Validation Results:", validation_results)

    sample_prompts = [
        "Describe the symptoms of",
        "What is the ICD-10 code for",
        "Explain the treatment for",
    ]

    print("\nTesting the student model on sample prompts:")
    for prompt in sample_prompts:
        generated_text = generate_text(prompt, student_model, tokenizer)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}\n")

    print("\nPerforming comprehensive evaluation on the validation set...")
    test_results = comprehensive_evaluation(student_model, validation_dataset)
    print("Test Set Results:", test_results)

    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())

    print(f"Teacher Model Parameters: {teacher_params}")
    print(f"Student Model Parameters: {student_params}")
    print(f"Parameter Reduction: {(teacher_params - student_params) / teacher_params * 100:.2f}%")

    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(student_model, validation_dataset, tokenizer)

    print("\nEvaluating inference time...")
    student_inference_time = evaluate_inference_time(student_model, validation_dataset)
    teacher_inference_time = evaluate_inference_time(teacher_model, validation_dataset)

    print("Student Model Inference Time:", student_inference_time)
    print("Teacher Model Inference Time:", teacher_inference_time)

    print("Script execution completed.")


    #Notes: Spilit -> save first 4 and last couple (34-40) -> 
    #Take the model that and see their output - double check if it just has to be bigger or not 
    #Run it for a fewer epochs and then do checkpoints - start from a checkpoint then continue training 
