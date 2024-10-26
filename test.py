import torch
from datasets import load_dataset, Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import pandas as pd
from typing import Optional, Dict, Any
import numpy as np

def load_and_preprocess_csv_data(file_path: str, test_size: float = 0.1):
    """Load CSV data and preprocess it."""
    # Load CSV data into a DataFrame
    df = pd.read_csv(file_path)
    
    # Ensure the DataFrame has 'article' and 'highlights' columns
    required_columns = ['article', 'highlights']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Take a subset for faster testing (10% of the data)
    subset_size = int(len(df) * 0.1)
    df_subset = df.iloc[:subset_size]
    
    # Convert DataFrame to a Huggingface Dataset
    dataset = Dataset.from_pandas(df_subset)
    
    # Split into train and validation sets
    dataset = dataset.train_test_split(test_size=test_size)
    
    return dataset

def preprocess_function(examples, tokenizer, max_length: int = 512):
    """Preprocess the examples for GPT-2."""
    # Format: "Summarize: {article} TL;DR: {summary}"
    prompts = [
        f"Summarize: {article} TL;DR: {summary}"
        for article, summary in zip(examples['article'], examples['highlights'])
    ]
    
    # Tokenize the prompts
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create labels (same as input_ids for causal language modeling)
    tokenized['labels'] = tokenized['input_ids'].clone()
    
    return tokenized

class GPT2SummarizationTrainer:
    def __init__(
        self,
        csv_path: str,
        max_length: int = 512,
        batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        output_dir: str = "./gpt2_summarization_model"
    ):
        self.csv_path = csv_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        
        # Initialize GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize GPT-2 model
        self.model = GPT2LMHeadModel.from_pretrained(
            'gpt2',
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare model for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],  # GPT-2 specific attention modules
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA adapter
        self.model = get_peft_model(self.model, lora_config)
        
        # Load dataset
        self.dataset = load_and_preprocess_csv_data(csv_path)
    
    def preprocess_dataset(self):
        """Preprocess the dataset."""
        tokenized_dataset = self.dataset.map(
            lambda x: preprocess_function(x, self.tokenizer, self.max_length),
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        return tokenized_dataset
    
    def create_trainer(self, tokenized_dataset):
        """Create trainer instance."""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            remove_unused_columns=False,  # Important for GPT-2
            gradient_accumulation_steps=4,  # Helps with memory usage
        )
        
        # Use GPT-2 specific data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Important: GPT-2 uses causal language modeling
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )
        
        return trainer
    
    def train(self):
        """Train the model."""
        print("Preprocessing dataset...")
        tokenized_dataset = self.preprocess_dataset()
        
        print("Creating trainer...")
        trainer = self.create_trainer(tokenized_dataset)
        
        print("Starting training...")
        trainer.train()
        
        print("Saving model...")
        trainer.save_model(self.output_dir)
    
    def generate_summary(self, article: str) -> str:
        """Generate a summary for a given article."""
        prompt = f"Summarize: {article} TL;DR:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=4,
                no_repeat_ngram_size=3,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the summary part (after TL;DR:)
        summary = summary.split("TL;DR:")[-1].strip()
        return summary

def main():
    # Initialize trainer with path to your CSV file
    trainer = GPT2SummarizationTrainer(
        csv_path="./train.csv",  # Replace with your CSV file path
        max_length=512,
        batch_size=4,
        num_epochs=3,
        learning_rate=2e-5,
        output_dir="./gpt2_summarization_model"
    )
    
    # Train the model
    trainer.train()
    
    # Example usage
    article = """
    The researchers discovered a new species of butterfly in the Amazon rainforest. 
    The butterfly, which has distinctive blue wings with golden spots, was found in 
    a remote area that had never been explored before. Scientists believe this 
    discovery could indicate the presence of other unknown species in the region.
    """
    
    summary = trainer.generate_summary(article)
    print(f"Generated Summary: {summary}")

if __name__ == "__main__":
    main()