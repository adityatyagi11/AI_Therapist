import argparse
import os
import torch
import pandas as pd
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from utils import prepare_dataset_for_gpt2

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on mental health datasets")
    parser.add_argument("--dataset", type=str, default="amod", choices=["amod", "psychology", "empathetic_dataset_100"],
                        help="Dataset to use for fine-tuning")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Base model name or path to previously fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="./models",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
   
    os.makedirs(args.output_dir, exist_ok=True)
    
   
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token by default
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    
    # Load dataset
    if args.dataset == "amod":
        dataset = load_dataset("Amod/mental_health_counseling_conversations")
    elif args.dataset == "empathetic_dataset_100":
        dataset = pd.read_json("Data/empathetic_dataset_100.json")
    else:
        dataset = load_dataset("samhog/psychology-10k")
    
    tokenized_dataset = prepare_dataset_for_gpt2(dataset, tokenizer, args.max_length)
    
   
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  
    )
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"gpt2_{args.dataset}"),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="no",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",  
    )
    
   
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
    )
    
   
    trainer.train()
    
   
    model.save_pretrained(os.path.join(args.output_dir, f"final_{args.dataset}"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, f"final_{args.dataset}"))
    
    print(f"Model fine-tuned on {args.dataset} and saved to {args.output_dir}/final_{args.dataset}")

if __name__ == "__main__":
    main()