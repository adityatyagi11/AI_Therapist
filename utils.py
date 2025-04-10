from datasets import Dataset
import pandas as pd

def prepare_dataset_for_gpt2(dataset, tokenizer, max_length):
    formatted_dataset = {}

    if isinstance(dataset, dict) and "train" in dataset:
        try:
            dataset_name = dataset["train"].info.dataset_name.lower()
        except AttributeError:
            dataset_name = ""

        if "mental_health_counseling_conversations" in dataset_name:
            def format_amod(examples):
                conversations = []
                for query, response in zip(examples["Context"], examples["Response"]):
                    conversation = f"Human: {query}\nAssistant: {response}"
                    conversations.append(conversation)
                return {"text": conversations}
            
            formatted_dataset["train"] = dataset["train"].map(format_amod, batched=True)

        elif "psychology" in dataset_name:
            def format_psychology(examples):
                conversations = []
                for input_text, output_text in zip(examples["input"], examples["output"]):
                    conversation = f"Human: {input_text.strip()}\nAssistant: {output_text.strip()}"
                    conversations.append(conversation)
                return {"text": conversations}
            
            formatted_dataset["train"] = dataset["train"].map(format_psychology, batched=True)

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    elif isinstance(dataset, pd.DataFrame):
        
        formatted_dataset["train"] = Dataset.from_pandas(dataset[["text"]])

    else:
        raise ValueError("Dataset must be either a Hugging Face DatasetDict or a pandas DataFrame.")

  
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized_dataset = {
        "train": formatted_dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset["train"].column_names,
        )
    }

    return tokenized_dataset
