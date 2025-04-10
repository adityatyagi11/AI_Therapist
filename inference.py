import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned GPT-2 model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model")
    args = parser.parse_args()
    
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    
    
    model.eval()
    
    print("Mental Health Chatbot is ready. Type 'quit' to exit.")
    print("="*50)
    
    while True:
        
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        
        prompt = f"Human: {user_input}\nAssistant:"
        
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=100 + inputs["input_ids"].shape[1],  
                temperature=0.7,  
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.2,  
                no_repeat_ngram_size=3,  
                do_sample= True,
                pad_token_id=tokenizer.eos_token_id
            )
        
       
        response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
       
        if "Assistant:" in response:
            response_parts = response.split("Assistant:", 1)
            response = response_parts[1].strip()
            
            
            if "Human:" in response:
                response = response.split("Human:")[0].strip()
        
        print(f"Assistant: {response}")
        print("-"*50)

if __name__ == "__main__":
    main()