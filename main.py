from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import pandas as pd
import torch
import os

device = "mps" if torch.mps.is_available else "cpu"
print(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(tokenizer)

SAVE_DIR = "imdb_model"

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

def predict(model,text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    model.to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    print(text, outputs.logits)
    prediction = outputs.logits.argmax(dim=1)
    return prediction.item()

def train(save_dir: str):
    #
    # 1. Prepare the data
    #
    
    # 1.1. Load default cache dir: ~/.cache/huggingface/datasets
    dataset = load_dataset("imdb") # use can customize cache_dir="/path/to/my/big/drive/cache"

    # 1.2 Transform for training 
    # Dataset.map() method (whether batched or not) is cache-aware by design:
    # It writes these to cache files on disk (usually in ~/.cache/huggingface/datasets/.../cache-*.arrow files). 
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    #
    # 2. Create the model and train
    #
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)
    training_args = TrainingArguments(output_dir="./results",
        eval_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir="./logs")
    trainer = Trainer(model=model,args=training_args,
        train_dataset=dataset["train"].shuffle(seed=42).select(range(3000)),
        eval_dataset=dataset["test"].shuffle(seed=42).select(range(1000))
    )
    trainer.train()

    #
    # Print trainer stats
    #
    history = trainer.state.log_history
    df = pd.DataFrame(history)
    print(df)

    #
    # 3. Save the model
    #
    trainer.save_model(save_dir)

def is_model_saved(save_dir: str) -> bool:
    if not os.path.isdir(save_dir):
        return False
    
    files = set(os.listdir(save_dir))
    # Check for the essentials (safetensors is now standard)
    has_weights = "model.safetensors" in files or "pytorch_model.bin" in files
    has_config  = "config.json" in files
    
    return has_weights and has_config

def main():
    if not is_model_saved(SAVE_DIR):
        print(f"No saved model found in {SAVE_DIR} → starting training")
        train(SAVE_DIR)
    else:
        print(f"Saved model found in {SAVE_DIR} → starting inference")
    # Load and Infer
    
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR) 

    print("===== Negative =======")
    print("-Prediction 1:", predict(model,"This movie was terrible and boring."))
    print("-Prediction 2:", predict(model,"Damn can't watch even one time."))
    print("-Prediction 3:", predict(model,"Food was cold, tasteless, and the portion was tiny for the price."))
    print("-Prediction 4:", predict(model,"The book was so slow and predictable — I skimmed the last 100 pages."))
    print("-Prediction 5:", predict(model,"The concert was a disaster — sound was terrible and the band was off-key."))    
    print("")
    print("===== Positive =======")
    print("+Prediction 1:", predict(model,"That was the most awesome movie i have ever seen."))
    print("+Prediction 2:", predict(model,"Super cool movie."))
    print("+Prediction 3:", predict(model,"Most exciting movie ever"))
    print("+Prediction 4:", predict(model,"The customer service was outstanding — they really went above and beyond."))
    print("+Prediction 5:", predict(model,"This album is a masterpiece. Every track is pure gold."))
    print("")
    print("===== Tricky =======")
    print("+/-Prediction 1:", predict(model,"It's not bad, but I've seen much better versions of this kind of thing."))
    print("+/-Prediction 2:", predict(model,"I wanted to like it, but it just didn't click for me."))
    print("+/-Prediction 3:", predict(model,"Visually stunning movie — unfortunately the plot was weak."))
    print("+/-Prediction 4:", predict(model,"It was okay I guess… nothing special but not terrible either."))
    print("+/-Prediction 5:", predict(model,"Super slow shipping, but the product itself is nice."))


if __name__ == "__main__":
    main()
