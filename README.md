# Sentiment classifier using BERT and IMDB dataset


## Core Workflow
1. Download the IMDB data set using datasets with default cache location of ~/.cache/huggingface/datasets. 
2. Transfom the dataset using the tokenizer of BERT
3. Train and test the dataset using the trainer with 3000 reviews. Print the stats using pandas.
4. Save the trained model.
5. Infer for positive/negative/boundary conditions. 


## Dependencies
*transformers*: This is the core library by Hugging Face. It provides the pre-trained model architectures (like DistilBERT, GPT, etc.) and the tools to load, use, and fine-tune them.

*accelerate*: This library acts as a "hardware abstraction layer." It handles the tricky details of running your code across different hardware (CPU, GPU, or Apple's MPS) automatically, so you don't have to manually manage devices in your code.

*datasets*: This makes data handling efficient. Instead of loading a 50GB file into your RAM, it uses memory-mapping to stream data from your disk, ensuring your training process doesn't crash your computer.

*evaluate*: A centralized library for metrics. It gives you a consistent way to calculate accuracy, F1-score, or other benchmarks so you can compare your model's performance reliably.

*pandas*: The "Swiss Army Knife" for data. In your script, you use it to organize your logs and training history into a table (DataFrame) for easy viewing or exporting.