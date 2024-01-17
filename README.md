# EPTML
EPTML (Efficient Prompt Tuning within Meta-Learning framework) is an improved (speed & accuracy) method based on the previous PBML code (https://github.com/MGHZHANG/PBML) for few-shot text classification.

# Dataset
### FewRel 
A dataset for few-shot relation classification, containing 100 relations. Each statement has an entity pair and is annotated with the corresponding relation. The position of the entity pair is given, and the goal is to predict the correct relation based on the context. The 100 relations are split into 64, 16, and 20 for training, validation, and test, respectively. 

### HuffPost headlines 
A dataset for topic classification. It contains news headlines published on HuffPost between 2012 and 2018 (Misra, 2018). The 41 topics are split into 20, 5, 16 for training, validation and test respectively. These headlines are shorter and more colloquial texts.

### Reuters-2157 
A dataset of Reuters articles over 31 classes (Lewis, 1997), which are split into 15, 5, 11 for training, validation and test respectively. These articles are longer and more grammatical texts.

### Amazon product data 
A dataset contains customer reviews from 24 product categories. Our goal is to predict the product category based on the content of the review. The 24 classes are split into 10, 5, 9 for training, validation and test respectively.


# Code
+ `train.py` contains the meta-learning framework
+ `model.py` contains the overall model architechture.
+ `dataloader.py` contains the data loading and preparing process.
+ `main.py` contains the whole runing process.

# Run

### Prepare dataset
+ For benchmark: FewRel, HuffPost, Reuters and Amazon, download the dataset processed by Bao et al.,(2020), from https://people.csail.mit.edu/yujia/files/distributional-signatures/data.zip, then put the file to the corresponding benchmark directory `data/{benchmark}/`.
+ `cd data/` and run `split_xxx.py` to split each benchmark to `train.json, val.json and test.json`

### prepare pretrained weights for BERT
+ `mkdir .bert-base-uncased` in the project directory, download `config.json  pytorch_model.bin  tokenizer.json  vocab.txt` from https://huggingface.co/bert-base-uncased/tree/main to the directory.

### Prepare label word embedding
+ For benchmark: FewRel, `data/FewRel/P-info.json` provides for each relation, a list of alias, serving as candidate words. you need to obtain this file from https://github.com/thunlp/MIML/tree/main/data
+ For benchmark: HuffPost, Reuters, and Amazon, `data/{benchmark}/candidate_words.json` contain candidate words of each class (the files are provided, you can also define your own candidate words).
+ `data/{benchmark}/candidate_ebds.json` contains candidate word embeddings of each class.  `cd data/` and run `word2ebd.py` to obtain this file.

### Run the code
+ Run command `python main.py {benchmark} {shot}` in the project directory to repeat the reported results where `{shot}` can be set to `1 or 5` for 5-way N-shot setting.

## Run supplementary experiment for LLMs
+ For HuffPost, make sure the GPU memory size >= 40GB; For other datasets, use GPU with larger memory.
+ change line No. 77 and No. 82 in model.py according to the comments
+ change line No. 33 and No. 36 in main.py according to the comments
+ `cd data/` and run `word2llm-ebd.py` to prepare the LLM word embeddings of each class 
+ `mkdir llama-2-7b-hf` in the project directory, download `config.json generation_config.json pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin pytorch_model.bin.index.json special_tokens_map.json tokenizer_config.json tokenizer.json tokenizer.model` from https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main to the directory.

# Requirements
+ Pytorch>=0.4.1
+ Python3
+ numpy
+ transformers
+ json
+ apex (https://github.com/NVIDIA/apex)
+ peft
