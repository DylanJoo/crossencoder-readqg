# Crossencoder with ReadQG-augmeneted data
This repository is only for fine-tuning cross-encoder with ReadQG dataset. 
The details of ReadQG (generator) is in this repo: [ReadQG](https://github.com/DylanJoo/readqg).

---
## Preliminary

#### Requirement
```
pip install pyserini
pip install transformers
pip install datasets
pip install sentence-transformers
```

#### BEIR Dataset
- Corpus/queires. 
We download the datasets from [BEIR repository](https://github.com/beir-cellar/beir).

- First-stage retrieval results (runs)
You can reproduce via [Pyserini's 2CR](https://castorini.github.io/pyserini/2cr/beir.html).
Or download from our [Huggingface dataset](https://huggingface.co/datasets/DylanJHJ/beir-runs/tree/main). There are also preprocessed qrels, which is the same as in this repo.

#### Generated ReadQG Data (for training cross-encoder)
For each dataset in BEIR, we've constructed 10 queries for each documents. 
We uploaded the datset in [Huggingface ](https://huggingface.co/datasets/DylanJHJ/beir-readqg). 
The dataset includes mulitple versions of data, comprsing of 
(1) different ``decoding`` methods (greedy, top10, beam3), and 
(2) different ``generator`` models. 

Please checkout [ReadQG](https://github.com/DylanJoo/readqg) repo for more details regarding the generator.

#### Evaluation tools
We use the official `trec_eval` for evaluation. 
See [TREC NIST's page](https://trec.nist.gov/trec_eval/)


## Training
You should replace the data/model path in scripts. The argument description will be updated soon.
```
bash train_oodrerank.sh
```

## Reranking
You should replace the data/model path in scripts. The argument description will be updated soon.
```
bash rerank_oodrerank.sh
```
