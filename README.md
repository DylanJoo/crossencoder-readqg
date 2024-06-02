# Crossencoder with ReadQG-augmeneted data
This repository is only for fine-tuning cross-encoder with ReadQG dataset. The details of ReadQG (generator) is in this repo: [ReadQG](https://github.com/DylanJoo/readqg).

---
### BEIR Dataset
We download the datasets from [BEIR repository](https://github.com/beir-cellar/beir).

### Generated ReadQG Data (for training cross-encoder)
For each dataset in BEIR, we've constructed 10 queries for each documents. 
We uploaded the datset in [Huggingface ](https://huggingface.co/datasets/DylanJHJ/beir-readqg). 
The dataset includes mulitple versions of data, comprsing of 
(1) different ``decoding`` methods (greedy, top10, beam3), and 
(2) different ``generator`` models. 

Please checkout [ReadQG](https://github.com/DylanJoo/readqg) repo for more details regarding the generator.
