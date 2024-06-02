from typing import Dict, Type, Callable, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration
from transformers import AutoTokenizer
from sentence_transformers.cross_encoder import CrossEncoder

class MonoT5Reranker(CrossEncoder):

    def __init__(self, 
                 model_name='castorini/monot5-base-msmarco',
                 max_length: int = None,
                 verbose=True):

        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.true_id = self.tokenizer.encode('true')[0]
        self.false_id = self.tokenizer.encode('false')[0]

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch):
        # add prompts
        texts = [[] for _ in range(len(batch[0]))]

        prompts = [
                "Query: {query} Document: {text} Relevant:".format(
                    query=query.strip(), text=text.strip()
                ) for (query, text) in batch
        ]

        tokenized = self.tokenizer(
                prompts,
                padding=True, 
                truncation=True,
                return_tensors="pt", 
                max_length=self.max_length
        ).to(self.device)

        return tokenized

    def predict(self, sentences: List[List[str]], 
                batch_size: int = 32, 
                show_progress_bar: bool = None,
                num_workers: int = 0,
                apply_softmax = False,
                convert_to_numpy: bool = True,
                convert_to_tensor: bool = False) -> List[float]:
        
        input_was_string = False
        # print(sentences[0])
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(
                sentences, 
                batch_size=batch_size, 
                collate_fn=self.smart_batching_collate_text_only, 
                num_workers=num_workers, 
                shuffle=False
        )

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        pred_scores = []
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for features in iterator:

                model_outputs = self.model.generate(
                        **features, 
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True
                )

                # extract tokens and apply softmax by default 
                model_predictions = model_outputs.scores[0][:, [self.false_id, self.true_id]]
                model_predictions = torch.nn.functional.log_softmax(model_predictions, dim=1)

                logits = model_predictions[:, 1].exp()
                pred_scores.extend(logits)

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores
