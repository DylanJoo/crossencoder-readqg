import os
import torch
from typing import Dict, Type, Callable, List, Tuple
from sentence_transformers.cross_encoder import CrossEncoder

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm, trange

from pacerr.inputs import GroupInputExample

class StandardCrossEncoder(CrossEncoder):

    def __init__(
        self, 
        model_name: str, 
        num_labels: int = None, 
        max_length: int = None, 
        device: str = None, 
        tokenizer_args: Dict = {},
        automodel_args: Dict = {}, 
        default_activation_function = None, 
        classifier_dropout: float = None,
        query_centric: bool = False,
        document_centric: bool = False,
        change_dc_to_qq: bool = False,
        q_self_as_anchor: bool = False
    ):
        super().__init__(
                model_name, num_labels, max_length, 
                device, tokenizer_args, automodel_args, 
                default_activation_function
        )
        self.query_centric = query_centric
        self.document_centric = document_centric
        self.change_dc_to_qq = change_dc_to_qq
        self.q_self_as_anchor = q_self_as_anchor

    def perge(self, init_name):
        self.model.bert = self.model.bert.from_pretrained(init_name)

    def compute_loss(
        self, 
        features,
        labels,
        loss_fct,
        activation_fct=nn.Identity(),
    ):
        if features is not None:
            model_predictions = self.model(**features, return_dict=True)
            logits = activation_fct(model_predictions.logits)
            if self.config.num_labels == 1:
                logits = logits.view(-1)
            loss_value = loss_fct(logits, labels)
            return loss_value
        else:
            return 0

    def fit(
        self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        loss_fct_dc=None,
        loss_fct_qc=None,
        activation_fct=nn.Identity(),
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        wandb = None,
    ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct_dc: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param loss_fct_qc: the loss function for traininig query-centric (standard) examples.
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            # [NOTE] add bidirectional training
            for features_dc, labels_dc, features_qc, labels_qc in tqdm(
                train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar
            ):
                if use_amp:
                    with autocast():
                        # [NOTE] add bidirectional training
                        loss_value_dc = self.compute_loss(features_dc, labels_dc, loss_fct_dc)
                        loss_value_qc = self.compute_loss(features_qc, labels_qc, loss_fct_qc)
                        loss_value = loss_value_dc + loss_value_qc # normal addition

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    # [NOTE] add bidirectional training
                    loss_value_dc = self.compute_loss(features_dc, labels_dc, loss_fct_dc)
                    loss_value_qc = self.compute_loss(features_qc, labels_qc, loss_fct_qc)
                    loss_value = loss_value_dc + loss_value_qc # normal addition

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                
                if training_steps % 100 == 0:
                    wandb.log({"loss": loss_value, "loss_qc": loss_value_qc, "loss_dc": loss_value_dc})

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                # default evaluation and saving flow
                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    score = self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback
                    )

                    wandb.log({"eval_score": score})
                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                score = self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)
                wandb.log({"eval_score": score})
                output_path_epoch = os.path.join(output_path, str(epoch))
                self.save(output_path_epoch)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
        return score

class PACECrossEncoder(StandardCrossEncoder):

    def smart_batching_collate(self, batch):
        # document centric
        ## chnage it d1-q+, d1-q- towards q+, q-
        tokenized_dc = labels_dc = None
        if self.document_centric:
            (texts_0, texts_1), scores = self.collate_from_inputs(batch, False)
            tokenized_dc = self.tokenizer(texts_0, texts_1, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
            tokenized_dc.to(self._target_device)
            labels_dc = torch.tensor(scores, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        # query centric
        tokenized_qc = labels_qc = None
        if self.query_centric:
            batch = _reverse_batch_negative(batch)
            (texts_0, texts_1), scores = self.collate_from_inputs(batch, True)
            tokenized_qc = self.tokenizer(texts_0, texts_1, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
            tokenized_qc.to(self._target_device)
            labels_qc = torch.tensor(scores, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)
        return tokenized_dc, labels_dc, tokenized_qc, labels_qc

    def collate_from_inputs(self, batch, query_is_center=False):
        sent_left = []
        sent_right = []
        labels = []
        for example in batch:
            center = example.center.strip()
            for i, text in enumerate(example.texts):
                if query_is_center:
                    # query centric with document from other batch
                    sent_left.append(center) 
                    sent_right.append(text.strip()) 
                    labels.append(example.labels[i])
                else:
                    # document centric with queris d [q1, q2, ...qn]
                    # fixed the left as positive query
                    ## Option1: set the postiive as self-consistent
                    ## Option2: set the postiive as truth qd pair
                    if self.change_dc_to_qq and i != 0:
                        # if self.q_similarity_as_anchor:
                        sent_left.append(example.texts[0].strip()) 
                        sent_right.append(text)
                        labels.append(example.labels[i])
                    else: # i==0 and qq as anchor
                        if self.q_self_as_anchor:
                            sent_left.append(example.texts[0].strip()) 
                            sent_right.append(example.texts[0].strip())
                        else: # i==0 and qd as anchor (normal setting)
                            sent_left.append(text.strip()) 
                            sent_right.append(center)
                        labels.append(example.labels[i])
        return (sent_left, sent_right), labels


def _reverse_batch_negative(batch):
    batch_return = []

    centers = [ex.center.strip() for ex in batch]
    batch_sides = [ex.texts for ex in batch] 
    batch_labels = [ex.labels for ex in batch] 

    for i, (sides, labels) in enumerate(zip(batch_sides, batch_labels)):
        positive = [centers[i]]
        ibnegatives = centers[:i] + centers[(i+1):]

        for j, (side, label) in enumerate(zip(sides, labels)):
            # [NOTE] So far, we use only the first query as positive
            # other (j > 0) are considered as negative
            if j == 0:
                batch_return.append(GroupInputExample(
                    center=side, 
                    texts=positive+ibnegatives,
                    labels=[1]+[0]*len(ibnegatives)
                ))
    return batch_return
