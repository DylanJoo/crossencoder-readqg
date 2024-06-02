import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import MarginRankingLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

class InbatchBCELoss(nn.Module):
    def __init__(self, 
                 examples_per_group: int = 1, 
                 reduction: str = 'mean', 
                 batch_size: int = None, 
                 negative_selection: 'str' = 'all'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = BCEWithLogitsLoss(reduction=reduction)
        self.batch_size = batch_size
        self.negative_selection = negative_selection
        assert (negative_selection != 'all') == (reduction == 'none'), \
                'reduction should be the none thus negative selection.'

    def forward(self, logits: Tensor, labels: Tensor):
        if self.batch_size:
            n_rows = self.batch_size * 1 # this can be more than 1
            logits = logits.view(n_rows, -1) 
        else:
            n_cols = self.examples_per_group
            logits = logits.view(-1, n_cols) # reshape (B n). this is document-centirc
        targets = torch.zeros(logits.size()).to(logits.device)
        targets[:, 0] = 1.0

        # pooled the logits and targets into one list
        loss = self.loss_fct(logits.view(-1), targets.view(-1))

        if self.negative_selection == 'hard':
            loss_matrix = loss.view(targets.size())
            loss = loss_matrix[:, 0].mean()
            loss += loss_matrix[:, 1:].max(-1).values.mean()
            return loss / 2
        elif self.negative_selection == 'random':
            loss_matrix = loss.view(targets.size())
            B, N = targets.size(0), targets.size(1)
            samples = 1+torch.randint(N-1, (B, 1), device=logits.device)
            loss = loss_matrix[:, 0].mean()
            loss += loss_matrix.gather(1, samples).mean()
            return loss / 2
        else:
            return loss.mean()

class HingeLoss(nn.Module):
    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 1, 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = MarginRankingLoss(margin=margin, reduction=reduction)
        self.activation = nn.Sigmoid()

    def forward(self, logits: Tensor, labels: Tensor):
        """ Try using labels as filter"""
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        logits_negaitve = logits[:, 0] # see `filter`
        logits_positive = logits[:, 1] # see `filter`
        targets = torch.ones(logits.size(0)).to(logits.device) 
        # 1 means left>right
        loss = self.loss_fct(logits_positive, logits_negaitve, targets)
        return loss

class GroupwiseHingeLoss(HingeLoss):
    """ [NOTE 1] It can also be like warp """
    def forward(self, logits: Tensor, labels: Tensor):
        loss = 0
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        targets = torch.ones(logits.size(0)).to(logits.device)
        for idx in range(logits.size(-1)-1):
            loss += self.loss_fct(logits[:, 0], logits[:, idx+1], targets)
        return loss / (logits.size(-1) - 1)

class GroupwiseHingeLossV1(nn.Module):
    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 1, 
                 stride: int = 1,    # the size between selected positions
                 dilation: int = 1,  # the position of the paired negative 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = MarginRankingLoss(
                margin=margin, 
                reduction=reduction
        )
        self.activation = nn.Sigmoid()
        self.stride = stride
        self.dilation = dilation
        self.sample_indices = list(
                range(0, examples_per_group-dilation, stride)
        )

        for i, idx in enumerate(self.sample_indices):
            print(f"The {i+1} pair: + {idx}; - {idx+dilation}")

    def forward(self, logits, labels):
        loss = 0
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        targets = torch.ones(logits.size(0)).to(logits.device) 
        for idx in self.sample_indices:
            logits_negative = logits[:, (idx+self.dilation)]
            loss += self.loss_fct(logits[:, idx], logits_negative, targets)
        return loss / len(self.sample_indices)

class CELoss(nn.Module):
    """ [NOTE] Temperature is a hyperparameter. """
    def __init__(self, 
                 examples_per_group: int = 1, 
                 reduction: str = 'mean', 
                 batch_size: int = None,
                 temperature: float = 1.0):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = CrossEntropyLoss(reduction=reduction)
        self.batch_size = batch_size
        self.tau = temperature

    def forward(self, logits: Tensor, labels: Tensor):
        if self.batch_size:
            n_rows = self.batch_size * 1 # this can be more than 1
            logits = logits.view(n_rows, -1) 
        else:
            n_cols = self.examples_per_group
            logits = logits.view(-1, n_cols) # reshape (B n). this is document-centirc
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        return self.loss_fct(logits/self.tau, targets)

class GroupwiseCELoss(CELoss):
    def forward(self, logits, labels):
        loss = 0
        logits = logits.view(-1, self.examples_per_group)
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        for idx in range(logits.size(-1) - 1):
            loss += self.loss_fct(logits[:, [0, idx+1]], targets)
        return loss / (logits.size(-1) - 1)

class GroupwiseCELossV1(nn.Module):
    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 1, 
                 stride: int = 1,    # the size between selected positions
                 dilation: int = 1,  # the position of the paired negative 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = CrossEntropyLoss(reduction=reduction)
        self.stride = stride
        self.dilation = dilation
        self.sample_indices = list(
                range(0, examples_per_group-dilation, stride)
        )

        for i, idx in enumerate(self.sample_indices):
            print(f"The {i+1} pair: + {idx}; - {idx+dilation}")

    def forward(self, logits, labels):
        loss = 0
        logits = logits.view(-1, self.examples_per_group)
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        for idx in self.sample_indices:
            logits_ = logits[:, [idx, (idx+self.dilation)] ]
            loss += self.loss_fct(logits_, targets)
        return loss / len(self.sample_indices)

class MSELoss(nn.Module):

    def __init__(self, 
                 examples_per_group: int = 1, 
                 reduction='mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.activation = nn.Sigmoid()
        self.loss_fct = MSELoss(reduction='mean')

    def forward(self, logits: Tensor, labels: Tensor):
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        labels = labels.view(-1, self.examples_per_group)
        return self.loss_fct(logits, labels)
