<h2>The settings of PACE-RR</h2>

## Filters

1. Identity
2. Boundary
    - params: num: int = 1
    - params: random-sample: int = 0
3. Top 
    - params: num: int = 1


## Objectives

### Document-centric objectives
1. Regression loss --
we use the relevance scores ranged of [0, 1] as labels.
    - BCELogitLoss
    - Binary MSE
    - Distillation MSE

2. Ranking pairwise loss --
we use the relevance scores 0 and 1 as a negative and a positive.
    - Hinge
    - CrossEntropy (CE)

3. Ranking pairwise loss with negative queries --
in addition to the boundary 0 and 1, we pair the positive one and each of the other negatives with relevance scores between 0 and 1 (as soft negative). For example, [ {d, q+}-{d, q--}, {d, q+}-{d, q-}, ...]
    - Hinge 
    - CE

4. Ranking pairwise loss (V1) within rolling relevance --
For example, [ {d, qi}-{d, qi+1}, {d, qi+1}-{d, qi+2}, ...]
    - Hinge 
    - CE

5. Ranking pairwise loss (V2) with query  
For example, [ {d, qi, qi+1, ...qn} ]
    - Hinge 
    - CE

### Query-centric objectives
1. Ranking groupwise In-batch negatives
    - CE

### Composite loss
1. TBD
