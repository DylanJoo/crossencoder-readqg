from sentence_transformers.cross_encoder import CrossEncoder
from cross_encoder import StandardCrossEncoder, PACECrossEncoder
from sentence_transformers import InputExample
from operator import itemgetter
import random

import os
import json
import datetime
import logging
import argparse 
import wandb
import torch

from torch.utils.data import DataLoader

from pacerr.filters import filter_function_map
from pacerr.utils import load_corpus, load_pseudo_queries
from pacerr.utils import load_queries, load_and_convert_qrels
from pacerr.utils import LoggingHandler
from pacerr.inputs import GroupInputExample
from pacerr.loss_handler import LossHandler
from pacerr.evaluation import CERerankingEvaluator_ndcg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--pseudo_queries", type=str, default=None)
    parser.add_argument("--qrels", type=str, default=None)
    parser.add_argument("--run_bm25", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--init_model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_length", type=int, default=384)
    # setting
    parser.add_argument("--filtering", type=str, default="{}")
    # training
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--objective_dc", type=str, default='')
    parser.add_argument("--objective_qc", type=str, default='')
    parser.add_argument("--query_centric", action='store_true', default=False)
    parser.add_argument("--document_centric", action='store_true', default=False)
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--change_dc_to_qq", action='store_true', default=False)
    parser.add_argument("--q_self_as_anchor", action='store_true', default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reduction", type=str, default='mean')
    # evaluation
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    # saving 
    parser.add_argument("--save_last", action='store_true', default=False)
    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #### Reranking using Cross-Encoder model
    if 'pointwise' in args.objective_dc or 'pointwise' in args.objective_qc:
        reranker = StandardCrossEncoder(args.model_name, num_labels=1,)
    else:
        reranker = PACECrossEncoder(args.model_name, 
                                    num_labels=1, 
                                    device=args.device,
                                    max_length=args.max_length,
                                    query_centric=args.query_centric,
                                    document_centric=args.document_centric,
                                    change_dc_to_qq=args.change_dc_to_qq,
                                    q_self_as_anchor=args.q_self_as_anchor)

    #### Add wandb 
    if args.debug:
        objectives = "debug"
    else:
        objectives = f"qc:{args.objective_qc}-dc:{args.objective_dc}"
        if args.change_dc_to_qq:
            objectives += "-qq"

    os.environ["WANDB_PROJECT"] = f"{objectives.replace(':', '-')}"
    wandb.init(
            name=f"{args.pseudo_queries.split('/')[-1]}-{objectives}",
            config=reranker.config
    )
    wandb.watch(reranker.model, log_freq=10)

    if args.init_model_name is not None:
        reranker.perge(init_name=args.init_model_name)
        logging.info(f'Initialize with {args.init_model_name}')

    #### Load data
    corpus_texts = load_corpus(os.path.join(args.dataset, 'corpus.jsonl'))
    pseudo_queries = load_pseudo_queries(args.pseudo_queries)

    #### Prepare a filter
    filter_args = json.loads(args.filtering)
    filter_name = filter_args.pop('name', 'testing')
    filter_fn = filter_function_map[filter_name]

    #### Prepare examples
    train_samples = []
    dev_samples = []
    for docid in pseudo_queries:
        document = corpus_texts[docid]

        #### Filtering
        pairs = filter_fn(pseudo_queries[docid], **filter_args)
        queries, scores = map(list, (list(zip(*pairs))) )
        train_samples.append(GroupInputExample(
            center=document, texts=queries, labels=scores
        ))
        # [deprecated] only consider pooled BCE when using pointwise
        # if 'pointwise' in args.objective_dc or 'pointwise' in args.objective_qc:
        #     for query, score in pairs:
        #         train_samples.append(InputExample(texts=[query, document], label=score))

    #### Prepare dataloader
    # [NOTE] remove this: collate_fn=reranker.smart_batching_collate, # in fact, no affect
    torch.manual_seed(2024)
    train_dataloader = DataLoader(
            train_samples, 
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True, 
            drop_last=True
    )
    n = len(scores)


    #### Prepare losses
    logging.info(
            f'The training data was built on document-wise batch, which is not in common.' +  \
            f'But we will replicate it and fit it to query-wise.' +  \
            f'Note that there are {len(scores)} queries per document batch'
    )
    loss_handler = LossHandler(
            examples_per_group=n,
            batch_size=args.batch_size,
            margin=args.margin,
            reduction=args.reduction,
            stride=1,
            dilation=1,
            logger=logging,
            temperature=args.temperature
    )
    loss_fct_dc = loss_handler.loss(args.objective_dc, False)
    loss_fct_qc = loss_handler.loss(args.objective_qc, True)

    #### Saving benchmark times
    start = datetime.datetime.now()

    #### Add evaluation
    logging.info(f'Loading the evaluation dataset...if provided')
    evaluator = None
    if args.do_eval:
        queries = load_queries(os.path.join(args.dataset, 'queries.jsonl'))
        dev_samples = load_and_convert_qrels(
                path=args.qrels,
                queries=queries,
                corpus_texts=corpus_texts,
                use_bm25_negatives=args.run_bm25
        )
        evaluator = CERerankingEvaluator_ndcg(dev_samples, name='test')

    #### Start training
    logging.info(f"The dataset has {len(train_dataloader)} batch")
    reranker.fit(
            train_dataloader=train_dataloader,
            loss_fct_dc=loss_fct_dc,
            loss_fct_qc=loss_fct_qc,
            evaluator=evaluator,
            epochs=args.num_epochs,
            evaluation_steps=len(train_dataloader) // 5,
            warmup_steps=len(train_dataloader) // 10,
            optimizer_params={'lr': args.learning_rate},
            output_path=args.output_path, # only save when evaluation --> only save per epoch
            save_best_model=False,
            use_amp=True,
            wandb=wandb
    )
    if args.save_last:
        reranker.save(args.output_path)

    #### Measure time to 
    end = datetime.datetime.now()
    
    #### Measuring time taken in ms (milliseconds)
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    logging.info("Training time: {:.2f}ms".format(time_taken))
