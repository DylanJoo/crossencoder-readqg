import os
import collections
import logging
import tqdm
import json
import numpy as np

class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def load_corpus(path):
    corpus = {}
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            docid = data['_id']
            title = data.get('title', "").strip()
            text = data.get('text', "").strip()
            corpus[str(docid)] = title + " " + text
    print('Example document:', title + " " + text, \
            '\ntotal amount', len(corpus))
    return corpus

def load_queries(path):
    queries = {}
    with open(os.path.join(path), 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            qid = data['_id']
            text = data['text'].strip()
            queries[str(qid)] = text
    # print('Example query:', text, \
    #         '\ntotal amount', len(queries))
    return queries

def load_pseudo_queries(path):
    pcentric_queries = {}
    with open(os.path.join(path), 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            docid = data['doc_id']
            scores = data['relevance_scores']
            texts = data['generated_query']
            to_return = [(t, s) for t, s in zip(texts, scores)]
            # sort by (given) scores (from large to small)
            pcentric_queries[str(docid)] = sorted(to_return, key=lambda x: x[1], reverse=True) 

    # print('Example pseudo query:', texts[0], scores[0], \
    #         '\ntotal amount', len(pcentric_queries))
    return pcentric_queries

def load_results(path, topk=2000):
    input_run = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= topk:
                input_run[str(qid)].append(str(docid))

    # print('Example run', (qid, docid, rank, score), \
    #         'total amount', len(input_run))
    return input_run

def load_and_convert_qrels(path, queries, corpus_texts, use_random_negatives=None, use_bm25_negatives=False):
    if use_bm25_negatives:
        doc_to_rerank = load_results(use_bm25_negatives, topk=100)
    else:
        doc_to_rerank = collections.defaultdict(list)

    positives = collections.defaultdict(list)
    negatives = collections.defaultdict(list)
    all_negative_docids = []

    # load qrels
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if int(rel) > 0:
                try:
                    positives[qid].append(corpus_texts[docid].strip())
                except:
                    print(f'the document {docid} was not found')
                try:
                    doc_to_rerank[qid].remove(docid)
                except:
                    pass
            else:
                negatives[qid].append(corpus_texts[docid].strip())
                all_negative_docids.append(docid)

    # add random (in-batch) negatives for evaluation
    all_negative_docids = list(set(all_negative_docids))
    N_neg = len(all_negative_docids)

    # convert qrels to samples
    sample_list = []
    for qid in positives:
        if use_random_negatives:
            selected_negative_docids = [all_negative_docids[i] for i in np.random.randint(0, N_neg, use_random_negatives)]
            negatives[qid] += [corpus_texts[docid].strip() for docid in selected_negative_docids]
        if use_bm25_negatives:
            negatives[qid] += [corpus_texts[docid].strip() for docid in doc_to_rerank[qid]]
        sample_list.append({
            "query": queries[qid].strip(),
            "positive": positives[qid], 
            "negative": negatives[qid]
        })
    return sample_list

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]
