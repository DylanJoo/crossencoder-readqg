# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do

data_dir=/work/jhju/beir-runs
# data_dir=/home/jhju/pyserini/topics-and-qrels

decoding=$1
pseudo_q=$2
objective=$3
for name in nfcorpus fiqa arguana scidocs scifact;do
    for run in run.pacerr.top100.readqg.${decoding}/*$pseudo_q*$objective*$name*;do
        echo ${run##*/}
        ~/trec_eval-9.0.7/trec_eval \
            -c -m ndcg_cut.10 \
            $data_dir/qrels.beir-v1.0.0-$name.test.txt $run \
            | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
    done
done

