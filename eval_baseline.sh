data_dir=/work/jhju/beir-runs

for model in ms-marco-MiniLM-L-6-v2;do
    echo $model 
    for name in scidocs;do
        echo -e $name " ";
        ~/trec_eval-9.0.7/trec_eval \
            -c -m ndcg_cut.10 \
            $data_dir/qrels.beir-v1.0.0-$name.test.txt $run \
            run.ce.top100/run.beir.$model.$name.txt \
            | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
    done
    echo -e
done

