# runing exps
variant=ibhn.qq-hinge
decoding=top10
data_dir=/work/jhju/beir-readqg/
model_dir=/work/jhju/oodrerank.readqg.${decoding}

for data in calibrate;do
    for name in scidocs;do
        for file in $data_dir/${name}_${decoding}/*${data}*.jsonl;do

            readqg=${file/.jsonl/}
            readqg=${readqg##*/}
            readqg=${readqg%.*}
            qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
            run_bm25=/work/jhju/beir-runs/run.beir.bm25-multifield.$name.txt

            python reranking/cross_encoder_train.py \
                --dataset datasets/$name \
                --pseudo_queries $file \
                --output_path ${model_dir}/${variant}/${name}/${readqg} \
                --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
                --batch_size 8 \
                --max_length 384 \
                --num_epochs 2 \
                --learning_rate 7e-6 \
                --do_eval \
                --qrels $qrels \
                --run_bm25 $run_bm25 \
                --filtering '{"name": "top_bottom", "n1": 1, "n2": 1}' \
                --query_centric \
                --objective_qc groupwise_bce_hard \
                --document_centric \
                --objective_dc hinge \
                --change_dc_to_qq \
                --margin 0 \
                --device cuda
        done
    done
done
