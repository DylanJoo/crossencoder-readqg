# | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

data_dir=/work/jhju/beir-runs

echo trec-covid
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  $data_dir/qrels.beir-v1.0.0-trec-covid.test.txt \
  $data_dir/run.beir.bm25-multifield.trec-covid.txt 

echo NFCorpus
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  $data_dir/qrels.beir-v1.0.0-nfcorpus.test.txt \
  $data_dir/run.beir.bm25-multifield.nfcorpus.txt

echo FiQA
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  $data_dir/qrels.beir-v1.0.0-fiqa.test.txt \
  $data_dir/run.beir.bm25-multifield.fiqa.txt

echo arguana
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  $data_dir/qrels.beir-v1.0.0-arguana.test.txt \
  $data_dir/run.beir.bm25-multifield.arguana.txt

echo Touche
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  $data_dir/qrels.beir-v1.0.0-webis-touche2020.test.txt \
  $data_dir/run.beir.bm25-multifield.webis-touche2020.txt

echo DBPedia
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  $data_dir/qrels.beir-v1.0.0-dbpedia-entity.test.txt \
  $data_dir/run.beir.bm25-multifield.dbpedia-entity.txt

echo Scidocs
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  $data_dir/qrels.beir-v1.0.0-scidocs.test.txt \
  $data_dir/run.beir.bm25-multifield.scidocs.txt

echo climate-fever
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  $data_dir/qrels.beir-v1.0.0-climate-fever.test.txt \
  $data_dir/run.beir.bm25-multifield.climate-fever.txt

echo scifact
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  $data_dir/qrels.beir-v1.0.0-scifact.test.txt \
  $data_dir/run.beir.bm25-multifield.scifact.txt
