for topk in 5 3;
do
    python dense_bm25.py \
        --mode eval \
        --topk ${topk} \
        --output_dir dense_retrieval_allhard_20
done