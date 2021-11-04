
# python inference_serini.py \
#     --output_dir ./outputs/dh_serini_pre_hard15_0.4_wnorm/ \
#     --dataset_name ../data/train_dataset/ \
#     --model_name_or_path Doohae/roberta \
#     --top_k_retrieval 9 \
#     --serini_ratio 0.4 \
#     --do_eval \

for ratio in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
do
    python inference_serini.py \
        --output_dir ./outputs/dh_serini_pre_hard15_${ratio}_eval/
        --dataset_name ../data/train_dataset/ \
        --model_name_or_path Doohae/roberta \
        --top_k_retrieval 9 \
        --serini_ratio ${ratio} \
        --do_eval \
    done
done

for ratio in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
do
    python inference_serini.py \
        --output_dir ./outputs/dh_serini_pre_hard15_${ratio}_eval/
        --dataset_name ../data/test_dataset/ \
        --model_name_or_path Doohae/roberta \
        --top_k_retrieval 9 \
        --serini_ratio ${ratio} \
        --do_predict \
done