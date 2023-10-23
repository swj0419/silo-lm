for raw_file in new-amazon Github cc-news Books3 MIMIC_III NIH_ExPorter #Wikipedia_(en)
do
    for dataset in cr agn dbpedia hyp mr rotten_tomatoes rte sst2 yelp
    do
        model=kernelmachine/silo-pdsw-1.3b
        K=1024
        KNN_TEMP=1
        inter_lambda=0.7

        PYTHONPATH=. python scripts/eval.py \
            --model $model  \
            --knn_model $model \
            --n_sample 1000 \
            --raw_file $raw_file \
            --inter_lambda $inter_lambda \
            --dataset_dir data_eval/benchmark/$dataset \
            --k $K \
            --dataset_name $dataset \
            --batch_size 5 \
            --knn_temp ${KNN_TEMP} \
            --index_path /gscratch/zlab/sewon/nplm-inference/out/ours-v1_1.3B_250B_semibalanced/train-0/ \
            --tokenized_dir /gscratch/zlab/sewon/nplm-inference/out/neoX/train-0 \
            --output_dir out/lm_fuzzy_knn_fuzzy \
            --log_file_name out/lm_fuzzy_knn_fuzzy/search_config_$raw_file.out \
            --lm_fuzzy_verbalizer \
            --search_hyper_parameters
    done
done