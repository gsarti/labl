sbatch_gpu_big_short "02-xcomet-divemt" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_xcomet_metrics.py \
    --model_id Unbabel/XCOMET-XXL \
    --dataset_name divemt \
    --langs ara ita nld tur ukr vie \
    --batch_size 4
"

sbatch_gpu_big_short "02-xcomet-wmt24" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_xcomet_metrics.py \
    --model_id Unbabel/XCOMET-XXL \
    --dataset_name wmt24esa \
    --langs en-cs en-hi en-ja en-zh cs-uk en-ru \
    --batch_size 4
"