sbatch_gpu_big_short "01-aya-encs" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_unsupervised_metrics.py \
        --model_id CohereLabs/aya-23-35B \
        --dataset_name wmt24esa \
        --langs en-cs \
        --output_dir outputs/metrics/wmt24esa;
    "