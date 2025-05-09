function sbatch_gpu_bigg_short() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:70g \
        --mail-type END \
        --mail-user vilem.zouhar@gmail.com \
        --ntasks-per-node=1 \
        --cpus-per-task=12 \
        --mem-per-cpu=9G --time=0-4 \
        --wrap="$JOB_WRAP";
}

sbatch_gpu_bigg_short "01-aya-encs-f8-nocompile" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_unsupervised_metrics_f8_nc.py \
        --model_id CohereLabs/aya-23-35B \
        --dataset_name wmt24esa \
        --langs en-cs \
        --output_dir outputs/metrics/wmt24esa;
    "
sbatch_gpu_bigg_short "01-aya-encs-f4-nocompile" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_unsupervised_metrics_f4_nc.py \
        --model_id CohereLabs/aya-23-35B \
        --dataset_name wmt24esa \
        --langs en-cs \
        --output_dir outputs/metrics/wmt24esa;
    "


sbatch_gpu_short "01-aya-encs-test" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_unsupervised_metrics_f8_nc.py \
        --model_id HuggingFaceTB/SmolLM2-135M-Instruct \
        --dataset_name wmt24esa \
        --langs en-cs \
        --output_dir outputs/metrics/test;
    "



sbatch_gpu_bigg_short "01-aya-encs-bf16-nocompile" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_unsupervised_metrics.py \
        --model_id CohereLabs/aya-23-35B \
        --dataset_name wmt24esa \
        --langs en-cs \
        --output_dir outputs/metrics/wmt24esa;
    "


load_model(
    "google/flan-t5-base",
    "dummy",
    model_kwargs={
        "attn_implementation": "eager",
        "load_in_8bit": True,
        "device_map": "auto"
    },
)