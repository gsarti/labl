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
        --cpus-per-task=9 \
        --mem-per-cpu=12G --time=0-4 \
        --wrap="$JOB_WRAP";
}
function sbatch_gpu_big_short() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:30g \
        --mail-type END \
        --mail-user vilem.zouhar@gmail.com \
        --ntasks-per-node=1 \
        --cpus-per-task=9 \
        --mem-per-cpu=12G --time=0-4 \
        --wrap="$JOB_WRAP";
}

sbatch_gpu_bigg_short "02-xcomet-divemt" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_xcomet_metrics.py \
    --model_id Unbabel/XCOMET-XXL \
    --dataset_name divemt \
    --langs ara ita nld tur ukr vie \
    --batch_size 4
"


sbatch_gpu_bigg_short "02-xcomet-wmt24" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_xcomet_metrics.py \
    --model_id Unbabel/XCOMET-XXL \
    --dataset_name wmt24esa \
    --langs en-cs en-hi en-ja en-zh cs-uk en-ru \
    --batch_size 4
"

sbatch_gpu_bigg_short "02-xcomet-qe4pe" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_xcomet_metrics.py \
    --model_id Unbabel/XCOMET-XXL \
    --dataset_name qe4pe \
    --langs ita nld \
    --batch_size 4 \
    --do_continuous
"

sbatch_gpu_bigg_short "02-xcometxl-wmt24" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_xcomet_metrics.py \
    --model_id Unbabel/XCOMET-XL \
    --dataset_name wmt24esa \
    --langs en-cs en-hi en-ja en-zh cs-uk en-ru \
    --batch_size 4
"

sbatch_gpu_big_short "02-xcometxl-qe4pe" "\
    cd examples/unsup_wqe; \
    python3 scripts/compute_xcomet_metrics.py \
    --model_id Unbabel/XCOMET-XL \
    --dataset_name qe4pe \
    --langs ita nld \
    --batch_size 4
"