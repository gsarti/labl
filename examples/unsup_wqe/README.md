# Unsupervised Word-level Quality Estimation for Machine Translation

## Setup

```bash
# Clone the repository
git clone https://github.com/gsarti/labl
cd labl/examples/unsup_wqe

# Create a virtual environment using uv
uv venv
source .venv/bin/activate

# Install the required packages
uv pip install -r requirements.txt
cd ../..
uv pip install -e .
```

### Computing Token-level Unsupervised Metrics

```bash
# Available langs for Aya23 in WMT24ESA: en-cs, en-ja, en-zh, en-hi, cs-uk, en-ru
python scripts/compute_unsupervised_metrics.py \
    --model_id CohereLabs/aya-23-35B \
    --dataset_name wmt24esa \
    --langs en-cs \
    --output_dir outputs/metrics/wmt24esa
```
