# Federated Learning with CineMA

Federated fine-tuning of [CineMA](https://github.com/mathpluscode/CineMA) — a vision foundation model for cine cardiac MRI — using [Flower](https://flower.ai).

Supports three clinical tasks across centralized and federated paradigms:

| Task            | Description                                              |
| --------------- | -------------------------------------------------------- |
| Classification  | Cardiac disease detection (multi-class and binary)       |
| Regression      | Continuous clinical variables (e.g. LVEF, volumes)       |
| Segmentation    | Ventricle and myocardium segmentation                    |

## Prerequisites

- **Git** (only needed to pull the CineMA submodule)
- **Conda** (Miniconda or Anaconda)
- **NVIDIA GPU** with CUDA 11.8+ — Flower's Simulation Engine does not support CPU-only execution

## Setup

### 1. Get the code

If you received the project as a ZIP, extract it and open a terminal inside the folder.

### 2. Pull the CineMA submodule

The CineMA model lives at `models/cinema/`. Pull it with:

```bash
git submodule update --init --recursive
```

If the command above complains that the submodule is not registered, add it explicitly:

```bash
git submodule add https://github.com/mathpluscode/CineMA.git models/cinema
```

If you don't have git configured at all, download the repository manually from [github.com/mathpluscode/CineMA](https://github.com/mathpluscode/CineMA) and place the contents inside `models/cinema/`.

### 3. Create the conda environment

```bash
conda env create -f envs/environment.yml 
conda activate flwr-cinema
```

Then install cinema from the root of the proyect

```bash
pip install ./models/cinema
```

This installs all Python dependencies (PyTorch, MONAI, Flower, HuggingFace Hub, etc.) and registers both the main package in editable mode.

### 4. Verify PyTorch

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False` or you get CUDA-related errors, the PyTorch version installed from `pyproject.toml` does not match your system. Uninstall it and reinstall the correct build for your CUDA version:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # example for CUDA 12.4
```

Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the exact command matching your setup.

## Data

The combined ACDC + MnMs dataset is available on Hugging Face:

```
https://huggingface.co/datasets/RodoV1204/acdc_mnms_combined/resolve/main/acdc_mnms_combined.zip?download=true
```

Download and extract it anywhere on your machine. Then edit the `data.combined.dir` path in every config file you intend to use:

```
configs/centralized/cinema_classification_centralized.yaml  -> data.combined.dir
configs/centralized/cinema_regression_centralized.yaml      -> data.combined.dir
configs/centralized/cinema_segmentation_centralized.yaml    -> data.combined.dir
configs/federated/cinema_classification_fed.yaml            -> data.combined.dir
configs/federated/cinema_regression_fed.yaml                -> data.combined.dir
configs/federated/cinema_segmentation_fed.yaml              -> data.combined.dir
```

The extracted directory must contain `train_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`, and the preprocessed volumes.

## Centralized training

Centralized training runs on a single GPU with the full combined dataset.

```bash
centralized_train --config cinema_classification_centralized
```

Available configs (omit the `.yaml` extension):

| Flag                                      | Task           |
| ----------------------------------------- | -------------- |
| `cinema_classification_centralized`       | Classification |
| `cinema_regression_centralized`           | Regression     |
| `cinema_segmentation_centralized`         | Segmentation   |

Checkpoints and logs are saved under `outputs/cinema_<task>/centralized/<dataset>/<target_column>/lora_<bool>/seed<N>/<date>/<time>/`.

## Centralized evaluation

Evaluate a trained centralized model on the test set:

```bash
centralized_eval --checkpoint-dir outputs/cinema_classification/centralized/acdc_mnms_combined/binary_label/lora_True/seed0/2026-05-20/11-00-00
```

Key flags:

| Flag                  | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `--checkpoint-dir`    | Path to the checkpoint directory (required)                  |
| `--checkpoint-path`   | Path to a specific `.pt` file (alternative to --checkpoint-dir) |
| `--checkpoint-epoch`  | Evaluate a specific epoch instead of the best one            |
| `--mc-dropout`        | Enable Monte Carlo Dropout uncertainty estimation            |
| `--mc-dropout-passes` | Number of MC forward passes (default: 30)                    |
| `--output-dir`        | Custom output directory for results                          |
| `--output-file-name`  | Output JSON filename (default: evaluation_results.json)      |

Results are saved as `evaluation_results.json` next to the checkpoint.

## Federated training

Federated training uses Flower's Simulation Engine, which spawns multiple clients on the same machine with fractional GPU allocation.

```bash
flwr run . --stream \
  --federation-config="num-supernodes=5 client-resources-num-cpus=2 client-resources-num-gpus=0.4 init-args-num-gpus=2" \
  --run-config "hydra_config_name='cinema_classification_fed' hydra_overrides='...'"
```

Checkpoints and logs are saved under `outputs/cinema_<task>/federated/<dataset>/<target_column>/lora_<bool>/seed<N>/<date>/<time>/`.

### Federation config (`--federation-config`)

Controls the simulation resources:

| Parameter                       | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- |
| `num-supernodes`                | Number of federated clients. Must match `federated.n_clients` in the config |
| `client-resources-num-cpus`     | CPU cores per client                                          |
| `client-resources-num-gpus`     | Fractional GPU per client (e.g. 0.4)                          |
| `init-args-num-gpus`            | Total GPUs available to the simulation                        |

### Run config (`--run-config`)

Two key overrides. `hydra_overrides` is an inline alternative to editing the YAML config file — useful for sweep scripts or quick experiments where changing the file would be impractical.

| Parameter              | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| `hydra_config_name`    | Which Hydra config to load (one of the federated YAML files)             |
| `hydra_overrides`      | Hydra-style overrides for any config key                                 |

Available `hydra_config_name` values:

| Config                           | Task           |
| -------------------------------- | -------------- |
| `cinema_classification_fed`      | Classification |
| `cinema_regression_fed`          | Regression     |
| `cinema_segmentation_fed`        | Segmentation   |

### Changing the aggregation strategy

Override `federated.aggregation_method.name` in `hydra_overrides`. Supported strategies and their extra config:

| Strategy          | Config overrides                                                      |
| ----------------- | --------------------------------------------------------------------- |
| `fedavg`          | _(none)_                                                              |
| `fedce`           | `+federated.aggregation_method.config.combination_method=sum`         |
| `fedprox`         | `+federated.aggregation_method.config.proximal_mu=0.1`                |
| `fedratemylora`   | `+federated.aggregation_method.config.lambda_init=0.2 +federated.aggregation_method.config.lambda_decay=0.95` |

Example: FedProx with proximal regularization on a regression task:

```bash
flwr run . --stream \
  --federation-config="num-supernodes=5 client-resources-num-cpus=2 client-resources-num-gpus=0.4 init-args-num-gpus=2" \
  --run-config "hydra_config_name='cinema_regression_fed' hydra_overrides='federated.aggregation_method.name=fedprox +federated.aggregation_method.config.proximal_mu=0.1'"
```

## Federated evaluation

Evaluate a trained federated model on the test set:

```bash
federated_eval --checkpoint-dir outputs/cinema_<task>/federated/<dataset>/<target_column>/lora_<bool>/seed<N>/<date>/<time>/federated_server/checkpoints
```

Key flags:

| Flag                  | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `--checkpoint-dir`    | Path to the `checkpoints/` directory (required)              |
| `--checkpoint-path`   | Path to a specific `.pt` file (alternative to --checkpoint-dir) |
| `--mc-dropout`        | Enable Monte Carlo Dropout uncertainty estimation            |
| `--mc-dropout-passes` | Number of MC forward passes (default: 30)                    |
| `--output-dir`        | Custom output directory for results                          |
| `--output-file-name`  | Output JSON filename (default: evaluation_results.json)      |

The evaluator automatically selects the best round by the primary metric configured in `train.early_stopping.metric`.

## Config reference

Key configuration sections shared across centralized and federated YAML files:

| Section                  | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `data.combined.dir`      | Path to the extracted dataset                          |
| `data.class_column`      | Target column name (classification)                    |
| `data.regression_column` | Target column name (regression)                        |
| `federated.n_rounds`     | Number of communication rounds                         |
| `federated.local_epochs` | Local epochs per round per client                      |
| `federated.n_clients`    | Number of expected clients                             |
| `federated.client_ids`   | List of client identifiers                             |
| `lora.rank`              | LoRA rank dimension                                    |
| `lora.alpha`             | LoRA scaling factor                                    |
| `lora.target_modules`    | Attention modules to adapt (`q`, `kv`, `proj`)         |
| `train.use_lora`         | Enable/disable LoRA fine-tuning                        |
| `train.batch_size`       | Effective batch size (after gradient accumulation)     |
| `train.lr`               | Learning rate                                          |
| `train.n_epochs`         | Maximum training epochs                                |
| `train.early_stopping`   | Metric, mode, patience, and min_delta for stopping     |
| `model.name`             | Architecture: `convvit` or `resnet`                    |
