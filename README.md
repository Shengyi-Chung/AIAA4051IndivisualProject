
# AIAA 4051: Introduction to NLP - Individual Project

## Project Overview
This project involves fine-tuning the **Llama 2-7B** model using **LoRA (Low-Rank Adaptation)**. The objective is to utilize the `PEFT` library to adapt the model to a specific dataset and evaluate its performance based on response accuracy.

## Task Description
* **Model**: Llama 2-7B.
* **Method**: Parameter-Efficient Fine-Tuning (PEFT) using LoRA.
* **Dataset**: The provided dataset must be split into a training set and a validation set.
* **Metric**: Performance is measured by **Accuracy**, defined as the number of correct answers appearing in the generated responses divided by the total number of questions.

## Requirements
### LoRA Implementation
* Implementation must use the `PEFT` library.
* Key hyperparameters to configure and report include:
    * Rank ($r$).
    * Alpha.
    * Dropout.
    * Epochs.

### Submission Structure
All files must be organized into a single folder named `StudentID_Name` (e.g., `50011190_Yazheng Liu`). The directory structure should be:
* `StudentID_Name.pdf`: A detailed report covering the training process, hyperparameter summary, loss curves (training and validation), and accuracy results.
* `StudentID_Name_code/`: A folder containing the source code for the PEFT-based LoRA fine-tuning.
* `StudentID_Name_model/`: A folder containing only the saved LoRA-related adapter files (do not include original model weights).

## Resources
* **Model Weights**: Llama 2-7B is available via ModelScope.
* **Training Platform**: Diandong platform instructions are provided in the project documentation.
* **Documentation**: Refer to the official [PEFT documentation](https://huggingface.co/docs/peft/index) and [LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora).

## Important Dates & Evaluation
* **Deadline**: April 15, 2026, 23:55 PM.
* **Grading**: Grades are based on model accuracy on a hidden test set, as well as the clarity and quality of the submitted report.
* **Note**: Understanding the code is crucial as it may be relevant to the final exam.

## Quick Start (Generated Starter)
The following starter files are included in this workspace:
* `requirements.txt`: Required Python packages.
* `model.py`: Core data structures (QAExample, QADataset), data loading, model & tokenizer initialization, LoRA setup.
* `eval.py`: Validation logic (run_accuracy_eval) for computing accuracy on validation set.
* `train.py`: Main training pipeline (split, train, save LoRA adapter, evaluate, export metrics).
* `run_train.ps1`: PowerShell helper command to start training with customizable hyperparameters.

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run training
```powershell
./run_train.ps1 -ModelName "meta-llama/Llama-2-7b-hf" -Epochs 3 -Rank 16 -Alpha 32 -Dropout 0.05
```

You can also run the Python script directly:
```bash
python train.py --model_name meta-llama/Llama-2-7b-hf --epochs 3 --rank 16 --alpha 32 --dropout 0.05
```

### 2.1) Optional: Enable W&B logging
```bash
pip install wandb
wandb login
```

Run training with W&B enabled:
```bash
python train.py --model_name meta-llama/Llama-2-7b-hf --epochs 3 --rank 16 --alpha 32 --dropout 0.05 --use_wandb --wandb_project AIAA4051-Llama2-LoRA --wandb_run_name exp-01
```

Run in offline mode (then sync later):
```bash
python train.py --model_name meta-llama/Llama-2-7b-hf --epochs 3 --rank 16 --alpha 32 --dropout 0.05 --use_wandb --wandb_mode offline --wandb_project AIAA4051-Llama2-LoRA --wandb_run_name exp-offline-01
```

### 3) Output folders
* `StudentID_Name_model/`: LoRA adapter files only (submission model folder).
* `outputs/metrics.json`: Summary metrics and hyperparameters.
* `outputs/eval_predictions.json`: Validation predictions and match results.
* `data_splits/train_split.json` and `data_splits/val_split.json`: Saved split files.

### 4) Accuracy definition used in the script
For each validation question, the generated response is marked correct if it contains `correct_answer` (case-insensitive, normalized spaces).