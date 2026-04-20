# AIAA4051 Individual Project Report

- Student ID: 50012314
- Name: Shengyi Chung
- Course: AIAA4051
- Project: PEFT-based LoRA Fine-tuning on Llama-2-7B

## 1. Task Overview

This project fine-tunes Llama-2-7B on a QA dataset using PEFT LoRA.
The objective is to improve answer accuracy while keeping trainable parameters and storage cost low.

## 2. Dataset and Split

### 2.1 Dataset

- Source file: `dataset.json`
- Task format: question -> short answer
- Prompt style:
  - System prompt: "Answer the question with a short and precise phrase."
  - User prompt: `Question: ...`
  - Model completion: `Answer: ...`

### 2.2 Data Split

- Split method: random split with fixed seed
- Validation ratio: 0.1
- Seed: 42
- Resulting size:
  - Train: 4500
  - Validation: 500

## 3. Model Setup

### 3.1 Base Model

- Base model: `meta-llama/Llama-2-7b-hf`
- Local base model folder: `./model`

### 3.2 PEFT LoRA Setup

- Framework: Hugging Face PEFT
- Task type: causal language modeling
- Adapter target modules (best config): `q_proj, k_proj, v_proj, o_proj`
- Saved model type: LoRA adapter only (`adapter_model.safetensors`, `adapter_config.json`)

### 3.3 Precision / Device

- CUDA GPU training
- Mixed precision enabled by Trainer settings (`fp16/bf16` auto-selected by environment)

## 4. Fine-tuning Procedure

1. Load and split dataset.
2. Load tokenizer and base model.
3. Apply LoRA adapters to target modules.
4. Train with Hugging Face Trainer.
5. Evaluate on validation set by generation-based accuracy.
6. Save adapter files only.

## 5. Hyperparameters Summary

### 5.1 Best Configuration (current best)

| Item | Value |
|---|---|
| Experiment tag | `lr1e4-r32-e3-d01-bs4-ga4_maskq-false-padright` |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.1 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 3 |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Max length | 256 |
| Padding side | right |
| Mask question | false |
| Early stopping patience | 3 |

### 5.2 Other Tested Configurations (example)

| Tag | Key Difference | Val Accuracy |
|---|---|---|
| `lr1e4-r32-e3-d01-bs4-ga4_maskq-false-padright` | right pad + qkvo + 3 epochs | 0.578 |
| `cmp-padright-qkvo` | right pad + qkvo + 2 epochs | 0.580 |
| `lr1e4-r16-e2-d01-bs4-ga4_maskq-false-padright-tm_qv` | right pad + qv | 0.578 |
| `lr1e4-r48-e2-d01-bs4-ga4_maskq-false-padright` | higher LoRA capacity (r48/a96) | 0.570 |

## 6. Training and Validation Curves

> Insert your figures in this section.

### 6.1 Training Loss Curve

- Figure 1: training loss vs steps/epochs

![Training Loss Curve](training%20loss.png)

- Observation:
  - Loss decreases steadily in early and middle training.
  - Later stage becomes flatter.

### 6.2 Validation Loss Curve

- Figure 2: validation loss vs epochs

![Validation Loss Curve](validation%20loss.png)

- Observation:
  - Validation loss improves from epoch 1 to epoch 2 in most stable runs.
  - In some runs, lower eval loss does not always mean highest final generation accuracy.

### 6.3 How to Export Curves (optional notes)

- Use W&B run charts or Trainer log history.
- Export as PNG and insert here.

## 7. Accuracy Results

### 7.1 Validation Accuracy

- Validation accuracy of selected model (`lr1e4-r32-e3-d01-bs4-ga4_maskq-false-padright`): **0.578**
- Best observed validation accuracy among all runs: **0.580**

### 7.2 Training-set Accuracy

- Training accuracy: **0.6313** (2841/4500)
- Validation accuracy (selected model): 0.578 (289/500)
- Test accuracy (selected model, `individual_project_test.py`): 56.00% (280/500)

## 8. Analysis and Discussion

### 8.1 What Worked

- Right padding generally outperformed left padding in this project.
- `qkvo` target modules performed better than `qv` in top runs.
- LoRA with rank 32 and alpha 64 gave a stable balance.
- The selected submission model (`r32/a64`, 3 epochs) achieved strong test-set performance (56.00%).

### 8.2 Observed Issues

- Environment may print bitsandbytes CUDA library warning (`libnvJitLink.so.13`) but training can still run.
- Selecting best model by `eval_loss` may not always match generation-accuracy objective.

### 8.3 Potential Improvements

- Run controlled learning-rate sweep around 1e-4 (for example, 8e-5 to 1.5e-4).
- Add explicit train-set generation-accuracy reporting each run.
- Align model selection metric with generation accuracy if required.

## 9. Reproducibility

### 9.1 Code Files

- `train.py`
- `model.py`
- `eval.py`
- `individual_project_test.py`

### 9.2 Run Command (example)

```bash
python train.py
```

### 9.3 Test Command (example)

```bash
ADAPTER_PATH="./model_lr1e4-r32-e3-d01-bs4-ga4_maskq-false-padright" python individual_project_test.py
```

## 10. Submission Checklist

- [ ] `50012314_ShengyiChung.pdf` - **TO BE GENERATED**
- [x] `50012314_ShengyiChung_code/` with source code - **COMPLETE**
- [x] `50012314_ShengyiChung_model/` with adapter files only - **COMPLETE**
- [x] Report includes:
  - [x] Training process details
  - [x] Hyperparameter summary
  - [x] Training loss curve - **INSERTED**
  - [x] Validation loss curve - **INSERTED**
  - [ ] Training-set accuracy - **PENDING (needs training run)**
  - [x] Validation-set accuracy

---

Appendix: You can convert this markdown to PDF after replacing all TODO parts and adding figures.
