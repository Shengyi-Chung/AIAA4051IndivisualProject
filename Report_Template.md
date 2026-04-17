# AIAA4051 Individual Project Report

**Student ID:** 50011190  
**Student Name:** Yazheng Liu  
**Date:** [Submission Date]

---

## 1. Training Process Description

### 1.1 Dataset Splitting
- **Dataset Name:** [e.g., SQuAD v2.0]
- **Dataset Size:** [Total samples]
- **Training Set:** [Number of samples] (percentage)
- **Validation Set:** [Number of samples] (percentage)
- **Data Preprocessing:**
  - Tokenization method: [e.g., AutoTokenizer from meta-llama/Llama-2-7b]
  - Max sequence length: [e.g., 512 tokens]
  - Padding strategy: [e.g., left padding, pad_token = eos_token]
  - Input format: [Describe prompt template used for training]

### 1.2 Model Setup
- **Base Model:** meta-llama/Llama-2-7b
- **Model Architecture:** Transformer-based LLM (7B parameters)
- **Quantization:** [e.g., None / 4-bit / 8-bit]
- **Device Configuration:** [e.g., CUDA GPU - RTX 3050 Ti 4GB / Server GPU]
- **Framework:** Hugging Face Transformers + PEFT

### 1.3 Fine-tuning Procedure
- **Method:** PEFT-based LoRA (Low-Rank Adaptation)
- **LoRA Application Points:** 
  - Target modules: [List applied modules, e.g., q_proj, v_proj]
  - Applied to all layers: [Yes/No]
- **Training Steps:**
  1. Load pretrained Llama-2-7b model with device_map="auto"
  2. Apply LoRA configuration via PEFT
  3. Freeze base model weights, only train LoRA adapter
  4. Load and preprocess QA dataset
  5. Initialize Trainer with early stopping callback
  6. Run training loop with specified hyperparameters
  7. Save only LoRA adapter files (model.save_pretrained)
- **Loss Function:** CrossEntropyLoss (default for causal LM)

### 1.4 Evaluation Steps
- **Evaluation Metric:** Accuracy (substring matching, case-insensitive)
- **Evaluation Strategy:** Evaluate at end of each epoch
- **Inference Configuration:**
  - Decoding method: Greedy (do_sample=False)
  - Max new tokens: 16
  - Temperature: N/A (greedy decoding)
- **Accuracy Calculation:** 
  - Generate predictions on validation set
  - Compare normalized (lowercase) generated text against ground truth answer
  - Accuracy = (correct predictions) / (total predictions)

---

## 2. Training Hyperparameters

### 2.1 LoRA-Related Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **rank (r)** | [e.g., 16] | Dimension of LoRA weight matrices |
| **lora_alpha** | [e.g., 32] | LoRA scaling factor (alpha/r determines effective scaling) |
| **lora_dropout** | [e.g., 0.05] | Dropout applied to LoRA layers (regularization) |
| **target_modules** | [List modules] | Which layers to apply LoRA (q_proj, v_proj, etc.) |
| **bias** | [e.g., "none"] | Whether to add trainable bias |

### 2.2 Training Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Learning Rate** | [e.g., 2e-4] | Peak learning rate for optimizer |
| **Learning Rate Scheduler** | [e.g., linear] | LR decay strategy |
| **Optimizer** | [e.g., paged_adamw_32bit] | Optimization algorithm |
| **Batch Size (Train)** | [e.g., 4] | Samples per training step |
| **Batch Size (Eval)** | [e.g., 8] | Samples per evaluation step |
| **Num Epochs** | [e.g., 3] | Number of complete training passes |
| **Max Steps** | [e.g., -1] | Max training steps (-1 = no limit) |
| **Warmup Steps** | [e.g., 100] | Linear warmup for learning rate |
| **Weight Decay** | [e.g., 0.01] | L2 regularization strength |

### 2.3 Additional Training Settings
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Mixed Precision** | [e.g., fp16] | Precision for training (fp16/bf16/no) |
| **Gradient Accumulation Steps** | [e.g., 1] | Steps before backward pass |
| **Max Grad Norm** | [e.g., 1.0] | Gradient clipping threshold |
| **Early Stopping** | Enabled | Stop if no improvement after N evaluations |
| **Early Stopping Patience** | [e.g., 2] | Epochs without improvement before stopping |
| **Save Strategy** | epoch | Save checkpoint after each epoch |
| **Evaluation Strategy** | epoch | Evaluate after each epoch |

---

## 3. Training Results

### 3.1 Training Loss Curve
[Insert plot: Training loss vs. epoch]
- **Description:** [Describe the trend, e.g., "Loss steadily decreased from X to Y, indicating effective learning"]
- **Final Training Loss:** [e.g., 0.45]

### 3.2 Validation Loss Curve
[Insert plot: Validation loss vs. epoch]
- **Description:** [Describe the trend, convergence behavior, any overfitting signs]
- **Final Validation Loss:** [e.g., 0.52]
- **Best Validation Loss:** [e.g., 0.48 at epoch 2]

### 3.3 Accuracy Results
| Dataset | Accuracy | Notes |
|---------|----------|-------|
| **Training Set** | [e.g., 75.3%] | Accuracy on training data |
| **Validation Set** | [e.g., 72.1%] | Accuracy on validation data |
| **Gap** | [e.g., 3.2%] | Difference indicating generalization |

**Analysis:** [Discuss whether the model generalizes well, any signs of overfitting, etc.]

---

## 4. Implementation Details

### 4.1 Key Code Components
- **Model Loading:** Used `AutoModelForCausalLM.from_pretrained()` with device_map="auto"
- **LoRA Setup:** Applied via PEFT's `LoraConfig` and `get_peft_model()`
- **Data Format:** [Describe how Q&A pairs are formatted into prompts]
- **Adapter Saving:** `model.save_pretrained()` saves only LoRA weights (~[size]MB)

### 4.2 Hardware & Environment
- **GPU Used:** [e.g., RTX 3050 Ti / Rented Server GPU]
- **GPU Memory:** [e.g., 4GB / 24GB]
- **Training Time:** [e.g., 2 hours for 3 epochs]
- **Framework Versions:**
  - PyTorch: 2.5.1+cu121
  - Transformers: 5.5.3
  - PEFT: 0.18.1

### 4.3 Key Challenges & Solutions
1. **Challenge:** [e.g., GPU memory insufficiency]
   **Solution:** [e.g., Used LoRA for parameter-efficient fine-tuning; rented server GPU]

2. **Challenge:** [e.g., Tokenizer padding consistency]
   **Solution:** [e.g., Used left padding in both training and evaluation]

---

## 5. Submission Artifacts

- **Adapter Files Location:** `50011190_Yazheng Liu_model/`
- **Adapter File Size:** [e.g., ~45MB]
- **Source Code:** `50011190_Yazheng Liu_code/`
- **Files Included:** train.py, model.py, eval.py, requirements.txt

---

## 6. Conclusion

[Summarize key findings: model performance, effectiveness of LoRA, recommendations for improvement, etc.]

---

**Appendix:** [Optional - additional plots, error analysis, or detailed hyperparameter search results]
