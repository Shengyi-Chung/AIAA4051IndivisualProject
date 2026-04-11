import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


SYSTEM_PROMPT = "Answer the question with a short and precise phrase."
DEFAULT_MODEL_DIR = "model"


@dataclass
class QAExample:
    question: str
    answer: str


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_prompt(question: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def load_dataset(path: str) -> List[QAExample]:
    """Load QA dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for row in raw:
        q = row["question"].strip()
        a = row["correct_answer"].strip()
        examples.append(QAExample(question=q, answer=a))
    return examples


def train_val_split(data: List[QAExample], val_ratio: float, seed: int):
    """Split data into train and validation sets."""
    random.seed(seed)
    shuffled = data[:]
    random.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio))
    val_set = shuffled[:val_size]
    train_set = shuffled[val_size:]
    return train_set, val_set


def _has_model_artifacts(local_dir: str) -> bool:
    """Check whether a local directory already contains downloaded model files."""
    if not os.path.isdir(local_dir):
        return False

    required_markers = [
        "config.json",
        "configuration.json",
        "model.safetensors.index.json",
        "pytorch_model.bin",
    ]
    files = set(os.listdir(local_dir))
    if any(marker in files for marker in required_markers):
        return True

    return any(
        name.endswith((".safetensors", ".bin"))
        for name in files
        if os.path.isfile(os.path.join(local_dir, name))
    )


def ensure_model_downloaded(model_name: str, local_dir: str = DEFAULT_MODEL_DIR) -> str:
    """Ensure a model exists locally; download it if the local folder is missing or empty."""
    if _has_model_artifacts(local_dir):
        print(f"Found local model directory: {local_dir}")
        return local_dir

    os.makedirs(local_dir, exist_ok=True)
    print(f"Local model not found. Downloading {model_name} to {local_dir}...")

    command = [
        sys.executable,
        "-m",
        "modelscope",
        "download",
        "--model",
        model_name,
        "--local_dir",
        local_dir,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Model download failed for {model_name}.")

    if not _has_model_artifacts(local_dir):
        raise RuntimeError(f"Model download completed but no artifacts were found in {local_dir}.")

    return local_dir


def ensure_gpu_ready(require_gpu: bool = True) -> None:
    """Validate CUDA runtime before training when GPU is required."""
    cuda_build = torch.version.cuda
    cuda_available = torch.cuda.is_available()

    if require_gpu and (not cuda_available or cuda_build is None):
        raise RuntimeError(
            "GPU training is required, but CUDA is not ready. "
            f"torch={torch.__version__}, torch.version.cuda={cuda_build}, "
            f"torch.cuda.is_available()={cuda_available}. "
            "Please install a CUDA-enabled PyTorch build and NVIDIA driver."
        )

    if cuda_available:
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")


class QADataset(Dataset):
    """Torch dataset for QA pairs."""

    def __init__(self, examples: List[QAExample], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = f"{build_prompt(ex.question)} {ex.answer}{self.tokenizer.eos_token}"
        item = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in item.items()}


def load_model_and_tokenizer(
    model_name: str,
    local_dir: str = DEFAULT_MODEL_DIR,
    require_gpu: bool = True,
):
    """Load base model and tokenizer from a local directory or remote model name."""
    ensure_gpu_ready(require_gpu=require_gpu)
    model_path = ensure_model_downloaded(model_name, local_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    return model, tokenizer


def setup_lora(model, rank: int, alpha: int, dropout: float, target_modules: List[str]):
    """Apply LoRA to model using PEFT."""
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model
