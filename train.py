import argparse
import json
import os

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from model import (
    QADataset,
    QAExample,
    load_dataset,
    train_val_split,
    load_model_and_tokenizer,
    setup_lora,
    ensure_gpu_ready,
)
from eval import run_accuracy_eval


def save_split(train_data: list, val_data: list, output_dir: str):
    """Save train and validation splits to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    train_json = [
        {"question": ex.question, "correct_answer": ex.answer} for ex in train_data
    ]
    val_json = [
        {"question": ex.question, "correct_answer": ex.answer} for ex in val_data
    ]

    with open(os.path.join(output_dir, "train_split.json"), "w", encoding="utf-8") as f:
        json.dump(train_json, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "val_split.json"), "w", encoding="utf-8") as f:
        json.dump(val_json, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for QA dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--allow_cpu", action="store_true", default=False)
    parser.add_argument("--dataset_path", type=str, default="dataset.json")
    parser.add_argument("--output_dir", type=str, default="StudentID_Name_model_lr1e4_e2_d01")
    parser.add_argument("--report_dir", type=str, default="outputs_lr1e4_e2_d01")
    parser.add_argument("--split_dir", type=str, default="data_splits")

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--use_wandb", type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, nargs='?', const=True, metavar='BOOL')
    parser.add_argument("--wandb_project", type=str, default="AIAA4051-Llama2-LoRA")
    parser.add_argument("--wandb_run_name", type=str, default="exp-lr1e4-r16-e2-d01")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated list of LoRA target modules",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_gpu = not args.allow_cpu
    run_name = args.wandb_run_name.strip() if args.wandb_run_name.strip() else None
    use_wandb = args.use_wandb and args.wandb_mode != "disabled"

    if use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_MODE"] = args.wandb_mode
        if run_name:
            os.environ["WANDB_NAME"] = run_name

    ensure_gpu_ready(require_gpu=require_gpu)

    os.makedirs(args.report_dir, exist_ok=True)

    # Load and split dataset
    print("Loading dataset...")
    data = load_dataset(args.dataset_path)
    train_data, val_data = train_val_split(data, args.val_ratio, args.seed)
    save_split(train_data, val_data, args.split_dir)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.model_dir,
        require_gpu=require_gpu,
    )

    # Setup LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    print(f"Setting up LoRA with rank={args.rank}, alpha={args.alpha}, dropout={args.dropout}")
    model = setup_lora(model, args.rank, args.alpha, args.dropout, target_modules)

    # Create datasets
    train_ds = QADataset(train_data, tokenizer, args.max_length)
    val_ds = QADataset(val_data, tokenizer, args.max_length)

    # Training arguments
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.report_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        fp16=use_cuda and not use_bf16,
        bf16=use_bf16,
        report_to="wandb" if use_wandb else "none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapter weights only
    print(f"Saving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)

    # Evaluate on validation set
    print("Evaluating on validation set...")
    accuracy, eval_logs = run_accuracy_eval(model, tokenizer, val_data)

    # Collect metrics
    metrics = {
        "num_train": len(train_data),
        "num_val": len(val_data),
        "accuracy": accuracy,
        "lora_config": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
        },
    }

    if use_wandb:
        try:
            import wandb

            wandb.log({
                "eval/accuracy": accuracy,
                "data/num_train": len(train_data),
                "data/num_val": len(val_data),
                "hparams/rank": args.rank,
                "hparams/alpha": args.alpha,
                "hparams/dropout": args.dropout,
                "hparams/epochs": args.epochs,
                "hparams/learning_rate": args.learning_rate,
                "hparams/batch_size": args.batch_size,
                "hparams/grad_accum": args.grad_accum,
            })
            wandb.finish()
        except Exception as e:
            print(f"Warning: W&B logging failed: {e}")

    # Save results
    with open(os.path.join(args.report_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.report_dir, "eval_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(eval_logs, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("FINAL METRICS")
    print("="*50)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
