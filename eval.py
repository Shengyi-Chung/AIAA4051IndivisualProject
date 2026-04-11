import json
from typing import Dict, List, Tuple

import torch

from model import QAExample, build_prompt, normalize_text


def run_accuracy_eval(
    model, tokenizer, val_examples: List[QAExample], max_new_tokens: int = 16
) -> Tuple[float, List[Dict]]:
    """
    Evaluate model on validation set.
    Accuracy = # of correct answers / total questions.
    A prediction is correct if it contains the normalized correct_answer.
    """
    model.eval()
    correct = 0
    logs = []

    for ex in val_examples:
        prompt = build_prompt(ex.question)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        is_correct = normalize_text(ex.answer) in normalize_text(generated_text)
        if is_correct:
            correct += 1

        logs.append(
            {
                "question": ex.question,
                "correct_answer": ex.answer,
                "model_output": generated_text,
                "matched": is_correct,
            }
        )

    accuracy = correct / len(val_examples) if len(val_examples) > 0 else 0.0
    return accuracy, logs
