import os
import tqdm
import json
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets




from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    get_linear_schedule_with_warmup,
)
import argparse

from data.dataloader import Dataloader

def parse_args():
    p = argparse.ArgumentParser(description="vit project")

    p.add_argument("--ai-dir", type=str, required=True)
    p.add_argument("--real-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--split", nargs=3, type=float, metavar=("TRAIN","VAL","TEST"))
    p.add_argument("--pin-memory", action="store_true")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--fp16", action="store_true", help="use mixed precision")
        # model/checkpointing
    p.add_argument("--model_id", type=str, default="google/vit-base-patch16-224-in21k")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--outdir", type=str, default="checkpoints/vit_face_clf")
    # misc
    p.add_argument("--seed", type=int, default=16)

    return p.parse_args()

def evaluate(model, loader, device, criterion=None):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if criterion is None:
                out = model(**batch)                 
                loss = out.loss
                logits = out.logits
            else:
                out = model(pixel_values=batch["pixel_values"]) 
                logits = out.logits
                loss = criterion(logits, batch["labels"])

            loss_sum += loss.item()
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].numel()
    return loss_sum / max(1, len(loader)), correct / max(1, total)

def main():
    args = parse_args()

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    os.makedirs(args.outdir, exist_ok=True) 
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.outdir, f"args_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = Dataloader(
        hf_dataset="Hemg/AI-Generated-vs-Real-Images-Datasets",
        split=args.split,
        seed = args.seed,
        model_id = args.model_id,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    ).create_loader()

    id2label = {0: "real", 1: "ai"}
    label2id = {"real": 0, "ai": 1}
    model = ViTForImageClassification.from_pretrained(
        args.model_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_acc = 0.0
    best_dir = os.path.join(args.outdir, "best")
    os.makedirs(best_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, device)
        train_loss = running / max(1, len(train_loader))
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "metrics.json"), "w") as f:
                json.dump({"epoch": epoch, "val_acc": val_acc}, f, indent=2)

    # Final test
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test: loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Save final run artifacts too
    model.save_pretrained(args.outdir)
    processor.save_pretrained(args.outdir)
    with open(os.path.join(args.outdir, "final_metrics.json"), "w") as f:
        json.dump({"best_val_acc": best_acc, "test_acc": test_acc}, f, indent=2)

    print(f"Artifacts saved to: {args.outdir} (best checkpoint in {best_dir})")



if __name__ == "__main__":
    main()