"""Core SUL training with a bias-free semantic-unit output projection.

Model implementation: casu_model.py. Dataset interface: casu_data.py.
This version requires a 5,000-image COCO validation split.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import BertTokenizer

import sul_data as data
from sul_model import SlotsModel, create_optimizer, precision_context


def get_argument_parser():
    """Parameters used by the core model and its training/data paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    paths = parser.add_argument_group("Data and output")
    paths.add_argument("--dataset", choices=("f30k", "coco"), default="f30k")
    paths.add_argument("--data_path", type=Path, required=True,
                       help="Flickr30K split root; contains f30k/*_caps.txt and *_ids.txt")
    paths.add_argument("--f30k_img_path", default="data/flickr30k-images")
    paths.add_argument("--coco_img_path", default="data/coco")
    paths.add_argument("--coco_annotations_path", default="data/coco/annotations_karpathy")
    paths.add_argument("--local_bert_path", required=True,
                       help="Local BERT-base-uncased model/tokenizer directory")
    paths.add_argument("--local_vit_path", required=True,
                       help="Local ViT-B/16-224-in21k model directory")
    paths.add_argument("--logger_name", type=Path, default=Path("results/sul"))

    model = parser.add_argument_group("Core model")
    model.add_argument("--embed_size", type=int, default=512)
    model.add_argument("--num_slots", type=int, default=7,
                       help="Number K of local semantic units; the global vector is additional")
    model.add_argument("--slot_iters", type=int, default=2)
    model.add_argument("--slot_readout_layer", type=int, choices=range(1, 13), default=8)
    model.add_argument("--img_res", type=int, choices=(224, 384), default=224)
    model.add_argument("--vit_type", choices=("vit", "vit_224", "vit_384"), default="vit")
    model.add_argument("--caption_chunk_size", type=int, default=32)
    model.add_argument("--shard_size", type=int, default=256)
    model.add_argument("--max_word", type=int, default=90)
    model.add_argument("--precision", choices=("fp32", "tf32", "bf16"), default="bf16")
    model.add_argument("--attn_impl", choices=("eager", "sdpa"), default=None)

    train = parser.add_argument_group("Training")
    train.add_argument("--num_epochs", type=int, default=30)
    train.add_argument("--batch_size", type=int, default=128)
    train.add_argument("--learning_rate", type=float, default=2e-4)
    train.add_argument("--margin", type=float, default=0.2)
    train.add_argument("--grad_clip", type=float, default=2.0)
    train.add_argument("--lr_schedules", type=int, nargs="+", default=[9, 15, 20],
                       help="Zero-based milestones for decay at epochs 10, 16, and 21")
    train.add_argument("--decay_rate", type=float, default=0.3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--workers", type=int, default=16)
    train.add_argument("--persistent_workers", type=int, choices=(0, 1), default=1)
    train.add_argument("--size_augment", type=int, choices=(0, 1), default=1)
    train.add_argument("--gpu-id", type=int, default=0)
    train.add_argument("--log_step", type=int, default=200)
    train.add_argument("--early_stop", type=int, choices=(0, 1), default=1)

    # Shared affinity projection and image-text retrieval supervision.
    parser.set_defaults(
        slot_shared_qk=1,
        output_projection_bias=False,
        coco_validation_images=5000,
        implementation_version="bias_free_output",
        max_violation=False,
        vse_mean_warmup_epochs=1,
        embedding_warmup_epochs=0,
    )
    return parser


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def save_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def save_checkpoint(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class OfficialSchedule:
    """Apply the original multiplicative schedule before each zero-based epoch."""

    def __init__(self, optimizer, milestones, factor):
        self.optimizer = optimizer
        self.milestones = list(milestones)
        self.factor = factor
        self.last_epoch = -1
        self.applied = []

    def step(self, epoch):
        if epoch <= self.last_epoch:
            raise ValueError("Learning-rate epoch repeated")
        if epoch in self.milestones:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.factor
            self.applied.append(epoch)
        self.last_epoch = epoch


def convergence(history, decay_count):
    """Original validation-only early-stopping criterion."""
    epoch = history[-1]["epoch"]
    recent = max(row["dev"]["rsum"] for row in history[-5:])
    prior = max((row["dev"]["rsum"] for row in history[:-5]), default=None)
    return (epoch in (5, 10, 15, 20, 25) and decay_count >= 2
            and prior is not None and recent - prior < 1.0)


def retrieval_metrics(scores):
    """Both retrieval directions; five consecutive paired captions per image."""
    n_images, n_captions = scores.shape
    if n_images == 0 or n_captions != 5 * n_images:
        raise ValueError("Expected an N-image by 5N-caption score matrix")

    image_ranks = np.empty(n_images, dtype=np.int64)
    for image in range(n_images):
        order = np.argsort(scores[image])[::-1]
        positions = np.empty(n_captions, dtype=np.int64)
        positions[order] = np.arange(n_captions)
        image_ranks[image] = positions[5 * image:5 * image + 5].min()

    text_ranks = np.empty(n_captions, dtype=np.int64)
    for caption in range(n_captions):
        order = np.argsort(scores[:, caption])[::-1]
        text_ranks[caption] = np.flatnonzero(order == caption // 5)[0]

    i2t = [float(100 * np.mean(image_ranks < k)) for k in (1, 5, 10)]
    t2i = [float(100 * np.mean(text_ranks < k)) for k in (1, 5, 10)]
    return {"i2t": i2t, "t2i": t2i, "rsum": sum(i2t) + sum(t2i)}


@torch.no_grad()
def evaluate_loader(model, loader):
    """Basic S(I,T) scoring used to select the model on validation data."""
    model.eval()
    image_features = None
    text_features = None
    text_lengths = None
    n_rows = len(loader.dataset)
    for batch in loader:
        images, captions, lengths, row_ids, _ = batch
        visual, textual, _ = model.forward_emb(images, captions, lengths)
        if image_features is None:
            image_features = torch.empty(n_rows, visual.shape[1], visual.shape[2])
            text_features = torch.zeros(n_rows, model.opt.max_word, textual.shape[2])
            text_lengths = torch.empty(n_rows, dtype=torch.long)
        if int(lengths.max()) > model.opt.max_word:
            raise ValueError("A caption exceeds --max_word; increase the feature-cache capacity")
        image_features[row_ids] = visual.cpu()
        text_features[row_ids, :textual.shape[1]] = textual.cpu()
        text_lengths[row_ids] = lengths.cpu()

    if image_features is None:
        raise ValueError("The validation split is empty")
    if n_rows % 5:
        raise ValueError("Validation requires five consecutive captions per image")
    for row in range(0, n_rows, 5):
        if len(set(loader.dataset.images[row // loader.dataset.im_div:
                                         (row + 4) // loader.dataset.im_div + 1])) != 1:
            raise ValueError("Validation captions are not grouped by image")

    # Construct units once per image, then reuse them across text blocks.
    image_features = image_features[::5]
    block = model.opt.shard_size
    unit_blocks = []
    with precision_context(model.opt):
        for first in range(0, len(image_features), block):
            units, _, _ = model.cross_net.compute_image_units(
                image_features[first:first + block].cuda())
            unit_blocks.append(units.cpu())
        image_units = torch.cat(unit_blocks)
        scores = np.empty((len(image_units), n_rows), dtype=np.float32)
        for first in range(0, len(image_units), block):
            units = image_units[first:first + block].cuda()
            for start in range(0, n_rows, block):
                values = model.cross_net.score_image_units(
                    units, text_features[start:start + block].cuda(),
                    text_lengths[start:start + block])
                scores[first:first + len(units), start:start + values.shape[1]] = (
                    values.float().cpu().numpy())
    return retrieval_metrics(scores)


def train_main(opt):
    if opt.img_res != (384 if opt.vit_type == "vit_384" else 224):
        raise ValueError("--img_res must match the selected ViT variant")
    if not torch.cuda.is_available():
        raise RuntimeError("The supplied model implementation requires a CUDA GPU")
    if opt.num_epochs < 1 or opt.batch_size < 1 or opt.shard_size < 1:
        raise ValueError("Epoch count, batch size, and shard size must be positive")
    if opt.num_slots < 1 or opt.slot_iters < 1 or opt.caption_chunk_size < 1:
        raise ValueError("Unit count, aggregation rounds, and caption chunk size must be positive")
    if opt.workers < 0 or opt.log_step < 1:
        raise ValueError("Worker count must be nonnegative and log interval must be positive")

    path = Path(opt.logger_name)
    path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path / "train.log", encoding="utf-8")],
        force=True,
    )
    torch.set_num_threads(4)
    torch.cuda.set_device(opt.gpu_id)
    set_seed(opt.seed)
    tokenizer = BertTokenizer.from_pretrained(opt.local_bert_path, local_files_only=True)
    train_loader = data.get_train_loader(
        opt, opt.data_path, tokenizer, opt.batch_size, opt.workers, "train")
    dev_loader = data.get_test_loader(
        opt, opt.data_path, tokenizer, opt.batch_size, opt.workers, "dev")
    if len(train_loader) == 0:
        raise ValueError("The training split has no complete batch")
    model = SlotsModel(opt).cuda()
    optimizer = create_optimizer(opt, model)
    scheduler = OfficialSchedule(optimizer, opt.lr_schedules, opt.decay_rate)
    # Preserve the supplied training path's reset after model construction.
    set_seed(opt.seed)
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(opt).items()}
    save_json(configuration, path / "config.json")

    history = []
    best_dev = -float("inf")
    for epoch in range(opt.num_epochs):
        scheduler.step(epoch)
        model.set_max_violation(epoch >= opt.vse_mean_warmup_epochs)
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            images, captions, lengths, _, image_ids = batch
            optimizer.zero_grad()
            warmup_alpha = step / len(train_loader) if epoch == opt.embedding_warmup_epochs else 1.0
            loss, _ = model(
                images.cuda(non_blocking=True), captions.cuda(non_blocking=True),
                lengths.cuda(non_blocking=True), img_ids=image_ids.cuda(non_blocking=True),
                warmup_alpha=warmup_alpha,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step + 1}")
            loss.backward()
            if opt.grad_clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            model.Eiters += 1
            total_loss += loss.detach().float().item()
            if (step + 1) % opt.log_step == 0:
                logging.info("epoch=%d step=%d/%d loss=%.6f", epoch + 1,
                             step + 1, len(train_loader), loss.detach().float().item())
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        dev = evaluate_loader(model, dev_loader)
        row = {"epoch": epoch + 1, "dev": dev, "loss": total_loss / len(train_loader),
               "training_seconds": training_seconds,
               "learning_rates": [group["lr"] for group in optimizer.param_groups]}
        history.append(row)
        logging.info("epoch=%d validation_rsum=%.4f", epoch + 1, dev["rsum"])
        if dev["rsum"] > best_dev:
            best_dev = dev["rsum"]
            save_checkpoint({"model": model.state_dict(), "opt": opt, "epoch": epoch + 1,
                             "best_rsum": best_dev, "Eiters": model.Eiters,
                             "selection_split": "dev"}, path / "model_best_dev.pth")
        save_json(history, path / "history.json")
        if opt.early_stop and convergence(history, len(scheduler.applied)):
            logging.info("Stopped at epoch %d by validation convergence", epoch + 1)
            break
    return model


if __name__ == "__main__":
    train_main(get_argument_parser().parse_args())
