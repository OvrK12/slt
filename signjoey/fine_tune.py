#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import argparse
import numpy as np
import os
import shutil
import wandb

from signjoey.model import build_model
from signjoey.batch import Batch
from signjoey.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
)
from signjoey.model import SignModel
from signjoey.prediction import validate_on_data
from signjoey.loss import XentLoss
from signjoey.data import load_data, make_data_iter
from signjoey.builders import build_optimizer, build_scheduler, build_gradient_clipper
from signjoey.prediction import test
from signjoey.metrics import wer_single
from signjoey.vocabulary import SIL_TOKEN
from signjoey.training import TrainManager
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
from typing import List, Dict
from torch import nn

def reinitialize_layer(layer):
    """Reinitialize the weights of specified layer types.

    Args:
        layer (nn.Module): The layer to be reinitialized. 
    """   
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.Embedding):
        nn.init.xavier_uniform_(layer.weight)


def fine_tune(cfg_file: str, pre_trained_ckpt: str) -> None:
    """Finetune using checkpoint of pretrained model.

    Args:
            cfg_file (str): Configuration file.
            pre_trained_ckpt (str): Checkpoint of pretrained model.
    """   
    cfg = load_config(cfg_file)
    set_seed(seed=cfg["training"].get("random_seed", 42))

    
    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(data_cfg=cfg["data"])

    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"]) if isinstance(cfg["data"]["feature_size"], list) else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    model_checkpoint = load_checkpoint(pre_trained_ckpt, use_cuda=cfg["training"].get("use_cuda", False))
    model_state_dict = model.state_dict()
    pretrained_state_dict = model_checkpoint["model_state"]
  
    matched_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
    mismatched_keys = [k for k in pretrained_state_dict.keys() if k not in matched_state_dict]

    model_state_dict.update(matched_state_dict)
    model.load_state_dict(model_state_dict)

    for key in mismatched_keys:
        layer_name = key.split('.')[0]
        layer = dict(model.named_modules())[layer_name]
        reinitialize_layer(layer)

    trainer = TrainManager(model=model, config=cfg)
    
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")
    log_cfg(cfg, trainer.logger)

    log_data_info(
        train_data=train_data,
        valid_data=dev_data,
        test_data=test_data,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        logging_function=trainer.logger.info,
    )

    trainer.logger.info(str(model))

    gls_vocab_file = f"{cfg['training']['model_dir']}/gls.vocab"
    gls_vocab.to_file(gls_vocab_file)
    txt_vocab_file = f"{cfg['training']['model_dir']}/txt.vocab"
    txt_vocab.to_file(txt_vocab_file)

    
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    del train_data, dev_data, test_data

    ckpt = f"{trainer.model_dir}/{trainer.best_ckpt_iteration}.ckpt"
    output_name = f"best.IT_{trainer.best_ckpt_iteration:08d}"
    output_path = os.path.join(trainer.model_dir, output_name)
    logger = trainer.logger
    del trainer
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "fine_tune"],
        default="train",
        help="Mode to run: train, test, or fine_tune",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint for fine-tuning or testing",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        train(cfg_file=args.config)
    elif args.mode == "test":
        if args.ckpt is None:
            raise ValueError("Checkpoint must be provided for testing")
        test(cfg_file=args.config, ckpt=args.ckpt)
    elif args.mode == "fine_tune":
        if args.ckpt is None:
            raise ValueError("Checkpoint must be provided for fine-tuning")
        fine_tune(cfg_file=args.config, pre_trained_ckpt=args.ckpt)