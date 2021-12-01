"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info
import timm
import torchvision
import time
import wandb
from inference import inference, get_dataloader

def train(
            model_name: Dict[str, Any],
            data_config: Dict[str, Any],
            log_dir: str,
            fp16: bool,
            device: torch.device,
        ) -> Tuple[float, float, float]:

    
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    ##################
    ####모델
    ##################
    model_path = os.path.join(log_dir, "best.pt")
    model_instance = timm.create_model(model_name, pretrained=True)
    model_instance.to(device)
    ###########################
    #######################
    
    # Create dataloader
    train_dl, val_dl, _ = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(
        #model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
        model_instance.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model_instance,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_save_path=model_path,
        verbose=1,
        name = model_name,
        save_folder = log_dir
    )

    if args.train == True:
        best_acc, best_f1 = trainer.train(
            train_dataloader=train_dl,
            n_epoch=data_config["EPOCHS"],
            val_dataloader=val_dl
        )

    t0 = time.monotonic()

    dataloader = get_dataloader(img_root='/opt/ml/data/test', data_config="/opt/ml/code/exp/latest/data.yml")
    # inference
    all_time, inference_time = inference(model_instance, dataloader, "code/model_time", t0, model_name,device)
    wandb.log({
                    "all_time" : all_time, 
                    "inference_time": inference_time,
                })
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    
    parser.add_argument("--data", default="configs/data/taco.yaml", type=str, help="data config")
    parser.add_argument("--savefolder_name", type=str)
    parser.add_argument("--train", action = "store_true")
    parser.add_argument("--eval", type=str, default="")
    parser.add_argument("--resume_train", type=str, default="")
    
    args = parser.parse_args()

    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", args.savefolder_name))
    os.makedirs(log_dir, exist_ok=True)

    import timm
    
    models = timm.list_models(pretrained=True)
    
    for model_name in models[:2]:

        print(model_name)
        train(
            model_name = model_name,
            data_config=data_config,
            log_dir=log_dir,
            fp16=data_config["FP16"],
            device=device,
        )