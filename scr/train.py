import os
import yaml
import time
import datetime
import warnings
from tqdm import tqdm

import torch
import timm
import pandas as pd
import numpy as np
import torch.nn as nn

from box import Box
from torch.utils.data import DataLoader
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from utils.create_dataset import BirdDataset
from utils.base_utils import set_seed
from utils.metrics import validation_epoch_end

warnings.filterwarnings("ignore", category=UserWarning)
date_now = datetime.datetime.now().strftime("%d_%B_%Y_%H_%M")



def main():
    for epoch_i in range(1, config.epochs + 1):
        if config.debug:
            k = 1
        start = time.time()
        logger.info(f'---------------------epoch:{epoch_i}/{config.epochs}---------------------')

        # loss
        avg_train_loss = 0
        avg_val_loss = 0
        predicted_labels_list = None
        true_labels_list = None

        ############## Train #############
        model.train()
        train_pbar = tqdm(train_loader, desc="Training")
        for batch in train_pbar:
            X_batch = batch[0].to(config.device)
            y_batch = batch[1].to(config.device)

            optimizer.zero_grad()
            res = model.forward(X_batch)
            loss = loss_f(res.float(), y_batch)

            if torch.cuda.is_available():
                train_pbar.set_postfix(gpu_load=f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB",
                                       loss=f"{loss.item():.4f}")
            else:
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            loss.backward()
            optimizer.step()

            avg_train_loss += loss * len(y_batch)
            del batch, res

            if config.scheduler:
                scheduler.step()

            if config.debug:
                k += 1
                if k > 5:
                    break

        model.eval()

        ########## VALIDATION ###############
        with torch.no_grad():
            for batch in (valid_loader):
                X_batch = batch[0].to(config.device)
                y_batch = batch[1].to(config.device)

                res = model.forward(X_batch)
                loss = loss_f(res.float(), y_batch)
                y_batch_onehot = y_batch

                avg_val_loss += loss * len(y_batch)

                # metrics
                res = res.detach().cpu().sigmoid().numpy()
                y_batch_onehot = y_batch_onehot.unsqueeze(1).detach().cpu().numpy()
                y_batch_onehot = y_batch_onehot.squeeze()

                if predicted_labels_list is None:
                    predicted_labels_list = res
                    true_labels_list = y_batch_onehot
                else:
                    predicted_labels_list = np.concatenate([predicted_labels_list, res], axis=0)
                    true_labels_list = np.concatenate([true_labels_list, y_batch_onehot], axis=0)

                del batch, res

                if config.debug:
                    k += 1
                    if k > 10:
                        break

        torch.cuda.empty_cache()

        avg_train_loss = avg_train_loss / len(dataset_train)
        avg_val_loss = avg_val_loss / len(dataset_test)

        all_predicted_labels = np.vstack(predicted_labels_list)
        all_true_labels = np.vstack(true_labels_list)
        all_true_labels = np.squeeze(all_true_labels)
        mask = (all_true_labels > 0) & (all_true_labels < 1)
        all_true_labels[mask] = 0
        avg_metric = metric(all_true_labels, all_predicted_labels)

        logger.info(f'epoch: {epoch_i}')

        logger.info("loss_train: %0.4f| loss_valid: %0.4f|" % (avg_train_loss, avg_val_loss))
        for m in avg_metric:
            logger.info(f"metric {m} : {avg_metric[m]:.<5g}")

        elapsed_time = time.time() - start
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        logger.info(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")

        if epoch_i % 4 == 0:
            torch.save(model, f'{path_save}/model_{config.model_name}_ep_{epoch_i}.pt')

        torch.save(model, f'{path_save}/model_{config.model_name}_last_version.pt')


if __name__  == "__main__":
    path_save = os.path.join("../experiment", date_now)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # Load config with params for training
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config = Box(config)

    logger.add(f"{path_save}/info_log{date_now}.log",
               format="<red>{time:YYYY-MM-DD HH:mm:ss}</red>| {message}")

    file_name = __file__
    logger.info(f'file for running: {file_name}')
    with open(file_name, 'r') as file:
        code = file.read()
        logger.info(code)

    logger.info(f"Folder with experiment- {path_save}")
    logger.info("----------params----------")

    for param in config:
        logger.info(f"{param}: {str(config[param])}")

    # Create device for training and set_seed:
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed=config.seed)

    # Read data
    df = pd.read_csv("../data/data.csv")
    df_train, df_test = (df[df.fold != 3].reset_index(drop=True),
                         df[df.fold == 3].reset_index(drop=True)
                         )

    logger.info(f"Size df_train- {df_train.shape[0]}")
    logger.info(f"Size df_test- {df_test.shape[0]}")

    dataset_train = BirdDataset(df=df_train,
                                path_to_folder_with_audio=config.path_to_files_base
                                )
    dataset_test = BirdDataset(df=df_test,
                               path_to_folder_with_audio=config.path_to_files_base
                               )

    train_loader = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    valid_loader = DataLoader(dataset_test,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers)

    logger.info(f"New experiment")
    model_name = config.model_name
    model = timm.create_model(model_name, pretrained=True).to(config.device)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 264)
    )
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(config.device)

    if config.metric == 'custom':
        metric = validation_epoch_end

    if config.loss_f == "nn.BCEWithLogitsLoss()":
        loss_f = nn.BCEWithLogitsLoss()

    # optimizer
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.optimizer_lr,
                                     weight_decay=config.optimizer_wd
                                     )

    if config.scheduler == "CosineAnnealingWarmRestarts":
        logger.info(f"Scheduler - {config.scheduler}")
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=10,
                                                T_mult=2,
                                                eta_min=0.000001,
                                                last_epoch=-1)

    # Train_loop
    logger.info(f"Starting train. Model - {config.model_name}")

    main()
