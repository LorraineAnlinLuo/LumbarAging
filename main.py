"""entry point for training a classifier"""

import argparse
import logging
import os
import random
import numpy as np
import torch
import warnings
from lib.base_trainer import Trainer
from lib import os_utils as os_utils
from lib.data_utils import get_loader
from lib.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
import models.slice_set as slice_set
import torch.optim as optim
import shutil

warnings.filterwarnings("ignore")
torch.cuda.set_device(1)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["NUMEXPR_MAX_THREADS"] = '16'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    str2bool = os_utils.str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu_id', default='1', type=str)

    #parser.add_argument("--root_path", default="./data/augmix123", type=str)
    # parser.add_argument("--train_csv", default="./data/mixed_train.csv", type=str)
    # parser.add_argument("--val_csv", default="./data/mixed_val.csv", type=str)
    # parser.add_argument("--test_csv", default="./data/mixed_test.csv", type=str)
    # parser.add_argument("--root_path", default="./data/crop_scale_intensity_resize123", type=str)
    #parser.add_argument("--root_path", default="G:/age_regression/yyl/work2/2d-slice-set-networks/data/val_crop_scal_intensity_resize", type=str)
    parser.add_argument("--root_path", default="G:/age_regression/yyl/work2/2d-slice-set-networks/data/crop_scale_intensity_resize45", type=str)
    #parser.add_argument("--root_path", default="./data/ori_scale_intensity_resize", type=str)
    #parser.add_argument("--root_path", default="./data/crop_scale_intensity_resize12345other", type=str)
   #  parser.add_argument("--train_csv", default="./data_aug/train.csv", type=str)
   #  parser.add_argument("--val_csv", default="./data_aug/val.csv", type=str)
   #  parser.add_argument("--test_csv", default="./data_aug/test.csv", type=str)

    # parser.add_argument("--root_path", default="./data/crop_scale_intensity_resize123", type=str)
    # "New"
    # parser.add_argument("--train_csv", default="./data/A_train_4dataset.csv", type=str)train-ynew3
    # parser.add_argument("--val_csv", default="./data/A_val_4dataset.csv", type=str)val-ynew
    # parser.add_argument("--test_csv", default="./data/A_test_4dataset.csv", type=str)
    #"Old"
    parser.add_argument("--train_csv", default="./data/Alldatatrain_data.csv", type=str)
    parser.add_argument("--val_csv", default="./data/Alldataval_data.csv", type=str)
    parser.add_argument("--test_csv", default="./data/test-ynew3.csv", type=str)

    # parser.add_argument("--frame_keep_style", default="random", type=str, choices=["random", "ordered"], help="style of keeping frames when frame_keep_fraction < 1")
    # parser.add_argument("--frame_keep_fraction", default=1, type=float, help="fraction of frame to keep (usually used during testing with missing frames)")
    # parser.add_argument("--frame_dim", default=3, type=int, choices=[1, 2, 3], help="choose which dimension we want to slice, 1 for sagittal, 2 for coronal, 3 for axial")
    # parser.add_argument("--impute", default="drop", type=str, choices=["drop", "fill", "zeros", "noise"])

    parser.add_argument("--attn_dim", default=32, required=False, type=int)  # 128
    parser.add_argument("--attn_num_heads", default=1, required=False, type=int)  # 2
    parser.add_argument("--attn_drop", default=True, required=False, type=str2bool)
    parser.add_argument("--agg_fn", default="attention", required=False, type=str, choices=["mean", "max", "attention"])

    parser.add_argument("--lstm_feat_dim", default=2, required=False, type=int)
    parser.add_argument("--lstm_latent_dim", default=128, required=False, type=int)

    parser.add_argument("--seed", default=0, required=False, type=int)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--result_folder", default="result/1new_TS/2d_slice_attention", required=False)
    parser.add_argument("--batch_size", default=16, required=False, type=int)
    parser.add_argument("--patience", default=50, required=False, type=int)
    parser.add_argument("--max_epoch", default=200, required=False, type=int)
    # 修改!!!
    parser.add_argument("--optimizer", default="adam", required=False, type=str, choices=["adam", "sgd"])
    parser.add_argument("--scheduler", default="warmup_linear", required=False, type=str)
    parser.add_argument("--gradient_norm_clip", default=-1, required=False, type=float)

    args = parser.parse_args()
    set_seed(args)

    os_utils.safe_makedirs(args.result_folder)
    args.result_folder = os_utils.get_state_params(args.run_id, args.result_folder)
    shutil.copy("./main.py", args.result_folder)
    shutil.copy("./models/slice_set.py", args.result_folder)
    shutil.copy("./lib/base_trainer.py", args.result_folder)

    train_loader, val_loader, test_loader = get_loader(args)

    # Encoder
    # model = slice_set.get_model(attn_num_heads=args.attn_num_heads, attn_dim=args.attn_dim, attn_drop=args.attn_drop)
    # Encoder+AggregationBlock
    # model = slice_set.get_model(attn_num_heads=args.attn_num_heads, attn_dim=args.attn_dim, attn_drop=args.attn_drop, agg_fn=args.agg_fn)
    # Encoder + CosineSimilarity
    # model = slice_set.get_model(attn_num_heads=args.attn_num_heads, attn_dim=args.attn_dim, attn_drop=args.attn_drop)
    # Encoder+AggregationBlock+CosineSimilarity
    model = slice_set.get_model(attn_num_heads=args.attn_num_heads, attn_dim=args.attn_dim, attn_drop=args.attn_drop, agg_fn=args.agg_fn)

    model.to(args.device)
    # print(model)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # elif args.optimizer == "sgd":
    #     optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=5e-4, momentum=0.9)

    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
    elif args.scheduler == "multi_step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[100, 200, 300])
    elif args.scheduler == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    # elif args.scheduler == "reduce_on_plateau":
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,factor=0.1,min_lr=1e-7,verbose=True,threshold=1e-7)
    elif args.scheduler == "warmup_cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=0, t_total=args.max_epoch)
    elif args.scheduler == "warmup_linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=args.max_epoch)
    elif args.scheduler is None:
        scheduler = None

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler,
                      gradient_norm_clip=args.gradient_norm_clip,
                      max_epoch=args.max_epoch,
                      patience=args.patience,
                      device=args.device,
                      result_dir=args.result_folder
                      )

    logger.info("starting training")
    trainer.train(train_loader, val_loader)
    logger.info("Training done;")

    logger.info("Loading best model")
    # trainer.load("./result/run_0001/best_model.pt")
    trainer.load(f"{trainer.result_dir}/best_model.pt")
    logger.info("evaluating model on test set")
    trainer.test(test_loader)