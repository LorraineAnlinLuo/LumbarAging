import os
import numpy as np
import SimpleITK as sitk
import cv2
import nibabel as nib
from scipy import ndimage

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

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled 

def crop_nifti_file(file_path, output_path, target_size=(224, 224, 8)):

    img = nib.load(file_path)
    data = img.get_fdata()
    (z, x, y) = data.shape
    print(x,y,z)
    cropped_data = data[140:390, 60:230 ,: ]
    cropped_img = nib.Nifti1Image(cropped_data, img.affine)

    nib.save(cropped_img, output_path)

def normalize_data(data):
    # b = np.percentile(data, 98)
    # t = np.percentile(data, 1)
    # data = np.clip(data,t,b)
    data = np.array(data,dtype=np.float32)
    means = data.mean()
    stds = data.std()
    # print(type(data),type(means),type(stds))
    data -= means
    data /= stds
    return data

image_path = 'G:/age_regression/yyl/work2/2d-slice-set-networks/test/nii_converted/'
names = os.listdir(image_path)
save_file_path = 'G:/age_regression/yyl/work2/2d-slice-set-networks/test/nii_silce'
if not os.path.exists(save_file_path):
                  os.makedirs(save_file_path)
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) #自适应直方图均衡化
for name in names:
    print(name)
    file_path = os.path.join(image_path, name)
    save_path = os.path.join(save_file_path,name)
    print(save_path)
    #save_path = os.path.join(save_file_path, name)
    crop_nifti_file(file_path, save_path, target_size=(224, 224, 8))
    print("Crop finished！")
    
    img = sitk.ReadImage(os.path.join(save_file_path, name))
    img_fdata = sitk.GetArrayFromImage(img)#将SimpleITK对象转换为ndarray

    (z, x, y) = img_fdata.shape
    print(x,y,z)

    saveimg=sitk.GetImageFromArray(img_fdata)
    saveimg.SetOrigin(img.GetOrigin())
    saveimg.SetDirection(img.GetDirection())
    saveimg.SetSpacing(img.GetSpacing())

    #图像与标签resize 128 128 96
    resize_img=resize_image_itk(saveimg, (224,224,8),resamplemethod= sitk.sitkLinear)
   # resize_mask=resize_image_itk(savemask, (512,512,161),resamplemethod= sitk.sitkNearestNeighbor)

    #图像标准化
    resize_imgarr=sitk.GetArrayFromImage(resize_img)#96 128 128
    nor_resize_imgarr=normalize_data(resize_imgarr)
    nor_resize_img=sitk.GetImageFromArray(nor_resize_imgarr)
    nor_resize_img.SetSpacing(resize_img.GetSpacing())
    nor_resize_img.SetOrigin(resize_img.GetOrigin())
    nor_resize_img.SetDirection(resize_img.GetDirection())

    resize_output_path = r'G:/age_regression/yyl/work2/2d-slice-set-networks/test/nii_resize'
    if not os.path.exists(resize_output_path):
                  os.makedirs(resize_output_path)
    #图像与标签保存
    sitk.WriteImage(nor_resize_img, os.path.join(resize_output_path,name))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    str2bool = os_utils.str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu_id', default='1', type=str)
    parser.add_argument("--root_path", default="G:/age_regression/yyl/work2/2d-slice-set-networks/test/nii_resize", type=str)

    parser.add_argument("--train_csv", default="./data/Alldatatrain_data.csv", type=str)
    parser.add_argument("--val_csv", default="./data/Alldataval_data.csv", type=str)
    parser.add_argument("--test_csv", default="G:/age_regression/yyl/work2/2d-slice-set-networks/test.csv", type=str)
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
    parser.add_argument("--result_folder", default="G:/age_regression/yyl/work2/2d-slice-set-networks/test", required=False)
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

#     logger.info("starting training")
#    # trainer.train(train_loader, val_loader)
#     logger.info("Training done;")

    logger.info("Loading best model")
    trainer.load("G:/age_regression/yyl/work2/2d-slice-set-networks/result/1new_TS/2d_slice_attention/run_0006/best_model.pt")
    #trainer.load(f"{trainer.result_dir}/best_model.pt")
    logger.info("evaluating model on test set")
    trainer.test(test_loader)
























