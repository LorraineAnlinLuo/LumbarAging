"""trainer code"""
import copy
import logging
import os
from typing import List, Dict, Optional, Callable, Union
import dill
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_absolute_error,mean_squared_error
import math

logger = logging.getLogger()


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim,
            scheduler: torch.optim.lr_scheduler,
            gradient_norm_clip=-1,
            max_epoch: int = 100,
            patience: int = 20,
            device=None,
            result_dir=None,
    ):
        """
            stopping_criteria : can be a function, string or none. If string it should match one
            of the keys in mae or should be loss, if none we don't invoke early stopping
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.gradient_norm_clip = gradient_norm_clip
        self.max_epoch = max_epoch
        self.patience = patience
        self.device = device
        self.result_dir = result_dir

        self.epoch = 0
        self.best_epoch = -1
        self.best_mae = None
        self.best_model = self.model.state_dict()

        if result_dir is not None:
            self.writer = SummaryWriter(log_dir=result_dir)

    def load(self, fname: str) -> Dict:
        data = torch.load(open(fname, "rb"), pickle_module=dill)

        if getattr(self, "model", None) and data.get("model") is not None:
            state_dict = self.model.state_dict()
            state_dict.update(data["model"])
            self.model.load_state_dict(state_dict)

        if getattr(self, "optimizer", None) and data.get("optimizer") is not None:
            optimizer_dict = self.optimizer.state_dict()
            optimizer_dict.update(data["optimizer"])
            self.optimizer.load_state_dict(optimizer_dict)

        if getattr(self, "scheduler", None) and data.get("scheduler") is not None:
            scheduler_dict = self.scheduler.state_dict()
            scheduler_dict.update(data["scheduler"])
            self.scheduler.load_state_dict(scheduler_dict)

        self.epoch = data["epoch"]
        self.best_mae = data["best_mae"]
        self.best_epoch = data["best_epoch"]
        return data

    def save(self, fname: str, **kwargs):
        kwargs.update({
                "model"        : self.model.state_dict(),
                "optimizer"    : self.optimizer.state_dict(),
                "epoch"        : self.epoch,
                "best_mae"     :  self.best_mae,
                "best_epoch"   : self.best_epoch,
        })

        if self.scheduler is not None:
            kwargs.update({"scheduler": self.scheduler.state_dict()})

        torch.save(kwargs, open(fname, "wb"), pickle_module=dill)

    def train(self, train_loader, valid_loader):
        while self.epoch < self.max_epoch:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("train/lr", lr, self.epoch)
            self.epoch += 1
            logger.info(f"Starting epoch {self.epoch} [lr:{lr}]")
            train_loss, train_mae = self.train_epoch(train_loader)
            self.writer.add_scalar("train/loss", train_loss, self.epoch)
            self.writer.add_scalar("train/mae", train_mae, self.epoch)
            logger.info(f"train_loss:{train_loss} train_mae:{train_mae}")
            val_loss, val_mae = self.validate(valid_loader)
            self.writer.add_scalar("val/loss", val_loss, self.epoch)
            self.writer.add_scalar("val/mae", val_mae, self.epoch)
            logger.info(f"val_loss:{val_loss}   val_mae:{val_mae}")

            if ((self.best_mae is None) or (self.best_mae > val_mae)):
                self.best_mae = val_mae
                self.best_epoch = self.epoch
                self.best_model = copy.deepcopy(
                    {k: v.cpu() for k, v in self.model.state_dict().items()})

                logger.info(f"Saving best model at epoch {self.epoch}, best_mae is {self.best_mae} !!!")
                self.save(f"{self.result_dir}/best_model.pt")

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_mae)
                else:
                    self.scheduler.step()

            if self.epoch - self.best_epoch > self.patience:
                logger.info(f"Patience reached stopping training after {self.epoch} epochs")
                break

        logger.info("Saving the last model")
        self.save(f"{self.result_dir}/last_model.pt")

    def train_epoch(self, train_loader):
        train_loss = []
        train_mae = []

        self.model.train()
        for i, (name, input, target) in enumerate(train_loader):
            input = input.to(self.device)
            target = target.to(self.device)

            output, p1, p2, z1, z2 = self.model(input)
          #  loss = criterion(output, p1, p2, z1, z2, target)
            #print("target:",target)
            #print("output:",output)
            #布尔值判断条件，
            mask = (output >=60)
            append_age = 0.65
            # mask_y = (output >= 60)
            # sub_age = 4.5
            # output[mask_y] = output[mask_y].add(sub_age)
         #   mask_y =  (output <= 20)
          #  sub_age = 3
         #   output[mask_y] = output[mask_y].sub(sub_age)
        #     append_age = torch.tensor([2.6816, 2.6817, 2.6822, 2.6820, 2.6813, 2.6820, 2.6816, 2.6811,
        # 2.6814, 2.6818, 2.6818, 2.6818, 2.6817, 2.6815, 2.6815, 2.6818,
        # 2.6816, 2.6818, 2.6817, 2.6815, 2.6816, 2.6815, 2.6817, 2.6814,
        # 2.6813, 2.6817, 2.6818, 2.6817, 2.6820, 2.6816, 2.6814, 2.6814], device='cuda:1')
           # print("Add age:",append_age)
            #print(output[mask])
            output[mask] = output[mask].add(append_age)
           # print(output)
            mae = metric(output,target)
            torch.autograd.set_detect_anomaly(True)

            loss = criterion(output, p1, p2, z1, z2, target)
            loss.backward()
            if self.gradient_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_norm_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss.append(loss.cpu().tolist())
            train_mae.append(mae)

        return np.mean(train_loss),np.mean(train_mae)

    def validate(self, valid_loader):
        val_loss = []
        val_mae = []

        self.model.eval()
        with torch.no_grad():
            for i, (name, input, target) in enumerate(valid_loader):
                input = input.to(self.device)
                target = target.to(self.device)

                output, p1, p2, z1, z2 = self.model(input)
             #   loss = criterion(output, p1, p2, z1, z2, target)
                # 布尔值判断条件，
                mask = (output >= 60)
                append_age = 0.65
                # mask_y = (output >= 60)
                # sub_age = 4.5
                # output[mask_y] = output[mask_y].add(sub_age)
              #  mask_y = (output <= 20)
              #  sub_age = 3
              #  output[mask_y] = output[mask_y].sub(sub_age)
             #   print(output[mask_y])
                #     append_age = torch.tensor([2.6816, 2.6817, 2.6822, 2.6820, 2.6813, 2.6820, 2.6816, 2.6811,
                # 2.6814, 2.6818, 2.6818, 2.6818, 2.6817, 2.6815, 2.6815, 2.6818,
                # 2.6816, 2.6818, 2.6817, 2.6815, 2.6816, 2.6815, 2.6817, 2.6814,
                # 2.6813, 2.6817, 2.6818, 2.6817, 2.6820, 2.6816, 2.6814, 2.6814], device='cuda:1')
                # print("Add age:",append_age)
              #  print(target)
              #  print(output[mask])
                output[mask] = output[mask].add(append_age)
              #  print(output)
                mae = metric(output, target)
            #    torch.autograd.set_detect_anomaly(True)

                loss = criterion(output, p1, p2, z1, z2, target)

                val_loss.append(loss.cpu().tolist())
                val_mae.append(mae)

        return np.mean(val_loss), np.mean(val_mae)

    def test(self, test_loader):

        test_loss = []
        test_mae = []
        name_list, output_list, target_list = [], [], []

        self.model.eval()
        with torch.no_grad():
            for i, (name, input, target) in enumerate(test_loader):
                input = input.to(self.device)
                target = target.to(self.device)

                output, p1, p2, z1, z2 = self.model(input)
              #  loss = criterion(output, p1, p2, z1, z2, target)

                # 布尔值判断条件，
                mask = (output >= 60)
                append_age = 0.65
                # mask_y = (output >= 60)
                # sub_age = 4.5
                # output[mask_y] = output[mask_y].add(sub_age)
            #    print(output[mask_y])
                #     append_age = torch.tensor([2.6816, 2.6817, 2.6822, 2.6820, 2.6813, 2.6820, 2.6816, 2.6811,
                # 2.6814, 2.6818, 2.6818, 2.6818, 2.6817, 2.6815, 2.6815, 2.6818,
                # 2.6816, 2.6818, 2.6817, 2.6815, 2.6816, 2.6815, 2.6817, 2.6814,
                # 2.6813, 2.6817, 2.6818, 2.6817, 2.6820, 2.6816, 2.6814, 2.6814], device='cuda:1')
                # print("Add age:",append_age)
             #   print(target)
                output[mask] = output[mask].add(append_age)
             #   print(output[mask])
#               mae = metric(output, target)
               # torch.autograd.set_detect_anomaly(True)

                loss = criterion(output, p1, p2, z1, z2, target)

                test_loss.append(loss.cpu().tolist())
#                test_mae.append(mae)

                for i in list(name):
                    name_list.append(i)
                for i in output.cpu().numpy().flatten().tolist():
                    output_list.append(i)
                for i in target.cpu().numpy().flatten().tolist():
                    target_list.append(i)

#            logger.info(f"test_loss:{np.mean(test_loss)}   test_mae:{np.mean(test_mae)}")

            print('\n')
#            print(target_list)
#            print(output_list)
            print("预测您的腰椎年龄为：",output_list,"岁")
            # diff_list = []
            # for i in range(len(target_list)):
            #     diff_list.append(math.fabs(target_list[i] - output_list[i]))
            #
            # zipped = zip(name_list,target_list,output_list,diff_list)
            # for res in sorted(zipped,key=lambda x:x[3],reverse=True):
            #     print(res)
            #     break

            # print('\n')
            # print("PCC", np.round(pearsonr(target_list,output_list)[0],3))  # 皮尔逊相关系数
            # print("SRCC", np.round(spearmanr(target_list,output_list)[0],3))  # 斯皮尔曼相关系数
            # print("RMSE", np.round(mean_squared_error(target_list, output_list) ** 0.5,3))  # 均方根误差

            # print('\n')
            # print('cs_0', cs_m(target_list, output_list, 0))
            # print('cs_1', cs_m(target_list, output_list, 1))
            # print('cs_2', cs_m(target_list, output_list, 2))
            # print('cs_3', cs_m(target_list, output_list, 3))
            # print('cs_4', cs_m(target_list, output_list, 4))
            # print('cs_5', cs_m(target_list, output_list, 5))
            # print('cs_6', cs_m(target_list, output_list, 6))
            # print('cs_7', cs_m(target_list, output_list, 7))
            # print('cs_8', cs_m(target_list, output_list, 8))
            # print('cs_9', cs_m(target_list, output_list, 9))
            # print('cs_10', cs_m(target_list, output_list, 10))

            # print('\n')
            # print('mcs_0', mcs_m(target_list, output_list, 0))
            # print('mcs_1', mcs_m(target_list, output_list, 1))
            # print('mcs_2', mcs_m(target_list, output_list, 2))
            # print('mcs_3', mcs_m(target_list, output_list, 3))
            # print('mcs_4', mcs_m(target_list, output_list, 4))
            # print('mcs_5', mcs_m(target_list, output_list, 5))
            # print('mcs_6', mcs_m(target_list, output_list, 6))
            # print('mcs_7', mcs_m(target_list, output_list, 7))
            # print('mcs_8', mcs_m(target_list, output_list, 8))
            # print('mcs_9', mcs_m(target_list, output_list, 9))
            # print('mcs_10', mcs_m(target_list, output_list, 10))

            # print('\n')
            # print('mae (11,20)', mae_range(11, 20, target_list, output_list))
            # print('mae (21,30)', mae_range(21, 30, target_list, output_list))
            # print('mae (31,40)', mae_range(31, 40, target_list, output_list))
            # print('mae (41,50)', mae_range(41, 50, target_list, output_list))
            # print('mae (51,60)', mae_range(51, 60, target_list, output_list))
            # print('mae (61,70)', mae_range(61, 70, target_list, output_list))
            # print('mae (71,80)', mae_range(71, 80, target_list, output_list))
            # print('mae (11,80)', mae_range(11, 80, target_list, output_list))

            # # ======= Draw scatter plot of predicted age against true age ======= #
            # lx = np.arange(np.min(target_list), np.max(target_list))
            # l1 = plt.plot(lx, lx, color='red', linestyle='solid')
            # plt.scatter(target_list, output_list, color='green')
            # plt.xlabel('True Age')
            # plt.ylabel('Predicted Age')
            # plt.legend(handles=l1, labels=["y=x"], loc="best")
            # plt.text(17.5, 67, f'MAE={mae_range(11, 80, target_list, output_list)}')
            # plt.title("ours")
            # plt.savefig(os.path.join(self.result_dir, "true_and_predicted_age"))
            # plt.close()


def criterion(output, p1, p2, z1, z2, target):
    output = output.squeeze()

    mse_criterion = torch.nn.MSELoss()
    mse_loss = mse_criterion(output,target)

    # a = np.random.randint(0, output.size(0), output.size(0))
    # b = np.random.randint(0, target.size(0), target.size(0))
    # diff_output = (output[a] - output[b])
    # diff_target = (target[a] - target[b])
    # age_difference_loss = torch.mean((diff_output - diff_target) ** 2)

    cossim_criterion = torch.nn.CosineSimilarity()  # 损失函数定义，余弦相似性
    cossim_loss = -(cossim_criterion(p1, z2).mean() + cossim_criterion(p2, z1).mean())*0.5

    lamda1 = 1
    lamda2 = 1
    loss = lamda1 * mse_loss + lamda2 * cossim_loss
    return loss


def metric(output, target):
    output = output.squeeze().cpu().data.numpy()
    target = target.cpu().data.numpy()
    # print(output)
    # print(target)
    mae = mean_absolute_error(target,output)
    return mae


def cs_m(target_numpy,predicted_numpy,m):
    num = 0
    for (i,j) in zip(target_numpy,predicted_numpy):
        if math.fabs(i-j) <= m:
            num += 1
    return np.round(100*num/len(target_numpy),3)


def mcs_m(target_numpy,predicted_numpy,m):
    cs_list = []
    for every_m in range(m+1):
        cs_list.append(cs_m(target_numpy,predicted_numpy,every_m))
    return np.round(np.mean(cs_list),3)


def mae_range(start,end,target_numpy,predicted_numpy):
    error_list = []
    for (i,j) in zip(target_numpy,predicted_numpy):
        if start <= i and i <= end:
            error_list.append(math.fabs(i - j))
    return np.round(np.mean(error_list),3)