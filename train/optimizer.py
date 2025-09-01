from importlib import import_module
import torch
import torch.optim.lr_scheduler as lr_scheduler


class Optimizer:
    def __init__(self, para, target):
        # create optimizer
        trainable = target.parameters()
        optimizer_name = para.optimizer
        module = import_module('torch.optim')  # 根据配置或用户输入来动态导入相应模块
        # self.optimizer = torch.optim.Adam(trainable, lr=para.lr, betas=para.betas, weight_decay=para.weight_decay)
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=para.lr, betas=para.betas, weight_decay=para.weight_decay)
        # create scheduler
        try:
            if para.lr_scheduler == "multi_step":
                milestones = para.milestones
                gamma = para.decay_gamma
                self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
            elif para.lr_scheduler == "cosine":
                print('using cosine scheduler')
                self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=para.end_epoch, eta_min=1e-7)
            elif para.lr_scheduler == "cosineW":
                self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2,
                                                                          eta_min=1e-7)
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()
