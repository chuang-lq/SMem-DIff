import os
from datetime import datetime

import torch

from utils import Logger
from .ddp import dist_process
from .dp import process
# from .dp2 import process
from .test import test


class Trainer(object):
    def __init__(self, para, config_path=None):
        self.para = para
        self.config_path = config_path
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

    def run(self):
        # recoding parameters
        self.para.time = datetime.now()
        logger = Logger(self.para, self.config_path)
        logger.record_para()

        # training
        if not self.para.test_only:
            if self.para.trainer_mode == 'ddp':
                dist_process(self.para)
            elif self.para.trainer_mode == 'dp':
                # with torch.autograd.set_detect_anomaly(True):
                process(self.para)

        # valid
        test(self.para, logger)
