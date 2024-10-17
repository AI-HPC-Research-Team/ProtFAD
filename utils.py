import os
import random
import numpy as np
import torch
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_checkpoint(model, optimizer, tracker, file_name):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'tracker': tracker,
    }
    torch.save(checkpoint, file_name)
    print(f"save the checkpoint at: {file_name}")

def load_checkpoint(model, optimizer, tracker, file_name, device="cpu"):
    checkpoint = torch.load(file_name, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    tracker.copy(checkpoint["tracker"])
    print(f"load the checkpoint from: {file_name}")

class train_tracker:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.lr = []

    def __len__(self):
        return len(self.train_losses)

    def append(self, train_loss, test_loss, lr):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.lr.append(lr)

    def plot(self, N=None):
        N = N if N is not None else self.__len__()
        plt.plot(self.train_losses[-N:],label='Train')
        plt.plot(self.test_losses[-N:], label='Eval')
        plt.legend()
        plt.show()

    def copy(self, tracker):
        self.train_losses = tracker.train_losses.copy()
        self.test_losses = tracker.test_losses.copy()
        self.lr = tracker.lr.copy()

    def get_sub_tracker(self, N):
        sub_tracker = train_tracker()
        sub_tracker.train_losses = self.train_losses[:N]
        sub_tracker.test_losses = self.test_losses[:N]
        sub_tracker.lr = self.lr[:N]
        return sub_tracker

import importlib

# 从字符串指定路径位置输入类或函数对象
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    module_imp = importlib.import_module(module, package=None) # import指定路径的文件化为对象导入
    if reload:
        importlib.reload(module_imp) # 在运行过程中若修改了库，需要使用reload重新载入
    return getattr(module_imp, cls) # getattr()函数获取对象中对应字符串的对象属性（可以是值、函数等）

# 从配置中载入模型
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module = get_obj_from_str(config["target"]) # target路径的类或函数模块
    params_config = config.get('params', dict()) # 对应模块的参数配置
    return module(**params_config)

def add_params(config, name:str, value):
    config['params'][name] = value
    return config

# set python hash seed
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

