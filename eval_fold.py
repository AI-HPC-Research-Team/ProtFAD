import argparse, os, sys, datetime, shutil, copy, time

import pprint
import logging
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf

from models.metrics import fmax, auprc
from utils import instantiate_from_config, get_obj_from_str, add_params
import utils

@torch.no_grad()
def test(dataloader):
	model.eval()
	# Iterate over the validation data.

	probs = []
	labels = []
	for data in dataloader:
		data = data.to(device)
		with torch.no_grad():
			prob = model(data).sigmoid().detach().cpu().numpy()
			y = np.stack(data.y, axis=0)
		probs.append(prob)
		labels.append(y)
	probs = np.concatenate(probs, axis=0)
	labels = np.concatenate(labels, axis=0)

	return fmax(probs, labels), auprc(probs, labels)

# 命令行指令
def parse_args():
	parser = argparse.ArgumentParser(description='Mul-Pro')
	parser.add_argument("-C", "--config", type=str, required=True, help="the path of config file")
	parser.add_argument("-M", "--model_path", type=str, required=True, help="the path of model file")
	parser.add_argument("--cuda", type=str, default="0", help="the num of used cuda")
	parser.add_argument("--batch_size", type=int, help="the batch size of train_loader")
	opt, unknown = parser.parse_known_args()
	return opt, unknown

if __name__=="__main__":
	sys.path.append(os.getcwd()) # 将本脚本所在文件夹加入环境变量
	utils.set_seed(2024)

	opt, unknown = parse_args()
	print(opt)
	
	# ---------------------
	#  project root config
	# ---------------------
	configs = OmegaConf.load(opt.config)
	configs["cuda"] = "cuda:"+opt.cuda if torch.cuda.is_available() else "cpu"
	if opt.batch_size:
		configs["data"]["batch_size"] = opt.batch_size

	# -----------
	#  Load data
	# -----------
	data_config = configs.get("data")
	valid_dset = instantiate_from_config(data_config["valid_data"])
	if OmegaConf.is_list(data_config["test_data"]):  # 考虑有多个测试集的情况
		test_dsets = [instantiate_from_config(c) for c in data_config["test_data"]]
	else:
		test_dsets = [instantiate_from_config(data_config["test_data"])]

	valid_loader = DataLoader(valid_dset, batch_size=data_config["batch_size"], shuffle=False, num_workers=data_config["workers"])
	test_loaders = [DataLoader(test_dset, batch_size=data_config["batch_size"], shuffle=False, num_workers=data_config["workers"]) for test_dset in test_dsets]

	# --------------
	#  model config
	# --------------
	device = torch.device(configs["cuda"])

	model_config = OmegaConf.to_container(configs.get("model"))
	add_params(model_config, 'num_classes', valid_dset.num_classes)
	model = instantiate_from_config(model_config).to(device)
    
	model_size = sum(p.numel() for p in model.parameters())
	print("Total params: %.2fM" % (model_size/1e6))

	# --------------------
	#  checkpoint restore
	# --------------------
	checkpoint = torch.load(opt.model_path, map_location=device)
	model.load_state_dict(checkpoint)

	# ----------
	#  evaluate
	# ----------
	valid_acc = test(valid_loader)
	test_accs = [test(dl) for dl in test_loaders]
	print(f"accuracy: Validation: {valid_acc:.4f}, Test: ", '\t'.join([f'{acc:.4f}' for acc in test_accs]))

	
		