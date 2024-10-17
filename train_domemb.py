import argparse, os, sys, datetime, shutil
import copy
import time
import traceback

import logging

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf

from models.metrics import fmax
from utils import instantiate_from_config, get_obj_from_str, add_params
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# 命令行指令
def parse_args():
	parser = argparse.ArgumentParser(description='Mul-Pro')
	parser.add_argument("-C", "--config", type=str, help="the path of config file")
	parser.add_argument("-R", "--resume", type=str, default=None, help="the path of resumed project root")
	parser.add_argument("--cuda", type=str, default="0", help="the num of used cuda")
	parser.add_argument("--epochs", type=int, default=500, help="the epochs num of model training")
	opt, unknown = parser.parse_known_args()
	return opt, unknown

def train(epoch, dataloader, loss_fn):
	model.train()
	for data in dataloader:
		optimizer.zero_grad()
		data = data.to(device)
		all_loss = model.get_loss(data, loss_fn)
		all_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # Gradient clippling
		optimizer.step()

@ torch.no_grad()
def test(dataloader):
	model.eval()
	# Iterate over the validation data.

	losses = 0
	batchs = 0
	for data in dataloader:
		data = data.to(device)

		batch_num = data.domain_id.shape[0]
		loss = F.mse_loss(model(data), data.p)
		
		losses += loss.detach().cpu().numpy() * batch_num
		batchs += batch_num

	return losses / batchs

if __name__=="__main__":
	sys.path.append(os.getcwd()) # 将命令本脚本所在文件夹加入环境变量
	utils.set_seed(2024)

	opt, unknown = parse_args()
	print(opt)
	#os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

	#torch.autograd.set_detect_anomaly(True)
	
	# ---------------------
	#  project root config
	# ---------------------
	log_root = os.path.expanduser("/workspace/wangmq/experiments/Mul-Pro/")
	#log_root = os.path.expanduser("./saved/")

	if not opt.resume:
		assert opt.config, "must give one config: -C(config) or -R(resume)"

		configs = OmegaConf.load(opt.config)
		config_name = os.path.basename(opt.config).split(".")[0]
		now_time = datetime.datetime.now().strftime("%Y%m%d-T%H-%M") # 训练开始时间
		data_name = configs['data']['train_data']['target'].split('.')[-1]
		project_root = os.path.join(log_root, data_name, config_name+"-"+now_time)
	else:  # when the project resume...
		base_project_root = os.path.expanduser(opt.resume)
		configs = OmegaConf.load(os.path.join(base_project_root, "train.yaml"))
		project_root = base_project_root.rstrip('/') + '_resume'
		logger.info(f"load exist project from {base_project_root}.")
		shutil.copytree(base_project_root, project_root)
	logger.info(f"set the project root at: {project_root}")
		
	configs["num_epochs"] = opt.epochs
	configs["cuda"] = "cuda:"+opt.cuda if torch.cuda.is_available() else "cpu"
	logger.info("The train config has been init:")
	print(OmegaConf.to_yaml(configs))

	os.makedirs(project_root, exist_ok=True)
	print("Project root: ", project_root)
	OmegaConf.save(configs, os.path.join(project_root, "train.yaml"))

	# -----------
	#  Load data
	# -----------
	data_config = configs.get("data")
	train_dset = instantiate_from_config(data_config["train_data"])
	valid_dset = instantiate_from_config(data_config["valid_data"])
	if OmegaConf.is_list(data_config["test_data"]):  # 考虑有多个测试集的情况
		test_dsets = [instantiate_from_config(c) for c in data_config["test_data"]]
	else:
		test_dsets = [instantiate_from_config(data_config["test_data"])]

	train_loader = DataLoader(train_dset, batch_size=data_config["batch_size"], shuffle=True, drop_last=True, num_workers=data_config["workers"])
	valid_loader = DataLoader(valid_dset, batch_size=data_config["batch_size"], shuffle=False, num_workers=data_config["workers"])
	test_loaders = [DataLoader(test_dset, batch_size=data_config["batch_size"], shuffle=False, num_workers=data_config["workers"]) for test_dset in test_dsets]

	# --------------
	#  model config
	# --------------
	device = torch.device(configs["cuda"])
	ckpt_path = os.path.join(project_root, "checkpoint.pt")

	model_config = OmegaConf.to_container(configs.get("model"))
	model = instantiate_from_config(model_config).to(device)

	optimizer_config = OmegaConf.to_container(configs.get("optimizer"))
	optimizer_config['params']['params'] = model.parameters()
	optimizer = instantiate_from_config(optimizer_config)

	loss_fn = get_obj_from_str(optimizer_config['loss_fn'])()
	tracker = utils.train_tracker()
    
	model_size = sum(p.numel() for p in model.parameters())
	print("Total params: %.2fM" % (model_size/1e6))

	# -------------------------
	#  learning rate scheduler
	# -------------------------
	scheduler_config = configs.get("scheduler")
	if scheduler_config is not None:
		lr_weights = []
		for i, milestone in enumerate(scheduler_config['lr_milestones']):
			if i == 0:
				lr_weights += [np.power(scheduler_config['lr_gamma'], i)] * milestone
			else:
				lr_weights += [np.power(scheduler_config['lr_gamma'], i)] * (milestone - scheduler_config['lr_milestones'][i-1])
		if scheduler_config['lr_milestones'][-1] < configs["num_epochs"]:
			lr_weights += [np.power(scheduler_config['lr_gamma'], len(scheduler_config['lr_milestones']))] * (configs["num_epochs"] + 1 - scheduler_config['lr_milestones'][-1])
		lambda_lr = lambda epoch: lr_weights[epoch]
		lr_scheduler = get_obj_from_str(scheduler_config['target'])(optimizer, lr_lambda=lambda_lr)

	# --------------------
	#  checkpoint restore
	# --------------------
	if os.path.exists(ckpt_path):
		utils.load_checkpoint(model, optimizer, tracker, ckpt_path, device=device)

	# -------------
	#  train model
	# -------------
	begin_epoch = len(tracker)
	end_epoch = configs["num_epochs"]
	epochs_to_train = end_epoch - begin_epoch

	if epochs_to_train > 0:
		begin_time = time.time()
		best_valid = np.inf
		best_epoch = 0
		print(f"Start train the {model_config['target']} model: rest {epochs_to_train} epochs")

		for epoch in range(begin_epoch, end_epoch):
			train(epoch, train_loader, loss_fn)
			if scheduler_config is not None: lr_scheduler.step()
			valid_loss = test(valid_loader)
			test_loss = [test(dl) for dl in test_loaders]

			tracker.append(valid_loss, test_loss, optimizer.param_groups[0]['lr'])
			print(f"Epoch: {epoch+1:03d}, Validation: {valid_loss:.4f}, Test: ", '\t'.join([f'{fmax:.4f}' for fmax in test_loss]))
			if valid_loss < best_valid:
				best_valid = valid_loss
				best_test = test_loss
				best_epoch = epoch + 1
				best_model_state = copy.deepcopy(model.state_dict())			

		running_time = time.time() - begin_time
		print(f"Train time: {running_time//3600}h {running_time%3600//60}m {int(running_time%60)}s")
		print(f"Best epoch: {best_epoch+1:03d}, Validation: {best_valid:.4f}, Test: ", '\t'.join([f'{fmax:.4f}' for fmax in best_test]))

		# save checkpoint
		best_model_path = os.path.join(project_root, f"model_ep{best_epoch+1:03d}.pt")
		torch.save(best_model_state, best_model_path)
		print("save the best model at: ", best_model_path)
		utils.save_checkpoint(model, optimizer, tracker, ckpt_path)
		
	else: print(f"The model has been trained for {begin_epoch} epochs, no need to train again.")