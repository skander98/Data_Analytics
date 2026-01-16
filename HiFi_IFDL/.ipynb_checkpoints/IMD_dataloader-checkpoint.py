# ------------------------------------------------------------------------------
# Author: Xiao Guo, Xiaohong Liu.
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
from torch.utils.data import DataLoader
from utils.load_data import TrainData, ValData
from utils.load_edata import *

def train_dataset_loader_init(args):
	train_dataset = TrainData(args)
	train_data_loader = DataLoader(
								train_dataset, 
								batch_size=args.train_bs, 
								shuffle=True, 
								# shuffle=False,
								num_workers=8
								)
	return train_data_loader

def infer_dataset_loader_init(args, shuffle=True, bs=8):
	val_dataset = ValData(args)
	val_data_loader = DataLoader(
								val_dataset, 
								batch_size=bs,
								shuffle=shuffle, 
								# shuffle=True, 
								num_workers=8
								)
	return val_data_loader

def eval_dataset_loader_init(args, val_tag, batch_size=1):
	
	if val_tag == 0:
		data_label = 'CASIA2'
		val_data_loader = DataLoader(ValColumbia(args), batch_size=batch_size, shuffle=False,
									 num_workers=0)
	return val_data_loader, data_label