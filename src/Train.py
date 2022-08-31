from Model import Cox_nnet
from Survival_CostFunc_CIndex import R_set, neg_par_log_likelihood, c_index

import torch
import torch.optim as optim
import copy
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader



dtype = torch.FloatTensor

def reset_weights(m):
	for layer in m.children():
		if hasattr(layer, 'reset_parameters'):
			layer.reset_parameters()

def trainCox_nnet(data2, \
			In_Nodes, Hidden_Nodes, Out_Nodes, \
			Learning_Rate, L2, l1_lambda, Num_Epochs, Dropout_Rate, batch_size):
	"""train the model

	Args:
		data2 (class): from the class Custom Data set
		Hidden_Nodes (int)
		Out_Nodes (int)
		L2 (float)
		l1_lambda (float)
		Num_Epochs (int)
		Dropout_Rate (float)
		batch_size (int)
	Returns:
		list: list of the loss and the accuracy for every epoch and every fold
	"""
	k_folds = 5
	kfold = KFold(n_splits=k_folds, shuffle=True)
	history_val=[[],[],[],[],[],[],[],[],[],[]]
	history_train=[[],[],[],[],[],[],[],[],[],[]]
	loss_batch_train = [[],[],[],[],[]]
	for fold,(train_idx,test_idx) in enumerate(kfold.split(data2)):
		net = Cox_nnet(In_Nodes, Hidden_Nodes, Out_Nodes, Dropout_Rate)
		opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2)
		print('------------fold no---------{}----------------------'.format(fold))
		train_loader = DataLoader(data2, batch_size=batch_size, sampler=train_idx)
		val_loader = DataLoader(data2, batch_size=batch_size, sampler=test_idx)
		for epoch in range(Num_Epochs+1):
			#training phase
			pred_train=[]
			for batch in train_loader: 
				loss = net.training_step(batch)
				loss = loss['val_loss']
				regularization_loss = 0
				for param in net.parameters():
					regularization_loss += torch.sum(abs(param))
				loss = loss+l1_lambda*regularization_loss
				loss_batch_train.append(loss)
				loss.backward() ###calculate gradients
				opt.step() ###update weights and biases
				opt.zero_grad() ###reset gradients to zeros
				pred_train.append(net.training_step(batch))
			result_train = net.training_epoch_end(pred_train)
			pred_val = 	[net.validation_step(batch) for batch in val_loader]
			result_val = net.validation_epoch_end(pred_val)
			net.epoch_end(epoch, result_val)
			history_val[fold].append(result_val)
			history_train[fold].append(result_train)
		net.apply(reset_weights)
	return (history_train, history_val)

