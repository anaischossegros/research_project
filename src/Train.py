from Model import Cox_nnet
from SubNetwork_SparseCoding import dropout_mask, s_mask
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
		# print(train_idx)
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
			# pred_final = net(train_x, train_age)
		# net.apply(reset_weights)
	return (loss_batch_train,history_train, history_val)

# def trainCox_nnet(train_loader, \
# 			val_loader, \
# 			In_Nodes, Hidden_Nodes, Out_Nodes, \
# 			Learning_Rate, L2, Num_Epochs, Dropout_Rate):
	
# 	net = Cox_nnet(In_Nodes, Hidden_Nodes, Out_Nodes, Dropout_Rate)
# 	###if gpu is being used
# 	if torch.cuda.is_available():
# 		net.cuda()
# 	###
# 	###optimizer
# 	# opt = torch.optim.SGD(net.parameters(), lr=Learning_Rate, weight_decay = L2, momentum= 0.9)
# 	opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2)
# 	history_val=[]
# 	history_train=[]
# 	for epoch in range(Num_Epochs+1):
# 		#training phase
# 		pred_train=[]
# 		for batch in train_loader:  
# 			loss = net.training_step(batch)
# 			loss = loss['val_loss']
# 			loss.backward() ###calculate gradients
# 			opt.step() ###update weights and biases
# 			opt.zero_grad() ###reset gradients to zeros
# 			pred_train.append(net.training_step(batch))
# 		result_train = net.training_epoch_end(pred_train)
# 		pred_val = 	[net.validation_step(batch) for batch in val_loader]
# 		result_val = net.validation_epoch_end(pred_val)
# 		net.epoch_end(epoch, result_val)
# 		history_val.append(result_val)
# 		history_train.append(result_train)
# 		# print("epoch", epoch, "Loss in Train: ", train_loss,"Loss in val", eval_loss)
# 		# print("epoch", epoch, "C index in Train: ", train_cindex,"C index in val", eval_cindex)
# 	# pred_final = net(train_x, train_age)
# 	return (history_train, history_val)