from Model2 import Cox_nnet
from SubNetwork_SparseCoding import dropout_mask, s_mask
from Survival_CostFunc_CIndex import R_set, neg_par_log_likelihood, c_index

import torch
import torch.optim as optim
import copy
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
dtype = torch.FloatTensor

def trainCox_nnet(train_loader, \
			val_loader, \
			In_Nodes, Hidden_Nodes, Out_Nodes, \
			Learning_Rate, L2, Num_Epochs, Dropout_Rate):
	
	net = Cox_nnet(In_Nodes, Hidden_Nodes, Out_Nodes, Dropout_Rate)
	###if gpu is being used
	if torch.cuda.is_available():
		net.cuda()
	###
	###optimizer
	# opt = torch.optim.SGD(net.parameters(), lr=Learning_Rate, weight_decay = L2, momentum= 0.9)
	opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2)
	history_val=[]
	history_train=[]
	for epoch in range(Num_Epochs+1):
		#training phase
		pred_train=[]
		for batch in train_loader:  
			loss = net.training_step(batch)
			loss = loss['val_loss']
			loss.backward() ###calculate gradients
			opt.step() ###update weights and biases
			opt.zero_grad() ###reset gradients to zeros
			pred_train.append(net.training_step(batch))
		result_train = net.training_epoch_end(pred_train)
		pred_val = 	[net.validation_step(batch) for batch in val_loader]
		result_val = net.validation_epoch_end(pred_val)
		net.epoch_end(epoch, result_val)
		history_val.append(result_val)
		history_train.append(result_train)
		# print("epoch", epoch, "Loss in Train: ", train_loss,"Loss in val", eval_loss)
		# print("epoch", epoch, "C index in Train: ", train_cindex,"C index in val", eval_cindex)
	# pred_final = net(train_x, train_age)
	return (history_train, history_val)