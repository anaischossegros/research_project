from Data_Loader import load_data
from Train import trainCox_nnet

import torch
import numpy as np
import pandas as pd

dtype = torch.FloatTensor
''' Net Settings'''
In_Nodes = 55553 ###number of genes
Hidden_Nodes = 100 ###number of hidden nodes
Out_Nodes = 30 ###number of hidden nodes in the last hidden layer
''' Initialize '''
Initial_Learning_Rate = [0.03, 0.01, 0.001, 0.00075]
L2_Lambda = [0.1, 0.01, 0.005, 0.001]
num_epochs = 300 ###for grid search
Num_EPOCHS = 2000 ###for training
###sub-network setup
Dropout_Rate = [0.7]

from Data_extraction import output_df2, x_df2  

data = pd.concat([x_df2,output_df2],axis=1)

# data = pd.concat([x_df2,output_df2],axis=1)

# def split_indices(n, val_pct1, val_pct2):
# 	n1 = int(val_pct1*n)
# 	n2 = int(val_pct2*n)
# 	idxs = np.random.permutation(n)
# 	return idxs[:n1], idxs[n1:n2], idxs[n2:]

# train_index, val_index, test_index = split_indices(len(data),0.6, 0.8)
# data_train = data.iloc[train_index]
# data_val = data.iloc[val_index]
# data_test = data.iloc[test_index]

# x_train, ytime_train, yevent_train, age_train = load_data(data_train, dtype)
# x_valid, ytime_valid, yevent_valid, age_valid = load_data(data_val, dtype)
# x_test, ytime_test, yevent_test, age_test = load_data(data_test,dtype)

x, ytime, yevent, age = load_data(data, dtype)

x_train, ytime_train, yevent_train, age_train = x[0:107], ytime[0:107], yevent[0:107], age[0:107]
x_valid, ytime_valid, yevent_valid, age_valid = x[107:142], ytime[107:142], yevent[107:142], age[107:142]
x_test, ytime_test, yevent_test, age_test = x[142:len(ytime)], ytime[142:len(ytime)], yevent[142:len(ytime)], age[142:len(ytime)]
opt_l2_loss = 0
opt_lr_loss = 0
opt_loss = torch.Tensor([float("Inf")])
###if gpu is being used
if torch.cuda.is_available():
	opt_loss = opt_loss.cuda()
###
opt_c_index_va = 0
opt_c_index_tr = 0

for l2 in L2_Lambda:
	for lr in Initial_Learning_Rate:
		loss_train, loss_valid, c_index_tr, c_index_va = trainCox_nnet(x_train, age_train, ytime_train, yevent_train, \
																x_valid, age_valid, ytime_valid, yevent_valid, \
																In_Nodes, Hidden_Nodes, Out_Nodes, \
																lr, l2, num_epochs, Dropout_Rate)
		if loss_valid < opt_loss:
			opt_l2_loss = l2
			opt_loss = loss_valid
			opt_c_index_tr = c_index_tr
			opt_c_index_va = c_index_va
		print ("L2: ", l2, "LR: ", lr, "Loss in Validation: ", loss_valid)

        ###train Cox-PASNet with optimal hyperparameters using train data, and then evaluate the trained model with test data
###Note that test data are only used to evaluate the trained Cox-nnet
loss_train, loss_test, c_index_tr, c_index_te = trainCox_nnet(x_train, age_train, ytime_train, yevent_train, \
							x_test, age_test, ytime_test, yevent_test,  \
							In_Nodes, Hidden_Nodes, Out_Nodes, \
							opt_lr_loss, opt_l2_loss, Num_EPOCHS, Dropout_Rate)
print ("Optimal L2: ", opt_l2_loss, "Optimal LR: ", opt_lr_loss)
print("C-index in Test: ", c_index_te)
