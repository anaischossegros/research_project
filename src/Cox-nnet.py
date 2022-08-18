from Data_Loader import load_data
from Train import trainCox_nnet

import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


dtype = torch.FloatTensor
''' Net Settings'''
Hidden_Nodes = 143 ###number of hidden nodes
Out_Nodes = 30 ###number of hidden nodes in the last hidden layer
''' Initialize '''
Initial_Learning_Rate = [0.03, 0.01, 0.001, 0.00075]
L2_Lambda = [0.1, 0.01, 0.005, 0.001]
L1_lambda= [0.1, 0.01, 0.005, 0.001]
num_epochs = 3 ###for grid search
Num_EPOCHS = 8 ###for training
Dropout_Rate = [0.7, 0.5]

from Data_extraction_lung import data_norm_df_lung, output_df_lung
data_norm_df= data_norm_df_lung.reset_index(drop=True)
output_df2 = output_df_lung.reset_index(drop=True)


data = pd.concat([data_norm_df,output_df2], axis=1)

def split_indices(n, val_pct1, val_pct2):
	n1 = int(val_pct1*n)
	n2 = int(val_pct2*n)
	idxs = np.random.permutation(n)
	train, val, test = idxs[:n1], idxs[n1:n2], idxs[n2:]
	train.sort(), val.sort(), test.sort()
	return(train, val, test)
x, ytime, yevent, age = load_data(data, dtype)

x = StandardScaler().fit_transform(x)
pca = PCA(.95)
pca.fit(x)
x = pca.transform(x)
x= torch.from_numpy(x)

from Data_Loader import CustomDataset
batch_size=16
data2 = CustomDataset(x, ytime, yevent, age)
In_Nodes = len(x[0,:]) ###number of genes

opt_l2_loss = 0
opt_lr_loss = 0
opt_do_loss = 0
opt_loss = torch.Tensor([float("Inf")])
###if gpu is being used
if torch.cuda.is_available():
	opt_loss = opt_loss.cuda()
###
opt_c_index_va = 0
opt_c_index_tr = 0

for l2 in L2_Lambda:
	for lr in Initial_Learning_Rate:
		for l1 in L1_lambda: 
			for do in Dropout_Rate:
				history_train, history_val = trainCox_nnet(data2, \
					In_Nodes, Hidden_Nodes, Out_Nodes, \
					lr, l2, l1, num_epochs, do, batch_size)
				loss_train2 = [k['loss'] for k in history_train[1]]
				if loss_train2[-1] =='nan': 
					break
				elif loss_train2[-1] < opt_loss:
					opt_l2_loss = l2
					opt_lr_loss = lr
					opt_do_loss = do
					opt_loss = loss_train2[-1]
					# opt_c_index_tr = c_index_tr
					# opt_c_index_va = c_index_va
				print ("L2: ", l2, "LR: ", lr, "Loss in Validation: ", opt_loss)

###train Cox-nnet with optimal hyperparameters using train data, and then evaluate the trained model with test data
###Note that test data are only used to evaluate the trained Cox-nnet
history_train, history_val = trainCox_nnet(data2, \
			In_Nodes, Hidden_Nodes, Out_Nodes, \
			opt_lr_loss, opt_l2_loss, Num_EPOCHS, opt_do_loss, batch_size)
print ("Optimal L2: ", opt_l2_loss, "Optimal LR: ", opt_lr_loss)