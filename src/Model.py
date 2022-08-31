import torch
import torch.nn as nn
from Survival_CostFunc_CIndex import neg_par_log_likelihood, c_index



class Cox_nnet(nn.Module):
    def __init__(self, In_Nodes, Hidden_Nodes, Out_Nodes, Dropout): 
        super(Cox_nnet, self).__init__()

		# # gene layer --> hidden layer 1
        self.sc1 = nn.Linear(In_Nodes, Hidden_Nodes)
        self.sc1_norm = nn.BatchNorm1d(Hidden_Nodes)
        self.sc1_do = nn.Dropout(Dropout) 
        self.tanh = nn.ReLU()

        #hiddenlayer 2 --> hidden layer 3 linear
        self.sc2 = nn.Linear(Hidden_Nodes, Out_Nodes)
        self.sc2_norm = nn.BatchNorm1d(Out_Nodes)
        self.sc2_do = nn.Dropout(Dropout) 
        self.tanh2 = nn.ReLU()

        #hiddenlayer 3 --> hidden layer 4 linear
        # self.sc3= nn.Linear(Out_Nodes, Out_Nodes, bias=False)
        # self.sc3_do = nn.Dropout(Dropout) 
        # self.sc3_norm = nn.BatchNorm1d(Out_Nodes)

		# hidden layer 2 + age --> Cox layer
        self.sc4 = nn.Linear(Out_Nodes+1, 1, bias=False)
        self.sc4.weight.data.uniform_(-0.001, 0.001)

        
    def forward(self, x_1, x_2):

        #Normal
        x_1 = self.tanh(self.sc1_do(self.sc1_norm(self.sc1(x_1))))
        x_1 = self.tanh2(self.sc2_do(self.sc2_norm(self.sc2(x_1))))
        # x_1 = self.tanh(self.sc3_do(self.sc3_norm(self.sc3(x_1))))
		
        # combine age with hidden layer 
        x_cat = torch.cat((x_1, x_2), 1)
        lin_pred = self.sc4(x_cat)
        return lin_pred

    def training_step(self, batch): 
        x_train_b, ytime_train_b, yevent_train_b, age_train_b = batch
        pred = self(x_train_b.float(), age_train_b) ###Forward
        loss = neg_par_log_likelihood(pred, ytime_train_b, yevent_train_b) ###calculate loss
        acc = c_index(pred, ytime_train_b, yevent_train_b) #calculate accuracy
        return{'val_loss': loss, 'val_acc': acc}

    def training_epoch_end(self, pred):
        batch_losses = [x['val_loss'] for x in pred]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in pred]
        epoch_acc = torch.stack(batch_accs).mean()
        return{'loss': epoch_loss.item(), 'c_index': epoch_acc.item()}

    def validation_step(self, batch): 
        x_eval_b, ytime_eval_b, yevent_eval_b, age_eval_b = batch
        eval_pred = self(x_eval_b.float(), age_eval_b)
        loss = neg_par_log_likelihood(eval_pred, ytime_eval_b, yevent_eval_b)
        acc = c_index(eval_pred, ytime_eval_b, yevent_eval_b)
        return{'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, pred): 
        batch_losses = [x['val_loss'] for x in pred]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in pred]
        epoch_acc = torch.stack(batch_accs).mean()
        return{'loss': epoch_loss.item(), 'c_index': epoch_acc.item()}
    
    def epoch_end(self, epoch, result): 
        print("Epoch [{}], loss: {:.4f}, c_index: {:.4f}".format(epoch, result['loss'], result['c_index']))