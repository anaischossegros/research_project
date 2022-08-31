import numpy as np
import pandas as pd
import torch



#sort the data 
def sort_data(data): 
    """_summary_

    Args:
        data(data frame): data frame with inputs and outputs

    Returns:
        list: list of the sorted data by survival time: gene expression, survival time, censoring index, age
    """
    data['days_to_death']= np.where(data['days_to_death'] == "'--", data['days_to_last_follow_up'],data['days_to_death'])
    data['days_to_death']=data['days_to_death'].astype(float)
    data['age_at_diagnosis']=data['age_at_diagnosis'].astype(float)
    data['vital_status'] = np.where((data.vital_status=='Alive'), 0, 1) 
    data.sort_values(['days_to_death'], ascending = False, inplace = True)
    x = data.drop(['case_submitter_id','days_to_death','vital_status','age_at_diagnosis','days_to_last_follow_up'], axis = 1).values
    ytime = data.loc[:,['days_to_death']].values
    yevent = data.loc[:,['vital_status']].values
    age = data.loc[:,['age_at_diagnosis']].values
    return (x, ytime, yevent, age)


#load the data into tensors
def load_data(data, dtype): 
    x, ytime, yevent, age = sort_data(data)
    X = torch.from_numpy(x).type(dtype)
    YTIME = torch.from_numpy(ytime).type(dtype)
    YEVENT = torch.from_numpy(yevent).type(dtype)
    AGE = torch.from_numpy(age).type(dtype)
    ###if gpu is being used
    if torch.cuda.is_available():
        X = X.cuda()
        YTIME = YTIME.cuda()
        YEVENT = YEVENT.cuda()
        AGE = AGE.cuda()
	###
    return(X, YTIME, YEVENT, AGE)

#make a class with the data
class CustomDataset():
    def __init__(self, x, y_time, y_event, age, transform=None, target_transform=None):
        self.ytime = y_time
        self.yevent = y_event
        self.x = x
        self.age = age
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ytime)

    def __getitem__(self, idx):
        x = self.x[idx]
        age = self.age[idx]
        y_time = self.ytime[idx]
        y_event = self.yevent[idx]
        if self.transform:
            x = self.transform(x)
            age = self.transform(age)
        if self.target_transform:
            y_time = self.target_transform(y_time)
            y_event = self.target_transform(y_event)
        return x, y_time, y_event, age
