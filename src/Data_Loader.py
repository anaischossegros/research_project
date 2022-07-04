import numpy as np
import pandas as pd
import torch

#extract the data from data extraction
# from Data_extraction import output_df2, x_df2 
# data=pd.concat([x_df2,output_df2],axis=1)

#sort the data 
def sort_data(data): 
    data['days_to_last_follow_up']=np.where(data['days_to_last_follow_up']=="'--", 1000, data['days_to_last_follow_up'])
    data['days_to_death']= np.where(data['days_to_death'] == "'--", data['days_to_last_follow_up'],data['days_to_death'])
    data['days_to_death']=data['days_to_death'].astype(float)
    data['age_at_diagnosis']=data['age_at_diagnosis'].astype(float)
    data['vital_status'] = np.where((data.vital_status=='Alive') & (data.days_to_last_follow_up.astype(float)<240), 0, 1) 
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
    return(X, YTIME, YEVENT, AGE)