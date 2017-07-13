from env import environment
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
env_=environment(random=False,MBS_load_th=100000,LOAD_TH=100000)
x=10.0*np.random.rand(100000,2)
user_list=pd.DataFrame(columns=['x', 'y'],data=x)
user_list['access time']=np.random.rand(100000)
user_list['selected_BS']=env_.user_association(user_list.iloc[:,[0,1]],beta=-5)
# print(user_list[user_list['selected_BS'] == 2].iloc[:,0],)
plt.scatter(user_list[user_list['selected_BS'] == 0].iloc[:,0],user_list[user_list['selected_BS'] == 0].iloc[:,1],label='1')
plt.scatter(user_list[user_list['selected_BS'] == 1].iloc[:,0],user_list[user_list['selected_BS'] == 1].iloc[:,1],)
plt.scatter(user_list[user_list['selected_BS'] == 2].iloc[:,0],user_list[user_list['selected_BS'] == 2].iloc[:,1],)
plt.scatter(user_list[user_list['selected_BS'] == -1].iloc[:,0],user_list[user_list['selected_BS'] == -1].iloc[:,1],)
plt.show()


