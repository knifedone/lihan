from env import environment
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#clf() # 清图  cla() # 清坐标轴 close() # 关窗口
plt.figure()
env1=environment(random=False)
plt.grid(True)
plt.xlim((0, 10))
plt.ylim((0, 10))
plt.ion()
plt.show()


for i in range(100):
    env1.user_coming1(beta=-8,lambda_=0.8)

    plt.xlim((0, 10))
    plt.ylim((0, 10))
    plt.scatter(env1.user_list[env1.user_list['selected_BS']==2].ix[:,0],env1.user_list[env1.user_list['selected_BS']==2].ix[:,1],label=2)
    plt.scatter(env1.user_list[env1.user_list['selected_BS']==1].ix[:,0],env1.user_list[env1.user_list['selected_BS']==1].ix[:,1],label=1)
    plt.scatter(env1.user_list[env1.user_list['selected_BS']==0].ix[:,0],env1.user_list[env1.user_list['selected_BS']==0].ix[:,1],label=0)
    plt.scatter(env1.user_list[env1.user_list['selected_BS']==-1].ix[:,0],env1.user_list[env1.user_list['selected_BS']==-1].ix[:,1],label=-1)
    plt.scatter(env1.bs_list.ix[0,0],env1.bs_list.ix[0,1],label=3,marker = 'x')
    plt.scatter(env1.bs_list.ix[1,0],env1.bs_list.ix[1,1],label=3,marker = 'x')
    plt.scatter(env1.bs_list.ix[2,0],env1.bs_list.ix[2,1],label=3,marker = 'x')
    env1.user_leaving()
    plt.pause(0.3)
    plt.clf()

    plt.xlim((0, 10))
    plt.ylim((0, 10))
    plt.scatter(env1.user_list[env1.user_list['selected_BS']==2].ix[:,0],env1.user_list[env1.user_list['selected_BS']==2].ix[:,1],label=2)
    plt.scatter(env1.user_list[env1.user_list['selected_BS']==1].ix[:,0],env1.user_list[env1.user_list['selected_BS']==1].ix[:,1],label=1)
    plt.scatter(env1.user_list[env1.user_list['selected_BS']==0].ix[:,0],env1.user_list[env1.user_list['selected_BS']==0].ix[:,1],label=0)
    plt.scatter(env1.user_list[env1.user_list['selected_BS']==-1].ix[:,0],env1.user_list[env1.user_list['selected_BS']==-1].ix[:,1],label=-1)
    plt.scatter(env1.bs_list.ix[0,0],env1.bs_list.ix[0,1],label=3,marker = 'x')
    plt.scatter(env1.bs_list.ix[1,0],env1.bs_list.ix[1,1],label=3,marker = 'x')
    plt.scatter(env1.bs_list.ix[2,0],env1.bs_list.ix[2,1],label=3,marker = 'x')
    plt.pause(0.3)
    plt.clf()


