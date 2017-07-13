from scipy import stats
import numpy as np
def compute_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    distance = np.linalg.norm(x - y)
    distance = np.maximum(distance, 0.1)
    return distance
def choose_action(battery):

    action=[0,0,0]
    for i in range(len(battery)):
        if battery[i]<100:
            action[i]=0
        if battery[i]>800:
            action[i]=1
    action_out=str(action[0])+str(action[1])+str(action[2])
    action_out=int(action_out,2)
    return action_out
def turn_str_action_into_list(action):
    actions = {
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 1, 0],
        3: [0, 1, 1],
        4: [1, 0, 0],
        5: [1, 0, 1],
        6: [1, 1, 0],
        7: [1, 1, 1]
    }
    bs_on_or_off=actions[action]
    return bs_on_or_off

def compute_SINR(bs_location, action,x,beta=-2):

    bs_on_or_off=turn_str_action_into_list(action)
    h = stats.expon.rvs(size=3)
    P_tx = 20.0
    N_0=7.96214341106997e-14
    MBS_transmission_power=100
    SINR_list = []
    Signal_sum = 0
    for i in range(3):
        if bs_on_or_off[i] == 1:
            Signal_sum += (np.sqrt(
                np.sum(np.square(np.array(bs_location[i]) - np.array(x)), axis=1)) * 100) ** (beta) * h[
                                  i] * P_tx

    for i in range(3):
        if bs_on_or_off[i] == 1:
            dist = np.sqrt(np.sum(np.square(np.array(bs_location[i]) - x), axis=1))
            mu = (dist * 100) ** (beta)
            SINR_list.append(P_tx * mu * h[i] / (Signal_sum - P_tx * mu * h[i] + N_0))
        else:
            SINR_list.append(np.zeros(np.shape(x)[0]))
    SINR_mbs = ((np.sqrt(np.sum(np.square(np.array([5, 5]) - np.array(x)), axis=1)) * 100) ** (beta) * (
    stats.expon.rvs()) * MBS_transmission_power) / N_0
    return np.array(SINR_list), SINR_mbs


def user_association(bs_location, bs_on_or_off,x, mbs_load,mbs_load_th,sbs_load,sbs_load_th,beta=-2):
    SINR_list_temp = compute_SINR(bs_location, bs_on_or_off,x,beta)
    SINR_list = np.array([SINR_list_temp[0][0], SINR_list_temp[0][1], SINR_list_temp[0][2], SINR_list_temp[1]])
    index = np.argmax(SINR_list, axis=0)
    for i in range(np.shape(index)[0]):
        if index[i] == 3:
            if mbs_load < mbs_load_th:
                index[i] = -1
            else:
                index[i] = -2
        else:
            if sbs_load[index[i]] > sbs_load_th:
                index[i] = -2
    return index
def energy(t):
    return 75 * np.sin(float(t/8.0)) + 75
def get_battery(self,battery,user_list, t,mbs_load,sbs_load):
    sbs_operation_power=90
    user_list_temp=user_list
    battery_th_max=1000.0
    mbs_load_th=1000
    for i in range(3):
            #             print('bs_list%d battery' %i,self.bs_list.ix[i,:])
        temp = np.minimum((battery[i] - sbs_operation_power*self.bs_list.iloc[i,2] + energy(t)),battery_th_max)
            #             print('%d s temp is'%i,temp)
        if (temp > 0):
            battery[i]= temp
        else:
            battery[i] = 0.0
            if action[i] == 1:
                transfer_index = np.array(user_list_temp[user_list_temp.ix[:, 3] == i].index)
                if len(transfer_index) < (mbs_load_th - mbs_load):
                    for j in transfer_index:
                        user_list_temp.ix[j, 3] = -1
                else:
                    counter = 0
                    for j in transfer_index:
                        if counter < (mbs_load_th - mbs_load):
                            user_list_temp.ix[j, 3] = -1
                        else:
                            user_list_temp.ix[j, 3] = -2
                        counter += 1
                            #                     user_list_temp.loc[lambda user_list_temp: self.user_list.ix[:,3]==i, 3]=-1
                index_temp = np.array(user_list_temp[user_list_temp.ix[:, 3] == i].index)
                for i in index_temp:
                    user_list_temp.ix[i, 3] = -1
                action[i]=0
                sbs_load[i] = 0.001
            else:
                print('error')
    user_list=user_list_temp
    return battery,user_list,action,mbs_load,sbs_load


