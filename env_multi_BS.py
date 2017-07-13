import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class environment:
    def __init__(self,
                 M_trans=100,  # 43dbm
                 b_th_max=1000.0,
                 b_th_min=9.0,
                 Noise=7.96214341106997e-14,
                 P_TX=20.0,  # 13dbm
                 sinr_load_th=0,
                 LOAD_TH=100,
                 MBS_load_th=1000,
                 Alpha=0.1,
                 M_fixed_power=500.0
                 ):
        self.MBS_transmission_power = M_trans
        self.MBS_fixed_power = M_fixed_power
        self.battery_th_max = b_th_max
        self.battery_th_min = b_th_min
        self.N_0 = Noise
        self.P_tx = P_TX
        self.SINR_plus_load_th = sinr_load_th
        self.load_th = LOAD_TH
        self.mbs_load_th = MBS_load_th
        self.bs_list = pd.DataFrame(
            [[1.4, 2.5, 1, 1000.0, 0.0], [5.0, 7.2, 1, 1000.0, 0.0], [7.2, 4.2, 1, 1000.0, 0.0]])
        self.number_of_sbs = len(self.bs_list)
        self.mbs_load = 0.0
        self.user_list = pd.DataFrame(columns=['x', 'y', 'access time', 'selected_BS'])
        self.alpha = Alpha
        self.system_time = 0
        self.user_number = 0.01
        self.sbs_operation_power = 90
        self.mbs_operation_power = 500


    def init_BS(self,number_of_sbs):
        a = []
        for i in range(number_of_sbs):
            temp = list(10 * np.random.random(size=2))
            temp.append(1)
            temp.append(1000.0)
            temp.append(0.0)
            a.append(temp)
        return pd.DataFrame(a)

    def compute_SINR(self, x, beta=-2):
        h = stats.expon.rvs(size=3)

        SINR_list = []
        Signal_sum = 0
        for i in range(3):
            if self.bs_list.ix[i, 2] == 1:
                Signal_sum += (np.sqrt(
                    np.sum(np.square(np.array(self.bs_list.ix[i, [0, 1]]) - np.array(x)), axis=1)) * 100) ** (beta) * h[
                                  i] * self.P_tx

        for i in range(3):
            if self.bs_list.ix[i, 2] == 1:
                dist = np.sqrt(np.sum(np.square(np.array(self.bs_list.ix[i, [0, 1]]) - x), axis=1))
                mu = (dist * 100) ** (beta)
                SINR_list.append(self.P_tx * mu * h[i] / (Signal_sum - self.P_tx * mu * h[i] + self.N_0))
            else:
                SINR_list.append(np.zeros(np.shape(x)[0]))
        SINR_mbs = ((np.sqrt(np.sum(np.square(np.array([5, 5]) - np.array(x)), axis=1)) * 100) ** (beta) * (
        stats.expon.rvs()) * self.MBS_transmission_power) / self.N_0
        return np.array(SINR_list), SINR_mbs

    def user_association(self, x, beta):

        SINR_list_temp = self.compute_SINR(x, beta)
        SINR_list = np.array([SINR_list_temp[0][0], SINR_list_temp[0][1], SINR_list_temp[0][2], SINR_list_temp[1]])
        index = np.argmax(SINR_list, axis=0)
        for i in range(np.shape(index)[0]):
            if index[i] == 3:
                if self.mbs_load < self.mbs_load_th:
                    index[i] = -1
                else:
                    index[i] = -2
            else:
                if self.bs_list.ix[index[i], 4] > self.load_th:
                    index[i] = -2
        return index

    def get_user_number_per_bs(self):
        for i in range(self.number_of_sbs):
            temp = np.shape(self.user_list[self.user_list.ix[:, 3] == i])[0]
            if temp == 0: temp = 0.001
            self.bs_list.ix[i, 4] = temp
        self.mbs_load = np.shape(self.user_list[self.user_list.ix[:, 3] == -1])[0]
        self.user_number = np.shape(self.user_list)[0]

    def get_user_block_rate(self):
        block_rate = np.shape(self.user_list[self.user_list.ix[:, 3] == -2])[0] / float(np.shape(self.user_list)[0])
        return block_rate

    def get_energy_cost(self):
        energy_cost = (self.mbs_load * 10.0 + self.MBS_fixed_power + int(
            np.sum(self.bs_list.ix[:, 2])) * self.sbs_operation_power) / self.user_number
        energy_cost_max=(self.user_number*10.0+self.MBS_fixed_power+3*self.sbs_operation_power)/self.user_number
        return energy_cost,energy_cost_max

    def energy(self, t):
        return 75 * np.sin(float(t/8.0)) + 75

    def get_battery(self, t):
        user_list_temp = self.user_list
        for i in range(self.number_of_sbs):
            #             print('bs_list%d battery' %i,self.bs_list.ix[i,:])
            temp = np.minimum((self.bs_list.ix[i, 3] - self.sbs_operation_power*self.bs_list.ix[i,2] + self.energy(t)),
                              self.battery_th_max)
            #             print('%d s temp is'%i,temp)
            if (temp > 0):
                self.bs_list.ix[i, 3] = temp
            else:
                self.bs_list.ix[i, 3] = 0.0
                if self.bs_list.ix[i, 2] == 1:
                    transfer_index = np.array(user_list_temp[user_list_temp.ix[:, 3] == i].index)
                    if len(transfer_index) < (self.mbs_load_th - self.mbs_load):
                        for j in transfer_index:
                            user_list_temp.ix[j, 3] = -1
                    else:
                        counter = 0
                        for j in transfer_index:
                            if counter < (self.mbs_load_th - self.mbs_load):
                                user_list_temp.ix[j, 3] = -1
                            else:
                                user_list_temp.ix[j, 3] = -2
                            counter += 1
                            #                     user_list_temp.loc[lambda user_list_temp: self.user_list.ix[:,3]==i, 3]=-1
                    index_temp = np.array(user_list_temp[user_list_temp.ix[:, 3] == i].index)
                    for i in index_temp:
                        user_list_temp.ix[i, 3] = -1
                    self.bs_list.ix[i, 2] = 0
                    self.bs_list.ix[i, 4] = 0.001
                else:
                    print('error')
        self.user_list = user_list_temp

    def change_bs_on(self, a):
        user_list_temp = self.user_list
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
        action = actions[a]
        for i in range(self.number_of_sbs):
            if action[i] == 1:
                self.bs_list.ix[i, 2] = 1
            else:
                if self.bs_list.ix[i, 2] == 1:
                    transfer_index = np.array(user_list_temp[user_list_temp.ix[:, 3] == i].index)
                    if len(transfer_index) < (self.mbs_load_th - self.mbs_load):
                        for j in transfer_index:
                            user_list_temp.ix[j, 3] = -1
                    else:
                        counter = 0
                        for j in transfer_index:
                            if counter < (self.mbs_load_th - self.mbs_load):
                                user_list_temp.ix[j, 3] = -1
                            else:
                                user_list_temp.ix[j, 3] = -2
                            counter += 1
                            #                     user_list_temp.loc[lambda user_list_temp: user_list_temp.ix[:,3]==i, 3]=-1
                    index_temp = np.array(user_list_temp[user_list_temp.ix[:, 3] == i].index)
                    for i in index_temp:
                        user_list_temp.ix[i, 3] = -1
                self.bs_list.ix[i, 2] = 0
                self.bs_list.ix[i, 4] = 0.001
        self.user_list = user_list_temp
        self.get_user_number_per_bs()

    def user_coming(self, beta=-2,lambda_=0.2,mu=0.5):
        temp = []
        for i in range(10):
            for j in range(10):
                arrival_number = stats.poisson.rvs(lambda_, loc=0)
                for k in range(arrival_number):
                    temp.append([i, j])
        temp = pd.DataFrame(temp, columns=['x', 'y'])
        temp = np.random.rand(np.shape(temp)[0], np.shape(temp)[1]) + temp
        access_time = stats.expon.rvs(scale=1 /mu, size=np.shape(temp)[0])
        temp['access time'] = access_time
        selected_bs = self.user_association(np.array(temp.ix[:, ['x', 'y']]), beta)
        temp['selected_BS'] = selected_bs
        self.user_list = self.user_list.append(temp)
        self.get_user_number_per_bs()
        #         for i in range(10):
        #             for j in range(10):
        #                 arrival_number = stats.poisson.rvs(mu=2, loc=0, size=1)[0]
        #                 for k in range(arrival_number):
        #                     # 添加用户到用户列表
        #                     temp = pd.DataFrame(
        #                         [[i, j, stats.expon.rvs(scale=1 / float(0.5), size=1)[0], self.user_association([i, j],beta)]],
        #                         columns=['x', 'y', 'access time', 'selected_BS'])
        #                     self.user_list = self.user_list.append(temp, ignore_index=True)
        #                     self.get_user_number_per_bs()
        self.system_time += 1
    def compute_distance(self,x,y):
        x=np.array(x)
        y=np.array(y)
        distance=np.linalg.norm(x-y)
        distance=np.maximum(distance,0.1)
        return distance
    def user_coming1(self, beta=-2,lambda_=0.2,mu=0.5):
        temp = []
        for i in range(10):
            for j in range(10):
                distance=self.compute_distance([i,j],[7,4])
                arrival_number = stats.poisson.rvs(lambda_*(float(1/distance)), loc=0)
                for k in range(arrival_number):
                    temp.append([i, j])
        temp = pd.DataFrame(temp, columns=['x', 'y'])
        temp = np.random.rand(np.shape(temp)[0], np.shape(temp)[1]) + temp
        access_time = stats.expon.rvs(scale=1 /mu, size=np.shape(temp)[0])
        temp['access time'] = access_time
        selected_bs = self.user_association(np.array(temp.ix[:, ['x', 'y']]), beta)
        temp['selected_BS'] = selected_bs
        self.user_list = self.user_list.append(temp)
        self.get_user_number_per_bs()
        #         for i in range(10):
        #             for j in range(10):
        #                 arrival_number = stats.poisson.rvs(mu=2, loc=0, size=1)[0]
        #                 for k in range(arrival_number):
        #                     # 添加用户到用户列表
        #                     temp = pd.DataFrame(
        #                         [[i, j, stats.expon.rvs(scale=1 / float(0.5), size=1)[0], self.user_association([i, j],beta)]],
        #                         columns=['x', 'y', 'access time', 'selected_BS'])
        #                     self.user_list = self.user_list.append(temp, ignore_index=True)
        #                     self.get_user_number_per_bs()
        self.system_time += 1

    def user_leaving(self):
        self.user_list.ix[:, 2] -= 1
        self.user_list = self.user_list[self.user_list.ix[:, 2] >= 0.0]
        self.user_list = self.user_list[self.user_list.ix[:, 3] > -2.0]
        self.user_list = pd.DataFrame(np.array(self.user_list), columns=['x', 'y', 'access time', 'selected_BS'])
        self.get_user_number_per_bs()

    def reset(self):
        self.mbs_load = 0.0
        self.bs_list = pd.DataFrame(
            [[1.4, 2.5, 1, 1000.0, 0.0], [5.0, 7.2, 1, 1000.0, 0.0], [7.2, 4.2, 1, 1000.0, 0.0]])
        self.user_list = pd.DataFrame(columns=['x', 'y', 'access time', 'selected_BS'])
        self.system_time = 0
        self.user_number = 0
        observation = [self.bs_list.ix[0, 2], self.bs_list.ix[0, 3], self.bs_list.ix[0, 4], self.energy(1),
                       self.bs_list.ix[1, 2], self.bs_list.ix[1, 3], self.bs_list.ix[1, 4], self.energy(1),
                       self.bs_list.ix[2, 2], self.bs_list.ix[2, 3], self.bs_list.ix[2, 4], self.energy(1),
                       self.system_time, self.mbs_load]
        return np.array(observation)

    def step(self, action, beta=-8):
        self.change_bs_on(action)
        self.user_coming1(beta)
        block_rate = self.get_user_block_rate()
        energy_cost,energy_cost_max = self.get_energy_cost()
        self.get_battery(self.system_time)
        self.user_leaving()
        observation = [self.bs_list.ix[0, 2], self.bs_list.ix[0, 3], self.bs_list.ix[0, 4],
                       self.energy(self.system_time + 1),
                       self.bs_list.ix[1, 2], self.bs_list.ix[1, 3], self.bs_list.ix[1, 4],
                       self.energy(self.system_time + 1),
                       self.bs_list.ix[2, 2], self.bs_list.ix[2, 3], self.bs_list.ix[2, 4],
                       self.energy(self.system_time + 1),
                       self.system_time, self.mbs_load]
        return np.array(observation), energy_cost,energy_cost_max


