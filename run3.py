from env import environment
from RL_brain import DuelingDQN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import algorithm_naive_1 as naive

env = environment(number_of_sbs=9)  # 定义使用 gym 库中的那一个环境
env_naive = naive.environment(bs_list=env.bs_list)
number = env.number_of_sbs
RL = DuelingDQN(n_actions=2 ** number,
                n_features=4 * number + 2,
                learning_rate=0.01, e_greedy=0.9,
                replace_target_iter=100, memory_size=2000,
                e_greedy_increment=0.0008,
                output_graph=True)
total_steps = 0  # 记录步数
a = pd.DataFrame(columns=['energy_cost'])
plt.figure()
plt.ion()
plt.show()
ep_r_total = []
count_time = 0
energy = []
EE_rate_total = np.zeros(50)
EE_rate_mean = []
counter = 0
mean_min = 10
min_index = 0
for i_episode in range(1000):
    print('iteration is %d' % i_episode)
    # 获取回合 i_episode 第一个 observation
    observation = env.reset()
    observation_naive = env_naive.reset(bs_list=env.bs_list)
    ep_r = 0

    while count_time <= 1000000000:
        count_time += 1
        action = RL.choose_action(observation)
        action_naive = env_naive.choose_action()
        observation_, energy_cost, energy_cost_max, temp = env.step(action, beta=-10, lambda_=0.5)  # 获取下一个 state
        # print(observation_)
        observation_naive_, energy_cost_naive, energy_cost_max_naive = env_naive.step(action_naive, temp, beta=-10)
        index_ob = np.zeros(number)
        for i in range(number):
            index_ob[i] = i * 4 + 1
        r1 = []
        for i in index_ob:
            r1.append(observation_[int(i)])

        r1 = pd.DataFrame(r1)
        # print(r1)
        r1 = (r1 - 500) / 500.0
        r1[r1.iloc[:, 0] != -1.0] = 0
        r1[r1.iloc[:, 0] == -1.0] = -50

        r1 = np.sum(r1)[0] + 1
        # print('r1',r1)
        r2 = (1 - (energy_cost / energy_cost_naive)) * 50.0
        # print(r2)
        # print('r2',r2)
        reward = r1 + r2
        ep_r += 1
        if count_time > 150000:
            if count_time % 10 == 0:
                energy.append(energy_cost / energy_cost_naive)
        if r1 <= -50:
            print(ep_r)
            ep_r_total.append(ep_r)
            break

        EE_rate = energy_cost / energy_cost_naive
        index = count_time % 50
        EE_rate_total[index] = EE_rate
        if index == 0:
            if len(EE_rate_mean) > 49:
                EE_rate_mean.pop(0)
                EE_rate_mean.append(EE_rate_total.mean())
            else:
                EE_rate_mean.append(EE_rate_total.mean())
        mean = np.array(EE_rate_mean).mean()
        if count_time >= 3000:
            if mean < mean_min:
                mean_min = mean
                min_index = count_time
        # print(EE_rate_mean)

        print(observation_[0:4 * number:4], '   ', observation_naive_[0:4 * number:4], energy_cost / energy_cost_naive)
        # print(observation_[2],observation_[6],observation_[10],observation_[13])

        b = pd.DataFrame([[float(energy_cost / energy_cost_max)]], columns=['energy_cost'])
        a = a.append(b, ignore_index=True)
        # 保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 100:
            RL.learn()  # 学习
        observation = observation_
        observation_naive = observation_naive_
        total_steps += 1
        if count_time % 50 == 0:
            plt.ylim((0.8, 1.2))
            plt.clf()
            plt.title('run3' + str(np.array(EE_rate_mean).mean()) + 'min ' + str(mean_min) + ' index ' + str(min_index))
            print(EE_rate_mean)
            plt.plot(EE_rate_mean)
            plt.pause(0.3)


            # if total_steps<=3:

            # print('env.user_list----------------------------')
            # print(env.user_list)
            # print('env_naive.user_list----------------------')
            # print(env_naive.user_list)
            #
            # print(ep_r_total)
            # plt.sca(ax1)
            # plt.plot(ep_r_total)
            # plt.sca(ax2)
            # plt.plot(energy)
            #
            # plt.show()
            # plt.plot(ep_r_total)
            # plt.draw()
            # plt.pause(0.1)
# 最后输出 cost 曲线
