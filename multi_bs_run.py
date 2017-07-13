from env import environment
from RL_brain import DuelingDQN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

env = environment()  # 定义使用 gym 库中的那一个环境

RL1 = DuelingDQN(n_actions=2,
                n_features=6,
                learning_rate=0.01, e_greedy=0.9,
                replace_target_iter=100, memory_size=2000,
                e_greedy_increment=0.0008,
                output_graph=True)
RL2 = DuelingDQN(n_actions=2,
                n_features=6,
                learning_rate=0.01, e_greedy=0.9,
                replace_target_iter=100, memory_size=2000,
                e_greedy_increment=0.0008,
                output_graph=True)
RL3 = DuelingDQN(n_actions=2,
                n_features=6,
                learning_rate=0.01, e_greedy=0.9,
                replace_target_iter=100, memory_size=2000,
                e_greedy_increment=0.0008,
                output_graph=True)
# total_steps = 0  # 记录步数
# a = pd.DataFrame(columns=['energy_cost'])
# # plt.figure()
# # plt.ion()
# # plt.show()
# ep_r_total = []
# for i_episode in range(500):
#     print('iteration is %d' % i_episode)
#     # 获取回合 i_episode 第一个 observation
#     temp = env.reset()
#     observation=np.array([])
#     for i in range(3):
#         observation.append([temp[i*0],temp[i*1],temp[i*2],temp[i*3],temp[12],temp[13]])
#     ep_r = 0
#     while True:
#         for i in range(3):
#             action=RL.choose_action()
#         action = RL.choose_action(observation)  # 选行为
#
#         observation_, energy_cost, energy_cost_max = env.step(action, beta=-8)  # 获取下一个 state
#         r1 = np.array([observation_[1], observation_[5], observation_[9]])
#
#         r1 = (r1 - 500) / 500.0
#         r1[r1 == -1.0] = -10
#         # print(r1)
#         r1 = np.sum(r1)
#         # print('r1',r1)
#         r2 = (1 - (energy_cost / energy_cost_max)) * 10.0
#         # print('r2',r2)
#         reward = r1 + r2
#         ep_r += 1
#         if r1 < -5:
#             print(ep_r)
#             ep_r_total.append(ep_r)
#             break
#
#         # print(r1,r2)
#         print(observation_[0], observation_[4], observation_[8])
#
#         b = pd.DataFrame([[float(energy_cost / energy_cost_max)]], columns=['energy_cost'])
#         a = a.append(b, ignore_index=True)
#         # 保存这一组记忆
#         RL.store_transition(observation, action, reward, observation_)
#
#         if total_steps > 100:
#             RL.learn()  # 学习
#         observation = observation_
#         total_steps += 1
# print(ep_r_total)
# plt.plot(ep_r_total)
# plt.show()
# # plt.plot(ep_r_total)
# # plt.draw()
# # plt.pause(0.1)
# # 最后输出 cost 曲线
