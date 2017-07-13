import pandas as pd
import numpy as np
import tensorflow as tf


class DuelingDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
            sess=None,
            number=3
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling  # decide to use dueling DQN or not

        self.learn_step_counter = 0
        self.BS_number = number
        self.memory = np.zeros((self.memory_size, ((n_features - 2) * number +2)* 2 + 2))
        self._build_net()
        if sess is None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
            #         self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2
            print('build %s' % c_names)
            return out

        # ------------------ build evaluate_net ------------------
        self.s = []
        self.q_target = []
        self.q_eval = []
        self.loss = []
        self._train_op = []
        self.s_ = []
        self.q_next = []

        for i in range(self.BS_number):
            self.s.append(tf.placeholder(tf.float32, [None, self.n_features], name='s' + str(i)))
            self.q_target.append(tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target' + str(i)))
        # self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        #         self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        for i in range(self.BS_number):
            print(i)
            with tf.variable_scope('eval_net' + str(i)):
                c_names, n_l1, w_initializer, b_initializer = \
                    ['eval_net_params' + str(i), tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

                self.q_eval.append(build_layers(self.s[i], c_names, n_l1, w_initializer, b_initializer))

            with tf.variable_scope('loss' + str(i)):
                self.loss.append(tf.reduce_mean(tf.squared_difference(self.q_target[i], self.q_eval[i])))
        for i in range(self.BS_number):
            with tf.variable_scope('train' + str(i)):
                self._train_op.append(tf.train.RMSPropOptimizer(self.lr).minimize(self.loss[i]))
        print('build evaluate net succeed')

        # ------------------ build target_net ------------------
        for i in range(self.BS_number):
            self.s_.append(tf.placeholder(tf.float32, [None, self.n_features], name='s_' + str(i)))
            # input
        for i in range(self.BS_number):
            with tf.variable_scope('target_net' + str(i)):
                c_names = ['target_net_params' + str(i), tf.GraphKeys.GLOBAL_VARIABLES]
                print(w_initializer)

                self.q_next.append(build_layers(self.s_[i], c_names, n_l1, w_initializer, b_initializer))




                #####以上是在建立N个神经网络

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        action=np.zeros(self.BS_number)
        for i in range(self.BS_number):
            index = list(np.arange(i * (self.n_features - 2), (i + 1) * (self.n_features - 2)))
            index.append(self.BS_number * (self.n_features - 2))
            index.append(self.BS_number * (self.n_features - 2) + 1)
            observation_temp = observation[index]
            observation_temp = observation_temp[np.newaxis, :]
            if np.random.uniform() < self.epsilon:  # choosing action
                actions_value = self.sess.run(self.q_eval[i], feed_dict={self.s[i]: observation_temp})
                action[i] = np.argmax(actions_value)
            else:
                action[i] = np.random.randint(0, self.n_actions)
        action_count = ''
        for i in range(self.BS_number):
            action_count += str(int(action[i]))
        # print (action_count)
        action_count = int(action_count, 2)
        return action_count

    def _replace_target_params(self):
        for i in range(self.BS_number):
            t_params = tf.get_collection('target_net_params' + str(i))
            e_params = tf.get_collection('eval_net_params' + str(i))
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')

        for i in range(self.BS_number):
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
            index_ = list(np.arange((self.BS_number + i) * (self.n_features - 2) + 4,
                                    (self.BS_number + i + 1) * (self.n_features - 2) + 4))
            index_.append(-2)
            index_.append(-1)
            index_=np.array(index_)

            # feature=6
            # y = np.arange((((feature-2)*3+2)*2+2)*10).reshape(10,(((feature-2)*3+2)*2+2))
            # for i in range(3):
            #     print('bs',i)
            #     index=list(np.arange((3+i)*(feature-2)+4,(3+i)*(feature-2)+4+4))
            #     index.append(-2)
            #     index.append(-1)
            #     print(y[:,index])
            observation=batch_memory[:, index_]
            q_next, q_eval4next, = self.sess.run(
                [self.q_next[i], self.q_eval[i]],
                feed_dict={self.s_[i]: observation,  # next observation
                           self.s[i]: observation})  # next observation
            index = list(np.arange(i * (self.n_features - 2), (i + 1) * (self.n_features - 2)))
            index.append(self.BS_number * (self.n_features  - 2))
            index.append(self.BS_number * (self.n_features - 2) + 1)
            index = np.array(index)
            #     index=list(np.arange(i*(feature-2),(i+1)*(feature-2)))
            #     index.append(3*(feature-2))
            #     index.append(3*(feature-2)+1)
            q_eval = self.sess.run(self.q_eval[i], {self.s[i]: batch_memory[:, index]})

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, (self.n_features - 2) * self.BS_number * 2].astype(int)
            reward = batch_memory[:, (self.n_features - 2) * self.BS_number * 2 + 1]

            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

            _, self.cost = self.sess.run([self._train_op[i], self.loss[i]],
                                         feed_dict={self.s[i]: batch_memory[:, index],
                                                    self.q_target[i]: q_target})
        # self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


