import datetime
import os
import warnings
import threading
from collections import namedtuple
from queue import Queue
import numpy as np
import qutip
import tensorflow as tf
from scipy.sparse import dia_matrix
from tensorflow import keras
from tensorflow.keras import layers, optimizers  # , losses

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(0)
np.random.seed(0)
[n, nt, theta] = [40, 50, np.pi/2]
dt = 0.3/nt
[s_size, a_size] = [6, 2]
[num_threads, num_maxloop] = [4, 10001]
[lr, gamma] = [1e-5, 0.99]
recording = np.zeros(shape=(num_maxloop, num_threads), dtype=np.float32)

a_r_recording = []
for i in range(num_maxloop):
    for j in range(num_threads):
        a_r_recording.append([0, list(np.zeros(nt, dtype=np.int8))])
a_r_recording = np.array(a_r_recording)


class ActorCritic(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense10 = layers.Dense(128, kernel_initializer='he_normal', activation='tanh')
        self.dense11 = layers.Dense(64, kernel_initializer='he_normal', activation='tanh')
        self.policy_logits = layers.Dense(action_size)
        self.dense20 = layers.Dense(128, kernel_initializer='he_normal', activation='tanh')
        self.dense21 = layers.Dense(64, kernel_initializer='he_normal', activation='tanh')
        self.values = layers.Dense(1)

    def call(self, inputs):
        x = self.dense11(self.dense10(inputs))
        logits = self.policy_logits(x)
        v = self.dense20(self.dense21(inputs))
        values = self.values(v)
        return logits, values


class Memory:
    def __init__(self):
        [self.states, self.actions, self.rewards] = [[], [], []]

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        [self.states, self.actions, self.rewards] = [[], [], []]


class Agent:
    def __init__(self):
        self.opt = optimizers.Adam(lr)  # Learning_rate 1e-4 is higher
        self.server = ActorCritic(s_size, a_size)  # (state_size, action_size)
        self.server(tf.random.normal((a_size, s_size)))

    def train(self):
        res_queue = Queue()
        workers = [Worker(self.server, self.opt, res_queue, ii)
                   for ii in range(num_threads)]
        for iii, worker in enumerate(workers):
            print("Starting worker {}".format(iii))
            worker.start()
        [w.join() for w in workers]


# spin Jz operator
def spin_Jz():
    m_list = [m for m in np.arange(-n / 2, (n + 1) / 2)]
    j_z = dia_matrix((m_list, 0), shape=(n + 1, n + 1), dtype=np.complex64)
    return qutip.Qobj(j_z.toarray())


# get_state
def get_state(q_state):
    expe_jx = qutip.expect(qutip.spin_Jx(n / 2), q_state)
    expe_jy = qutip.expect(qutip.spin_Jy(n / 2), q_state)
    expe_jz = qutip.expect(spin_jz, q_state)
    expe_jx2 = qutip.expect(qutip.spin_Jx(n / 2) * qutip.spin_Jx(n / 2), q_state)
    expe_jy2 = qutip.expect(qutip.spin_Jy(n / 2) * qutip.spin_Jy(n / 2), q_state)
    expe_jz2 = qutip.expect(spin_jz * spin_jz, q_state)
    return [expe_jx, expe_jy, expe_jz, expe_jx2, expe_jy2, expe_jz2]


# Interaction Function
def interaction(action, q_state, current_state):
    state0 = current_state
    if action == 0:
        q_state = u0 * q_state  # non-pulse
    elif action == 1:
        q_state = u1 * q_state  # x-pulse
    elif action == 2:
        q_state = u2 * q_state  # y-pulse

    state1 = get_state(q_state)
    in_qfi = qfi(q_state, spin_jz)
    record = namedtuple('record', ['q_state', 'qfi', 'action', 'state', 'next_state'])
    re = record
    [re.q_state, re.qfi, re.action, re.state, re.next_state] = [q_state, in_qfi, action, state0, state1]
    return re


def qfi(state, op):
    # state: psi0, [Qobj];  O: operator, matrix
    psi_theta = uf * state
    psi_theta_d = -1j * op * psi_theta
    f_q_theta = 4 * (psi_theta_d.dag() * psi_theta_d - (psi_theta_d.dag() * psi_theta).norm() ** 2)
    return np.array(f_q_theta, dtype=np.float32)[0][0]


class Worker(threading.Thread):
    def __init__(self, server, opt, result_queue, idx):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.server = server
        self.opt = opt
        self.client = ActorCritic(s_size, a_size)  # (state_size, action_size)
        self.worker_idx = idx
        self.ep_loss = 0.0

    def run(self):
        mem = Memory()
        for epi_counter in range(num_maxloop):
            q_state = q_state_0  # initial state [Qobj]
            current_state = np.array(get_state(q_state=q_state))
            mem.clear()
            # ep_reward = 0.  # total reward of one episode
            [states, actions, qfis, rewards] = [np.zeros((nt, s_size)),
                                                list(np.zeros(nt)),
                                                list(np.zeros(nt + 1)),
                                                list(np.ones(nt))]

            for t in range(nt):  # total time_step = 50
                current_state_tf = tf.constant(current_state, dtype=tf.float32)
                current_state_tf = tf.expand_dims(current_state_tf, axis=0)
                logits, _ = self.client(current_state_tf)
                probs = tf.nn.softmax(logits)
                action = np.random.choice(a_size, p=probs.numpy()[0])
                actions[t] = action
                rec = interaction(action=action, q_state=q_state, current_state=current_state)
                q_state = rec.q_state  # Qobj
                current_state = rec.next_state
                states[t] = rec.state
                qfis[t] = rec.qfi

            for k in range(nt):
                rewards[k] = max(qfis[k:])
                mem.store(states[k], actions[k], rewards[k])
            ep_reward = max(rewards)

            # save data & model
            now_time = datetime.datetime.now()
            recording[epi_counter, self.worker_idx] = np.array([ep_reward], dtype=np.float32)[0]
            a_r_recording[epi_counter * num_threads + self.worker_idx] = [ep_reward, actions]
            if epi_counter % 500 == 0 and self.worker_idx == 0:
                np.savez('n40_data/record_time50_n40_' + now_time.strftime('%m%d') + '.npz', recording=recording)
                np.savez('n40_data/a_r_record_time50_n40_' + now_time.strftime('%m%d') + '.npz',
                         a_r_recording=np.array(a_r_recording))
                self.server.save_weights(
                    'n40_data/saved_weights/in_{}-th_loop_weights.ckpt'.format(epi_counter))
            print('{}-th Loop'.format(epi_counter), 'Worker-{}'.format(self.worker_idx),
                  'R={:.3f}'.format(ep_reward), np.sum(actions), list(actions))

            with tf.GradientTape() as tape:
                total_loss = self.compute_loss(current_state, mem)
            grads = tape.gradient(total_loss, self.client.trainable_weights)
            self.opt.apply_gradients(zip(grads, self.server.trainable_weights))
            self.client.set_weights(self.server.get_weights())
            mem.clear()
            self.result_queue.put(ep_reward)

        self.result_queue.put(None)

    def compute_loss(self, new_state, memory):
        new_state = tf.constant(new_state, dtype=tf.float32)
        new_state = tf.expand_dims(new_state, axis=0)
        reward_sum = self.client(new_state)[-1].numpy()[0]  # V(s_T)
        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        # discounted_rewards = [V(s_1), V(s_2), ..., V(s_T-1), V(s_T)]_target
        logits, values = self.client(tf.constant(np.vstack(memory.states), dtype=tf.float32))
        # print(logits)
        advantage = tf.constant(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values
        # advantage = [V(s_1), V(s_2), ..., V(s_T-1), V(s_T)]_network - [...]_target
        value_loss = advantage ** 2  # loss of Critic Network
        policy = tf.nn.softmax(logits)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                     logits=logits)
        policy_loss = policy_loss * tf.stop_gradient(advantage)
        # Entropy Bonus
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
        policy_loss = policy_loss - 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


if __name__ == '__main__':
    chi = 1
    spin_jz = spin_Jz()
    h_0 = chi * (spin_jz * spin_jz)

    u0 = (-1j * h_0 * dt).expm()
    ux = (-1j * np.pi / 2 * qutip.spin_Jx(n / 2)).expm()
    uy = (-1j * np.pi / 2 * qutip.spin_Jy(n / 2)).expm()
    u1 = ux * u0
    u2 = uy * u0

    uf = (-1j * theta * spin_jz).expm()

    coherent = qutip.spin_coherent(n / 2, np.pi, 0)
    q_state_0 = uy * coherent  # initial state [Qobj]

    master = Agent()
    master.train()
