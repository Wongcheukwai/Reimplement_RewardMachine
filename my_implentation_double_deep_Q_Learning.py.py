import gym
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# tester haha
# todo is there any temporal difference between two objects?
''''
#  a small sample, the result shows you the reward the agent gets at its last episode
env = gym.make('CartPole-v0')

env.reset()
rewards = []
for _ in range(13):
    env.render()
    state, reward, done, info = env.step(env.action_space.sample())
    rewards.append(reward)
    if done:
        rewards = []
        env.reset()

env.close()

print('length of reward', len(rewards))
'''

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10, name='QNetwork'):
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')  # todo what is this variable used for，is None mean vector?

            # one_hot_action = [[1, 0], [0, 1]...]
            one_hot_actions = tf.one_hot(self.actions_, action_size) # todo 这里为什么不用self

            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target') # that is why no stop gradient is used here

            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)  # todo 如果这里fc1不加self会怎么样

            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, activation_fn=None)

            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1) # todo this is Q_eval

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size): # todo 第79行定义的batch_size会不会影响这个东西
        # idx is the id of the chosen sample
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx] # datatype: list

# Hyperparameters

# training
train_episode = 3
max_steps = 200
gamma = 0.99

# exploration
epsilon_max = 1.0
epsilon_min = 0.01
decay_rate = 0.0001

# Network
hidden_layer_size = 64
learning_rate = 0.0001

# Memory parameters
memory_size = 10000
batch_size = 20
pretrain_length = batch_size

tf.reset_default_graph() # todo 为什么要reset一下网络？

mainQN = QNetwork(name='main', hidden_size=hidden_layer_size, learning_rate=learning_rate)
TargetNetwork = QNetwork(name='targetQNetwork', hidden_size=hidden_layer_size, learning_rate=learning_rate)


def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "main")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "targetQNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# start the game
env = gym.make('CartPole-v0')
env.reset()
state, reward, done, _ = env.step(env.action_space.sample())
memory = Memory(max_size=memory_size)

# get the pretrained sample
for ii in range(pretrain_length):
    #env.render()
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state))

        env.reset()
        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        memory.add((state, action, reward, next_state))
        state = next_state

# let's start training
saver = tf.train.Saver()
rewards_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    step = 0
    state = env.reset()

    for ep in range(1, train_episode):
        if ep % 100 == 0:
            update_target = update_target_graph()
            sess.run(update_target)
            print('Model updated')
        total_reward = 0
        t = 0
        while t < max_steps:
            step += 1
            # env.render()
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate*step)
            if epsilon > np.random.rand():
                action = env.action_space.sample()
            else:
                feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                Qs = sess.run(mainQN.output, feed_dict=feed)  # todo i think this is an evaluation, loss计算了吗
                print('Qs here', sess.run(Qs))  # todo 卧槽为什么这个打印命令没用
                action = np.argmax(Qs)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                next_state = np.zeros(state.shape)
                t = max_steps # to stop the while loop

                print('Episode: {}'.format(ep),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss), # exucuse me，这里loss是什么鬼
                      'Explore P: {:.4f}'.format(epsilon))
                rewards_list.append((ep, total_reward))
                memory.add((state, action, reward, next_state))

                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
            else:
                memory.add((state, action, reward, next_state))
                state = next_state
                t += 1

            # sample minibatch
            batch = memory.sample(batch_size) # type: list, at least contains 20 samples
            states = np.array([each[0] for each in batch]) # change from list to array to fit tensor
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            # train network, trained every step
            q_eval_next = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
            target_Qs = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states})
            print('targets_Qs', target_Qs)

            episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)  # todo 前半部分不懂
            target_Qs[episode_ends] = (0, 0)

            target_Qs_batch = []

            for i in range(0, len(batch)):
                action_for_targets = np.argmax(q_eval_next[i])
                each_targets = rewards[i] + gamma * target_Qs[i][action_for_targets]
                target_Qs_batch.append(each_targets)

            targets_mb = np.array([each for each in target_Qs_batch])
            print('targets_mb', targets_mb)
            Q_fuck = sess.run(mainQN.Q,
                              feed_dict={mainQN.inputs_: states,
                                         mainQN.targetQs_: targets_mb,
                                         mainQN.actions_: actions})
            print('qfuck', Q_fuck)
            #targets = rewards + gamma * np.max(target_Qs, axis=1)

            loss, _ = sess.run([mainQN.loss, mainQN.opt],
                               feed_dict={mainQN.inputs_: states,
                                          mainQN.targetQs_: targets_mb, # here we feed targets to targetQs_
                                          mainQN.actions_: actions})
    saver.save(sess, "checkpoints/cartpole.ckpt")

#print('rewardfinal',rewards_list)
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

eps, rews = np.array(rewards_list).T

smoothed_rews = running_mean(rews, 10)
#print('somejgjgjggj',smoothed_rews)


plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
#print('done1')
plt.plot(eps, rews, color='grey', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()


test_episodes = 10
test_max_steps = 400
env.reset()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    for ep in range(1, test_episodes):
        t = 0
        while t < test_max_steps:
            env.render()

            # Get action from Q-network
            feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
            Qs = sess.run(mainQN.output, feed_dict=feed)
            action = np.argmax(Qs)

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)

            if done:
                t = test_max_steps
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                state = next_state
                t += 1
