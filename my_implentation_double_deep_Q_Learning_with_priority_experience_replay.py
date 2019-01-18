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

            self.ISWeights_ = tf.placeholder(tf.float32, [None,1], name='IS_weights')

            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target') # that is why no stop gradient is used here

            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)  # todo 如果这里fc1不加self会怎么样

            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, activation_fn=None)

            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1) # todo this is Q_eval

            self.absolute_errors = tf.abs(self.targetQs_ - self.Q)# for updating Sumtree

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(sampling_probabilities/p_min, -self.PER_b)

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
# Hyperparameters


# training
train_episode = 500
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
memory = Memory(memory_size)

# get the pretrained sample
for ii in range(pretrain_length):
    #env.render()
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if done:
        next_state = np.zeros(state.shape)
        experience = state, action, reward, next_state  # tuple
        memory.store(experience)

        env.reset()
        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        experience = state, action, reward, next_state  # tuple
        memory.store(experience)

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
                experience = state, action, reward, next_state  # tuple
                memory.store(experience)

                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
            else:
                experience = state, action, reward, next_state  # tuple
                memory.store(experience)

                state = next_state
                t += 1

            # sample minibatch
            tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
            states = np.array([each[0][0] for each in batch])  # change from list to array to fit tensor
            actions = np.array([each[0][1] for each in batch])
            rewards = np.array([each[0][2] for each in batch])
            next_states = np.array([each[0][3] for each in batch])

            # train network, trained every step
            q_eval_next = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
            target_Qs = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states})

            episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)  # todo 前半部分不懂
            target_Qs[episode_ends] = (0, 0)

            target_Qs_batch = []

            for i in range(0, len(batch)):
                action_for_targets = np.argmax(q_eval_next[i])
                each_targets = rewards[i] + gamma * target_Qs[i][action_for_targets]
                target_Qs_batch.append(each_targets)

            targets_mb = np.array([each for each in target_Qs_batch])
            # print('targets_mb', targets_mb)

            Q_fuck = sess.run(mainQN.Q,
                              feed_dict={mainQN.inputs_: states,
                                         mainQN.targetQs_: targets_mb,
                                         mainQN.actions_: actions})
            # print('qfuck', Q_fuck)

            # targets = rewards + gamma * np.max(target_Qs, axis=1)

            loss, absolute_errors, _ = sess.run([mainQN.loss, mainQN.absolute_errors, mainQN.opt],
                                                feed_dict={mainQN.inputs_: states,
                                                           mainQN.targetQs_: targets_mb,  # here we feed targets to targetQs_
                                                           mainQN.ISWeights_: ISWeights_mb,
                                                           mainQN.actions_: actions})
            # print('IS', ISWeights_mb)

            memory.batch_update(tree_idx, absolute_errors)
            # print('absolute',absolute_errors)

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