# sometimes the result might be wrong(all zeros), don't know why
import gym
import numpy as np
import random

env = gym.make("FrozenLake-v0")

'''
action = env.action_space
print('aciton', action)

state = env.observation_space
print('state', state)
'''

action_size = env.action_space.n
# print("action_size", action_size)

state_size = env.observation_space.n
# print("action_size", action_size)

q_table = np.zeros((state_size, action_size))

# print(q_table)


total_episode = 15000
learning_rate = 0.8
max_step = 99
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

rewards = []  # why list

# new episode starts from here
for episode in range(total_episode):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0

    # new iteration starts from here
    for step in range(max_step):
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state, :])

        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        total_reward += reward

        state = new_state

        if done == True:

            break

    # epsilon decay
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    rewards.append(total_reward)

print("scores" + str(sum(rewards)/total_episode))
print(q_table)


env.reset()
for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print('#####################')
    print('episode', episode)

    for step in range(max_step):
        action = np.argmax(q_table[state, :])
        new_state, reward, done, info = env.step(action)
        if done == True:
            # just to print the last frame
            env.render() # just to reload the image
            print('number of steps', step)
            break

        state = new_state
env.close()




















