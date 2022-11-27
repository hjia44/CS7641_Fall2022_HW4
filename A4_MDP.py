import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import mdptoolbox.example




def problem1_frozenlake():
    print('A4 Forzen Lake Started')
    env = gym.make("FrozenLake-v1")

    # uncomment this section if render is needed
    #env2 = gym.make("FrozenLake-v1", render_mode='human')
    #reset_results2 = env2.reset()
    #env2.render()

    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    gamma, tol, max_iter = 0.9, 1e-7, 1e3
    time_learner, iterations = [], []

    time_start = time.time()
    p_PI, v_PI, iter_PI = policy_iter(env, gamma, tol)
    gap_time = time.time() - time_start
    time_learner.append(gap_time)
    iterations.append(iter_PI)
    print('policy PI', p_PI)

    env.reset()
    time_start = time.time()
    p_VI, v_VI, iter_VI = value_iter(env, gamma, tol, int(max_iter))
    gap_time = time.time() - time_start
    time_learner.append(gap_time)
    iterations.append(iter_VI)
    print('policy VI', p_VI)

    env.reset()
    episode_num, max_step, epsilon, alpha, gamma, decay = 30000, 1000, 0.1, 0.2, 0.9, False
    time_start = time.time()
    q_table_1, rewards_qlearner = qLearner(env, episode_num, max_step, epsilon, alpha, gamma)
    policy_qlearner = qlearner_policy(env, q_table_1)
    gap_time = time.time() - time_start
    time_learner.append(gap_time)
    #print(time_learner)
    data_x = ['PI', 'VI']
    color_list = ['red', 'green']
    fig_plot_1_v2(data_x, iterations, 'Learner', 'iteration num', 'Iteration num vs. Learner', 'Learner_iteration_num_frozenlake.png', color_list)

    data_x = ['PI', 'VI', 'QLearning']
    color_list = ['red', 'green', 'blue']
    fig_plot_1_v2(data_x, time_learner, 'Learner', 'time(s)', 'Time vs. Learner', 'Learner_Time_frozenlake.png',
                  color_list)

    epi_num, mean_rewards = 10000, []
    mean_rewards.append(evaluate_Learner(env, epi_num, p_PI))
    mean_rewards.append(evaluate_Learner(env, epi_num, p_VI))
    qLearner_reward = evaluate_Learner(env, epi_num, policy_qlearner)
    mean_rewards.append(qLearner_reward)
    #print(mean_rewards)
    data_x = ['PI', 'VI', 'QLearning']
    color_list = ['red', 'green', 'blue']
    fig_plot_1_v2(data_x, mean_rewards, 'Learner', 'reward', 'reward vs. Learner', 'Learner_reward_frozenlake.png', color_list)

    ### Episode Number Effects ###
    env.reset()
    mean_rewards_PI, mean_rewards_VI, mean_rewards_qlearner = [], [], []
    data_x = np.arange(1000, 40000, 5000)
    for epi_num in range(1000, 40000, 5000):
        mean_rewards_PI.append(evaluate_Learner(env, epi_num, p_PI))
        mean_rewards_VI.append(evaluate_Learner(env, epi_num, p_VI))
        qLearner_reward = evaluate_Learner(env, epi_num, policy_qlearner)
        print('epi_num:', epi_num, qLearner_reward)
        mean_rewards_qlearner.append(qLearner_reward)
    xlabel, ylabel = 'Episode Numbers', 'Average Reward'
    title, fig_name = 'Average Reward vs. Episode Numbers', 'Average_Reward_vs_Episode_Numbers.png'
    #print(data_x)
    #print(mean_rewards_PI)
    #print(mean_rewards_VI)
    #print(mean_rewards_qlearner)
    fig_plot_3(data_x, mean_rewards_PI, mean_rewards_VI, mean_rewards_qlearner, xlabel, ylabel, title, fig_name)

    ### Epsilon Effects ###
    env.reset()
    episode_num, max_step, alpha, gamma = 10000, 2000, 0.2, 0.9
    data_x = np.arange(0.1, 1, 0.1)
    mean_rewards_qlearner_epsilon = np.zeros(len(data_x))
    for i in range(len(data_x)):
        q_table_2, rewards_qlearner_2 = qLearner(env, episode_num, max_step, data_x[i], alpha, gamma)
        policy_qlearner = qlearner_policy(env, q_table_2)
        mean_rewards_qlearner_epsilon[i] = evaluate_Learner(env, episode_num, policy_qlearner)
    xlabel, ylabel = 'Epsilon', 'Average Reward'
    title, fig_name = 'Average Reward vs. Epsilon', 'Average_Reward_vs_Epsilon_qleaner.png'
    print('data_x', data_x)
    print('data_y', mean_rewards_qlearner_epsilon)
    fig_plot_1(data_x, mean_rewards_qlearner_epsilon, xlabel, ylabel, title, fig_name)


def problem2_forest():
    # state_space = 100
    state_space = 1000
    iters, times = [], []
    P, R = mdptoolbox.example.forest(state_space, 10, 50, 0.1)

    ### Policy Iteration
    PI = mdptoolbox.mdp.PolicyIteration(P, R, 0.95)
    PI.run()
    iters.append(PI.iter)
    times.append(PI.time)

    ### Value Iteration
    VI = mdptoolbox.mdp.ValueIteration(P, R, 0.95)
    VI.run()
    iters.append(VI.iter)
    times.append(VI.time)

    ### QLearnning
    qleaner = mdptoolbox.mdp.QLearning(P, R, 0.95)
    qleaner.run()
    times.append(qleaner.time)

    epi_num, state_num, p = state_space, 1000, 0.1
    print(epi_num, state_num, p)
    rewards, episodes = reward_calculation(R, PI.policy, epi_num, state_num, p)
    print('rewards', rewards)
    print('episodes', episodes)
    xlabel, ylabel = 'Episodes', 'rewards'
    title, fig_name = 'Rewards vs. Episodes', 'Rewards_vs_Episodes_forest.png'
    #fig_plot_1(episodes, rewards, xlabel, ylabel, title, fig_name)

    data_x = list(range(1000, 3001, 1000))
    print(data_x)
    rewards3 = []
    for i in range(len(data_x)):
        rewards, episodes = reward_calculation(R, PI.policy, epi_num, state_num, p)
        rewards3.append(rewards)
    xlabel, ylabel = 'Episodes', 'Average Reward'
    title, fig_name = 'Average Reward vs. Episodes', 'Average_Reward_vs_Episodes_forest.png'
    print(rewards3[0])
    fig_plot_3(episodes, rewards3[0], rewards3[1], rewards3[2], xlabel, ylabel, title, fig_name)

    data_x = ['PI', 'VI', 'QLearning']
    color_list = ['red', 'green', 'blue']
    fig_plot_1_v2(data_x, times, 'Learner', 'time(s)', 'Learner vs. Time', 'Learner_Time_forest.png',
                  color_list)

    data_x = ['PI', 'VI']
    color_list = ['red', 'green']
    fig_plot_1_v2(data_x, iters, 'Learner', 'Iterations', 'Iterations vs. Learner', 'Learner_Iterations_forest.png',
                  color_list)

def reward_calculation(R, policy, epi_num, state_num, p):
    episodes = list(range(1, epi_num))
    rewards = np.zeros(len(episodes))
    s, reward = 0, 0.0
    for epi in range(1, epi_num):
        a = policy[s]
        reward = reward + R[s][a]
        if s == state_num or a == 1:
            s = 0
        # Fire
        prob_for_fire = np.random.uniform(0, 1)
        if a == 0 and prob_for_fire < p:
            s = 0
        if a == 0 and prob_for_fire >= p:
            s += 1
        #print(reward)
        rewards[epi-1] = reward / episodes[epi-1]
    return rewards, episodes

def evaluate_Learner(env, epi_num, policy):
    rewards = np.zeros((epi_num))
    total_reward = 0.0
    for i in range(epi_num):
        s = env.reset()[0]
        while True:
            a = policy[s]
            s_new, r, done, _, _ = env.step(a)
            s = s_new
            total_reward = total_reward + r
            if done:
                break
        rewards[i] = total_reward
        #print('reward:' + str(rewards[i]))
    mean_reward = np.sum(total_reward / epi_num)
    print('mean rewards' + str(mean_reward))
    return mean_reward

def qLearner(env, episode_num, max_step, epsilon, alpha, gamma):
    state_nums = env.observation_space.n
    action_nums = env.action_space.n
    q_table = np.zeros((state_nums, action_nums))
    rewards = np.zeros(episode_num)
    for epi in range(episode_num):
        state = env.reset()[0]
        done, step, r_epi = False, 0, 0.0
        for step in range(max_step):
            a = 0
            epsilon_rand = np.random.uniform(0, 1)
            if epsilon_rand > epsilon:
                a = np.argmax(q_table[state, :])
            else:
                a = env.action_space.sample()
            state_new, r, done, _, _ = env.step(a)
            q_table[state, a] = q_table[state, a] + alpha * (r + gamma * np.max(q_table[state_new, :]) - q_table[state, a])
            state = state_new
            r_epi = r_epi + r
            if done:
                break
        rewards[epi] = r_epi
    return q_table, rewards

def qlearner_policy(env, q_table):
    state_nums = env.observation_space.n
    policy = [0 for i in range(state_nums)]
    for s in range(state_nums):
        policy[s] = np.argmax(q_table[s, :])
    return policy

def value_iter(env, gamma, tol, max_iter):
    state_nums = env.observation_space.n
    action_nums = env.action_space.n
    iter_total = 0
    V = [0 for i in range(state_nums)]
    for i in range(max_iter):
        iter_total += 1
        delta = 0.0
        for s in range(state_nums):
            a_vals = [0 for i in range(action_nums)]
            for a in range(action_nums):
                for j in range(len(env.P[s][a])):
                    p, s_new, reward, _ = env.P[s][a][j]
                    a_vals[a] += p * (reward + gamma * V[s_new])
            #a_val_best = max(a_vals)
            delta = max(delta, abs(V[s] - max(a_vals)))
            V[s] = max(a_vals)
        if delta < tol:
            break
    policy = obtain_policy(env, V, gamma)
    return policy, V, iter_total


def policy_iter(env, gamma, theta):
    state_nums = env.observation_space.n
    iter_total = 0
    V = [0 for i in range(state_nums)]
    policy = [0 for i in range(state_nums)]
    while True:
        iter_total += 1
        V = policy_evaluation(env, V, policy, gamma, theta)
        policy, V, stable_flag = policy_improvement(env, V, policy, gamma)
        if stable_flag == True:
            break
    return policy, V, iter_total


def obtain_policy(env, V, gamma):
    state_nums = env.observation_space.n
    action_nums = env.action_space.n
    policy = [0 for i in range(state_nums)]
    for s in range(state_nums):
        a_vals = [0 for i in range(action_nums)]
        for a in range(action_nums):
            for i in range(len(env.P[s][a])):
                p, s_new, reward, _ = env.P[s][a][i]
                a_vals[a] += p * (reward + gamma * V[s_new])
        policy[s] = np.argmax(a_vals)
    return policy


def policy_evaluation(env, V, policy, gamma, tol):
    state_nums = env.observation_space.n
    flag = True
    while flag == True:
        delta = 0.0
        for s in range(state_nums):
            v_old, tmp = V[s], 0.0
            for p, s_new, r_new, _ in env.P[s][policy[s]]:
                tmp += p * (r_new + gamma * V[s_new])
            delta_new = abs(v_old - tmp)
            delta = max(delta_new, delta)
            V[s] = tmp
        if delta < tol:
            flag = False
    return V

def policy_improvement(env, V, policy, gamma):
    state_nums = env.observation_space.n
    action_nums = env.action_space.n
    stable_flag = True
    for s in range(state_nums):
        a_old = policy[s]
        a_val_list = [0 for i in range(action_nums)]
        for a in range(action_nums):
            for i in range(len(env.P[s][a])):
                p, s_new, reward, _ = env.P[s][a][i]
                a_val_list[a] += p * (reward + gamma * V[s_new])
        policy[s] = np.argmax(a_val_list)
        if a_old != policy[s]:
            stable_flag = False
    return policy, V, stable_flag


def fig_plot_3(data_x, data_y1, data_y2, data_y3, xlabel, ylabel, title, fig_name):
    plt.plot(data_x, data_y1, marker="o", color="red", label="PI")
    plt.plot(data_x, data_y2, marker="o", color="blue", label="VI")
    plt.plot(data_x, data_y3, marker="o", color="green", label="Qlearner")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    #plt.ylim(0.5, 1.0)
    plt.title(title, fontsize=18)
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

def fig_plot_1(data_x, data_y1, xlabel, ylabel, title, fig_name):
    plt.plot(data_x, data_y1, marker="o", color="red", label="PI")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    #plt.ylim(0.5, 1.0)
    plt.title(title, fontsize=18)
    #plt.legend()
    plt.savefig(fig_name)
    plt.show()

def fig_plot_1_v2(data_x1, data_y1, xlabel, ylabel, title, fig_name, color_list):
    plt.bar(data_x1, data_y1, color=color_list)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=18)
    #plt.ylim(y1, y2)
    #plt.legend()
    plt.savefig(fig_name)
    plt.show()

if __name__ == '__main__':
    print('Assignment4_Markov_Decision_Processes')
    problem1_frozenlake()
    problem2_forest()



