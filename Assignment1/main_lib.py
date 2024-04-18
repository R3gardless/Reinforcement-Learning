import numpy as np

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)

    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            # policy 는 deterministic & greedy 하게 최대값 선택
            V[s] = sum([p * (r + gamma * V[next_state]) for p, next_state, r in env.MDP[s][np.argmax(policy[s])]])
            delta = max(delta, abs(v - V[s]))
        if delta < theta: break

    return V

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA

    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * V[next_state]) for p, next_state, r in env.MDP[s][a]])
        policy[s] = np.eye(env.nA)[np.argmax(q_sa)]

    return policy

def policy_iteration(env, gamma=0.99, theta=1e-8): 

    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # policy evaluation 수행
        V = policy_evaluation(env, policy, gamma, theta)
        # policy improvement 수행
        new_policy = policy_improvement(env, V, gamma)
        # policy 가 변하지 않으면 종료
        if (new_policy == policy).all(): break
        else: policy = new_policy

    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    # implement value iteration
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max([sum([p * (r + gamma * V[next_state]) for p, next_state, r in env.MDP[s][a]]) for a in range(env.nA)])
            delta = max(delta, abs(v - V[s]))
        if delta < theta: break

    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * V[next_state]) for p, next_state, r in env.MDP[s][a]])
        policy[s] = np.eye(env.nA)[np.argmax(q_sa)]

    return policy, V