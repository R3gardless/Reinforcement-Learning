from collections import defaultdict
import numpy as np

class Agent:

    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.eps = 1.0
        self.num_episode = 1
        if self.mode == "mc_control":
            self.alpha = 0.001
            self.gamma = 0.9
            self.history = list()
        elif self.mode == "q_learning":
            self.alpha = 0.1
            self.gamma = 0.8

    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        if self.mode == "test_mode":
            return np.argmax(self.Q[state])
    
        else:
            # epsilon-greedy policy
            if np.random.rand() > self.eps:
                return np.argmax(self.Q[state])
            else:
                return np.random.choice(self.n_actions)


    def step(self, state, action, reward, next_state, done):
        
        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)

        """

        # GLIE (Greedy in the Limit with Infinite Exploration)
        if done:
            self.num_episode += 1
            self.eps = 1.0 / self.num_episode

        if self.mode == "mc_control":
            self.history.append((state, action, reward))
            if done:
                G = 0
                for state, action, reward in reversed(self.history):
                    G = reward + self.gamma * G
                    self.Q[state][action] += self.alpha * (G - self.Q[state][action])
                self.history.clear()

        elif self.mode == "q_learning":
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])