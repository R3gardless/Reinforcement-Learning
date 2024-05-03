import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from collections import deque


class ReplayBuffer:
    '''
    saves transition datas to buffer
    '''

    def __init__(self, buffer_size=100000, n_step=1, gamma=0.85):
        '''
        Replay Buffer initialize function

        args:
            buffer_size: maximum size of buffer
            n_step: n step if using n step DQN
            gamma: discount factor for n step
        '''
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.gamma = gamma
        
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)

        self.n_states = deque(maxlen=self.n_step)
        self.n_actions = deque(maxlen=self.n_step)
        self.n_rewards = deque(maxlen=self.n_step)
        self.n_next_states = deque(maxlen=self.n_step)
        self.n_dones = deque(maxlen=self.n_step)


    def __len__(self) -> int:
        # self.state 크기 = replay buffer 크기
        return len(self.states)


    def add(self, state, action, reward, next_state, done):
        '''
        add sample to the buffer
        '''
        
        if self.n_step > 1:
            self.n_states.append(state)
            self.n_actions.append(action)
            self.n_rewards.append(reward)
            self.n_next_states.append(next_state)
            self.n_dones.append(done)
            
            if len(self.n_states) == self.n_step:
                # append to main buffer by preprocessing n step

                # n_step 만큼 buffer 채워졌을 경우
                # n_step 만큼의 reward를 계산하여 buffer에 추가
                n_reward = sum([self.n_rewards[i] * (self.gamma ** i) for i in range(self.n_step)])
                self.states.append(self.n_states[0]) # n_step 중 첫번째 state
                self.actions.append(self.n_actions[0]) # n_step 중 첫번째 action
                self.rewards.append(n_reward) # n_step reward
                self.next_states.append(self.n_next_states[-1]) # n_step 중 마지막 state
                self.dones.append(self.n_dones[-1]) # n_step 중 마지막 done
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)

    
    def sample(self, batch_size, device=None):
        '''
        samples random batches from buffer

        args:
            batch_size: size of the minibatch
            device: pytorch device

        returns:
            states, actions, rewards, next_states, dones
        '''

        # buffer에서 batch_size 만큼의 sample을 뽑아서 반환
        indices = np.random.choice(len(self.states), batch_size)
        states = torch.FloatTensor(np.array(self.states)[indices]).to(device)
        actions = torch.LongTensor(np.array(self.actions)[indices]).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)[indices]).to(device)
        next_states = torch.FloatTensor(np.array(self.next_states)[indices]).to(device)
        dones = torch.FloatTensor(np.array(self.dones)[indices]).to(device)
        
        return states, actions, rewards, next_states, dones
    

class DQN(nn.Module):
    '''
    Pytorch module for Deep Q Network
    '''
    def __init__(self, input_size, output_size, hidden_size=128):
        '''
        Define your architecture here
        '''
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
       
    def forward(self, state):
        '''
        Get Q values for each action given state
        '''
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
        


class Agent:
    def __init__(self, state_size, action_size, learning_rate = 0.001, gamma = 0.99, n_step = 8):
        self.state_size = state_size
        self.action_size = action_size

        self.curr_step = 0
        self.learning_rate = learning_rate
        self.buffer_size = 50000
        self.batch_size = 64

        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        self.gamma = gamma
        self.n_step = n_step
        self.target_update_freq = 512
        self.gradient_update_freq = 1
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = DQN(state_size, action_size).to(self.device)
        self.target_network = deepcopy(self.network) # Copy Target Network

        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, n_step=self.n_step, gamma=self.gamma)


    def reward_function(self, state, reward):

        x = state[0]
        if(0.5 < x < 0.55):
            reward = 1
        else:
            reward = 0.01
        return reward

    def select_action(self, state, is_test=False):
        '''
        selects action given state

        returns:
            discrete action integer
        '''
        if np.random.rand() > self.epsilon or is_test:
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.network(state)
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.action_size)
        return action

    def train_network(self, states, actions, rewards, next_states, dones):
        # 현재 states 에 대한 q value 계산(network 사용)
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))
        # 다음 states 에 대한 q value 계산(target network 사용)
        next_q_values = self.target_network(next_states).max(dim=1)[0].detach()
        # episode 가 끝난 경우 mask 를 0 으로 설정
        mask = 1 - dones
        # target q value 계산
        target_q_values = rewards + (self.gamma ** self.n_step) * next_q_values * mask
        # loss 계산(mse loss 사용)
        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
        # optimizer 의 gradient 초기화
        self.optimizer.zero_grad()
        # 역전파 수행, 가중치에 대한 gradient 계산
        loss.backward()
        # gradient 사용하여 network 가중치 업데이트
        self.optimizer.step()


    def update_target_network(self):
        '''
        updates the target network to online
        '''
        # Use deepcopy of online network
        self.target_network = deepcopy(self.network)


    def step(self, state, action, reward, next_state, done):
        self.curr_step += 1
        reward = self.reward_function(state, reward)
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # replay buffer 가 batch size 보다 크고, gradient update freq 만큼 step이 지난 경우
        if len(self.replay_buffer) > self.batch_size and self.curr_step % self.gradient_update_freq == 0:
            self.train_network(*self.replay_buffer.sample(self.batch_size, device=self.device))

            if self.curr_step % self.target_update_freq == 0:
                self.update_target_network()

            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)




