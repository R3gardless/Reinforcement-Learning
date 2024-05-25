import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

ENV = gym.make("InvertedPendulumSwingupBulletEnv-v0")
OBS_DIM = ENV.observation_space.shape[0]
ACT_DIM = ENV.action_space.shape[0]
ACT_LIMIT = ENV.action_space.high[0]
ENV.close()

#########################################################################################################################
############ 이 template에서는 DO NOT CHANGE 부분을 제외하고 마음대로 수정, 구현 하시면 됩니다                    ############
#########################################################################################################################

## 주의 : "InvertedPendulumSwingupBulletEnv-v0"은 continuious action space 입니다.
## Asynchronous Advantage Actor-Critic(A3C)를 참고하면 도움이 될 것 입니다.

class NstepBuffer:
    '''
    Save n-step trainsitions to buffer
    '''
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        '''
        add sample to the buffer
        '''
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self):
        '''
        sample transitions from buffer
        '''
        return self.states, self.actions, self.rewards, self.next_states, self.dones

    def reset(self):
        '''
        reset buffer
        '''
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

class ActorCritic(nn.Module):
    '''
    Pytorch module for Actor-Critic network
    '''
    def __init__(self, state_dim=OBS_DIM, action_dim=ACT_DIM, hidden_size=256):
        '''
        Define your architecture here
        '''
        super(ActorCritic, self).__init__()

        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
    
        self.actor_layer_mean = nn.Linear(hidden_size, action_dim)
        self.actor_layer_std = nn.Linear(hidden_size, action_dim)

        self.critic_layer = nn.Linear(hidden_size, 1)

    def forward(self, states):
        x = F.relu(self.layer1(states))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        return x
    
    def actor(self, states):
        '''
        Get action distribution (mean, std) for given states
        '''
        x = self.forward(states)
        mu = F.tanh(self.actor_layer_mean(x))
        std = self.actor_layer_std(x)
        std = torch.clamp(std, 0.001, 0.3)

        return mu, std

    def critic(self, states):
        '''
        Get values for given states
        '''
        x = self.forward(states)
        value = self.critic_layer(x)
        
        return value


class Worker(object):
    def __init__(self, global_actor, global_epi, sync, finish, n_step, seed, lr=0.001, gamma=0.99):
        self.env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
        self.env.seed(seed)
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = 0.01

        ############################################## DO NOT CHANGE ##############################################
        self.global_actor = global_actor
        self.global_epi = global_epi
        self.sync = sync
        self.finish = finish
        self.optimizer = optim.Adam(self.global_actor.parameters(), lr=self.lr)
        ############################# ##############################################################################  
        
        self.n_step = n_step
        self.local_actor = ActorCritic()
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.nstep_buffer = NstepBuffer()

    def select_action(self, state):
        '''
        selects action given state

        return:
            continuous action value
        '''
        # action [-1, 1]로 clipping
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.local_actor.actor(state)
        dist = Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, -ACT_LIMIT, ACT_LIMIT)

        return action.data.numpy()[0]
    
    def train_network(self, states, actions, rewards, next_states, dones):
        '''
        Advantage Actor-Critic training algorithm
        '''

        if len(rewards) == 0: return
        states = states[0]
        actions = actions[0]
        reward = 0
        for reward in rewards[::-1]:
            reward = reward + self.gamma * reward
        next_states = next_states[-1]
        dones = dones[-1]

        states = torch.FloatTensor(np.array([states]))
        actions = torch.FloatTensor(np.array([actions]))
        rewards = torch.FloatTensor(np.array([rewards]))
        next_states = torch.FloatTensor(np.array([next_states]))
        dones = torch.FloatTensor(np.array([dones]))

        # Calculate critic loss
        values = self.local_actor.critic(states)
        next_values = self.local_actor.critic(next_states)
        target_values = rewards + self.gamma * next_values * (1 - dones)
        advantages = target_values - values
        critic_loss = advantages.pow(2).mean()

        # Calculate actor loss
        mu, std = self.local_actor.actor(states)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        actor_loss = -(log_prob * advantages.detach()).mean() - self.entropy_coef * entropy

        total_loss = actor_loss + critic_loss
        # print("Actor Loss: ", actor_loss.item(), "Critic Loss: ", critic_loss.item(), "Total Loss: ", total_loss.item())
        ############################################## DO NOT CHANGE ##############################################
        # Global optimizer update 준비
        self.optimizer.zero_grad()
        total_loss.backward()

        # Local parameter를 global parameter로 전달
        for global_param, local_param in zip(self.global_actor.parameters(), self.local_actor.parameters()):
                global_param._grad = local_param.grad

        # Global optimizer update
        self.optimizer.step()

        # Global parameter를 local parameter로 전달
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        ###########################################################################################################  

    def train(self):
        step = 1

        while True:
            state = self.env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.nstep_buffer.add(state, action, reward, next_state, done)

                # n step마다 한 번씩 train_network 함수 실행
                if step % self.n_step == 0 or done:
                    self.train_network(*self.nstep_buffer.sample())
                    self.nstep_buffer.reset()                    
                
                state = next_state
                step += 1

            ############################################## DO NOT CHANGE ##############################################
            # 에피소드 카운트 1 증가                
            with self.global_epi.get_lock():
                self.global_epi.value += 1
            
            # evaluation 종료 조건 달성 시 local process 종료
            if self.finish.value == 1:
                break

            # 매 에피소드마다 global actor의 evaluation이 끝날 때까지 대기 (evaluation 도중 파라미터 변화 방지)
            with self.sync:
                self.sync.wait()
            ###########################################################################################################

        self.env.close()
