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

ENABLE_GRAD_CLIPPING = True
GRAD_CLIP_MAX_NORM = 0.5


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

    def __len__(self):
        return len(self.states)


class ActorCritic(nn.Module):
    '''
    Pytorch module for Actor-Critic network
    '''
    def __init__(self, state_dim=OBS_DIM, action_dim=ACT_DIM, hidden_size=64):
        '''
        Define your architecture here
        '''
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc_mean = nn.Linear(hidden_size, action_dim)
        self.actor_fc_std = nn.Linear(hidden_size, action_dim)
        self.fc3 = nn.Linear(hidden_size, 32)
        self.value_critic = nn.Linear(32, 1)
        
    def forward(self, state):
        x = self.fc1(state) # ReLU activation after first FC layer
        x = self.fc2(x)      # ReLU activation after second FC layer
        return x

    def actor(self, state):
        '''
        Get action distribution (mean, std) for given states
        '''
        x = self.forward(state)
        mu = torch.tanh(self.actor_fc_mean(x)) * ACT_LIMIT  # action의 범위를 -ACT_LIMIT ~ ACT_LIMIT로 설정
        std = self.actor_fc_std(x)  # std는 항상 양수이므로 softplus 함수를 통과시켜 양수로 만듦
        std = torch.clamp(std, 1e-2, 0.3)  # std의 최소값을 1e-2로 설정
        return mu, std

    def critic(self, state):
        '''
        Get values for given states
        '''
        x = self.forward(state)
        x = F.relu(self.fc3(x))  # ReLU activation before the value prediction
        value = self.value_critic(x)
        return value

class Worker(object):
    def __init__(self, global_actor, global_epi, sync, finish, n_step, seed, lr=0.0006, gamma=0.99, entropy_coef=0.01):
        self.env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
        self.env.seed(seed)
        self.lr = lr
        self.gamma = gamma   
        self.entropy_coef = entropy_coef
        
        ############################################## DO NOT CHANGE ##############################################
        self.global_actor = global_actor
        self.global_epi = global_epi
        self.sync = sync
        self.finish = finish
        self.optimizer = optim.Adam(self.global_actor.parameters(), lr=self.lr)
        ###########################################################################################################  
        
        self.n_step = n_step
        self.local_actor = ActorCritic()
        self.local_actor.load_state_dict(global_actor.state_dict())
        self.nstep_buffer = NstepBuffer()

    def select_action(self, state):
        '''
        Selects action given state

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
        rewards_to_go = [] # n-step return을 저장할 리스트
        discounted_sum = 0 # n-step return을 계산하기 위한 변수
        for reward, done in zip(reversed(rewards), reversed(dones)): # n-step return 계산
            if done:
                discounted_sum = 0 # 마지막 state가 done인 경우 n-step return은 0
            discounted_sum = reward + (self.gamma * discounted_sum) # n-step return 계산
            rewards_to_go.insert(0, discounted_sum) # n-step return을 리스트에 저장
        
        rewards_to_go = torch.FloatTensor(rewards_to_go) # n-step return을 tensor로 변환

        states = torch.FloatTensor(np.array(states)) # states를 tensor로 변환
        actions = torch.FloatTensor(np.array(actions)) # actions를 tensor로 변환

        values = self.local_actor.critic(states).squeeze() # state에 대한 value를 계산
        advantage = rewards_to_go - values # advantage 계산

        critic_loss = advantage.pow(2).sum() # critic loss 계산

        mu, std = self.local_actor.actor(states) # state에 대한 mu, std 계산
        dist = Normal(mu, std) # action distribution 생성 
        log_probs = dist.log_prob(actions).sum(axis=-1) # log_probs 계산
        actor_loss = -(log_probs * advantage.detach()).sum() # actor loss 계산

        entropy = dist.entropy().sum() # entropy 계산
        actor_loss -= self.entropy_coef * entropy # actor loss에 entropy term 추가

        total_loss = actor_loss + critic_loss # total loss 계산
         
        ############################################## DO NOT CHANGE ##############################################
        # Global optimizer update 준비
        
        # Global Network와 Local Network의 모든 파라미터의 gradients를 0으로 초기화
        self.optimizer.zero_grad(set_to_none=False)
        
        total_loss.backward()

        # Gradient Clipping 관련 전역 변수가 정의되어 있는지 확인
        if 'ENABLE_GRAD_CLIPPING' in globals() and 'GRAD_CLIP_MAX_NORM' in globals():
            # 활성화 여부에 따라 Gradient Clipping 적용
            if ENABLE_GRAD_CLIPPING:
                torch.nn.utils.clip_grad_norm_(parameters=self.local_actor.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

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
                self.nstep_buffer.add(state, action.item(), reward, next_state, done)

                # n step마다 한 번씩 train_network 함수 실행
                if len(self.nstep_buffer) % self.n_step == 0 or done:
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
