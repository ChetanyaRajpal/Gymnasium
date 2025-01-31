import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader,TensorDataset

class Network(nn.Module):
    def __init__(self, action_size, seed = 42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cv1 = nn.Conv2d(3,32,kernel_size=8,stride = 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.cv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.cv3 = nn.Conv2d(64,64, kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.cv4 = nn.Conv2d(64,128,kernel_size=3,stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(10*10*128, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,action_size)
        
    def forward(self,state):
        x = F.relu(self.bn1(self.cv1(state)))
        x = F.relu(self.bn2(self.cv2(x)))
        x = F.relu(self.bn3(self.cv3(x)))
        x = F.relu(self.bn4(self.cv4(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

import gymnasium as gym
env = gym.make('BankHeistDeterministic-v0',render_mode = 'human')
state_shape = env.observation_space.shape
# state_size = env.observation_space.shape(0)
number_actions = env.action_space.n 
print('State shape: ', state_shape)
print("Number of actions : ", number_actions)

learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99

from PIL import Image
from torchvision import transforms
def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    return  preprocess(frame).unsqueeze(0)

class Agent():
    def __init__(self,action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.localnetwork = Network(action_size).to(self.device)
        self.targetnetwork = Network(action_size).to(self.device)
        self.optimizer  = optim.Adam(self.localnetwork.parameters(), lr = learning_rate)
        self.memory = deque(maxlen=10000)
        
    def step(self,state,action,reward,next_state,done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state,action,reward,next_state,done))
        
        if len(self.memory) > minibatch_size:
            experiences = random.sample(self.memory, k = minibatch_size)
            self.learn(experiences,discount_factor)
            
    def act(self,state, epsilon = 0.):
        state = preprocess_frame(state).to(self.device)
        self.localnetwork.eval()
        with torch.no_grad():
            action_values = self.localnetwork(state)
        self.localnetwork.train()
        
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self,experiences,discount_factor):
        states,actions,rewards,next_states,dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        next_q_targets = self.targetnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones) 
        q_expected = self.localnetwork(states).gather(1,actions)
        loss = F.mse_loss(q_expected,q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
agent = Agent(number_actions)

number_episodes = 2000
maximum_number_timesteps = 10000
epsilon_starting_value = 1.0
epsilon_decay_value = 0.995
epsilon_ending_value = 0.001
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen=100)

for episode in range(1,number_episodes + 1):
    state, _ = env.reset()
    score = 0
    
    for t in range(maximum_number_timesteps):
        action = agent.act(state,epsilon)
        
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state,action, reward,next_state,done)
        
        state = next_state
        score += reward
        
        if done:
            break
   
    scores_on_100_episodes.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
    if episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    if np.mean(scores_on_100_episodes) >= 100.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint2.pth')
        break
    
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('Boxing.mp4', frames, fps=30)

show_video_of_model(agent, 'Boxing')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()             
        