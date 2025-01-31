import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

#As before, we are gonna start by building the network class which will include the whole architecture of the A3C Model which will include the convolutional layers to build eyes, the fully connected layers to build neurons and the forward method to forward propagate the signal from the input images up to the final output layer.
#Rest of the A3C class will be implemented in the agent class later on.

class Network(nn.Module):
    def __init__(self,action_size):
        #We havent taken seed as an argument this time as we want all our agents to have different seeds, we'll initialize them later.
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3,3), stride = 2)#In the deep convolutional q learning model, we had three input channels here but actually now, we are gonna have four. It doenst correspond anymore to the three RGB channels for colored images because here actually for our A3C Model, we are gonna have a stack of four gray scale frames.
        #There is no rule of thumbs here for choosing the optimal numbers here. These values after a lot of experimentation were found to be good.
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)
        self.conv3 = nn.Conv2d(32,32, kernel_size=(3,3), stride=2)#This time we are not decreasing or increasing anything as we go down to creating more convolutional layers just because here, it doesnt improve the score or affect it in anyway.
        #Similarly, we dont have any batch normalization layers as they dont help in imporoving the score in this particular model.
        #Now, we are actually gonna create a flattening layer using the torch library's flatten() class.
        self.flatten = nn.Flatten()
        #We dont have to input anything here, the transition will happen thanks to the forward method.
        #Now, we are gonna start creating the fully connected layers.
        self.fc1 = torch.nn.Linear(512,128)
        #To know how we got this 512, we can put the architecture in chatgpt and ask it how we got that.
        #Now, we are gonna directly move on to the final output layer because indeed first, we are not gonna have any second intermediary fully connected layer.
        #But, here we are gonna have two output layers.
        #The first one will correspond to the action values which correspond to the q values of the possible actions.
        #The second one will correspond to the state value which will be a single output of the network that will provide an estimate of the value of the current state and that value is actually a prediction of the expected return from the current state if the agent follows the current policy.
        self.fc2a = torch.nn.Linear(128,action_size)
        self.fc2s = torch.nn.Linear(128,1)
        
        #Now, we are gonna forward propagate the singal from the input images up to the final output layers of the action values and the state value.
    def forward(self,state):
        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        #We are done forward propagating the signal from the convolutional layers and now we want to further forward propagate it to the flattening layer.
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.fc2a(x)#Putting the x through the output layer a to get the action values.
        state_value = self.fc2s(x)[0]#Puting the x through the output layer b to get the state value. To not get it in any form of array of anything, we'll just add the [0] to just access the value.
        return action_values,state_value
    
#Now, our first step here is to setup the environment.
#As setting up the kung fu environment is a bit more complex and therefore we need this PreprocessorAtari Class to preprocess it.

class PreprocessAtari(ObservationWrapper):
    #First, it has the init method, the constructor method which defines the properties of the environment with the different object variables here, like the color, the observation space, of course, and the frames.
    def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (height, width)
        self.crop = crop
        self.dim_order = dim_order
        self.color = color
        self.frame_stack = n_frames
        n_channels = 3 * n_frames if color else n_frames
        obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.frames = np.zeros(obs_shape, dtype = np.float32)

    #Then, we have this reset method to reset the environment.
    def reset(self):
        self.frames = np.zeros_like(self.frames)
        obs, info = self.env.reset()
        self.update_buffer(obs)
        return self.frames, info

    #Then the observation method to preprocess images of the environment.
    def observation(self, img):
        img = self.crop(img)
        img = cv2.resize(img, self.img_size)
        if not self.color:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.astype('float32') / 255.
        if self.color:
            self.frames = np.roll(self.frames, shift = -3, axis = 0)
        else:
            self.frames = np.roll(self.frames, shift = -1, axis = 0)
        if self.color:
            self.frames[-3:] = img
        else:
            self.frames[-1] = img
        return self.frames

    #The update buffer method is to update the buffer.
    def update_buffer(self, obs):
        self.frames = self.observation(obs)
#Then, we have our make_env class that will actually create the env variable.
def make_env():
    env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
    env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
    return env

env = make_env()

state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("State shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.env.env.get_action_meanings())
        
#We are gonna define the hyperparameters now.
learning_rate = 1e-4
discount_factor = 0.99
number_environments = 10 #We actually train multiple agentss in multiple environments in parallel.

#Now, we are gonna implement the agent class but this time it wont be an epsilon greedy policy, but a softmax policy.
#But, like before we will not implement the learn method but we will integrate the logic and function of the learn method in the step method only.

class Agent():
    def __init__(self,action_size):#As the inputs are in the form of frames so there is no state_size but just the action_size.
        self.device = torch.device("cuda : 0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(action_size).to(self.device)#We are not gonna implement two separate networks like we did before as we are doing A3C now, so there is just gonna be one network.
        self.optimizer = optim.Adam(self.network.parameters(),lr = learning_rate)
    
    #Next, we are gonna implement the act method which is gonna allow our agent to play an action according to the softmax strategy.
     
    def act(self,state):
        #First thing we are gonna do is make sure that our state is in a batch, we always need that extra dimension as thats what pytorch library expects.
        if state.ndim == 3: #if the dimensions is just 3then we gonna add an extra dimension.
            state = [state] #So, what this will do is add an extra dimension which will make the dimensions total= 4, which is exactly what we need.
        
        #Now, as we have our state with the batch number, we are gonna convert it to a torch tensor.
        state = torch.tensor(state, dtype = torch.float32, device=self.device)
        
        #Since we have exactly what is needed by our network in order to get the action values.
        #And as we remember, our forward method in the network class returns the state_value too but since we dont need it in the act method, we are simply gonna put an underscore here.
        action_values, _ = self.network(state) #We can just write this to call the forward method as the torch library knows what we would want by calling the network.
        #Now, after getting the action_values, what we need to do is apply the softmax strategy to select a certain action.
        
        policy = F.softmax(action_values, dim = -1) #Dimension  -1 means that the last dimension of the tensor.
        #The state is actually a batch of several states and therefore  what we are bound to return now is several actions, each one corresponding to each state in the batch. And, since we are returning several actions, we are gonna return them into a numpy array.
        return np.array([np.random.choice(len(p), p = p) for p in policy.detach().cpu().numpy()])
        #Policy here started as a tensor, a torch tensor, then we detached it from the computational graph, then we moved the tensor back to the CPU, and then we converted that tensor into a NumPy array.
    
    #Now, we are gonna implement the step method which will be called whenever the agent takes a step in the environment and its a method that will receive the current state, the action taken, the reward received, the next_state reached and whether the epsiode is done or not.
    #It will also update the model's parameters, meaning the weight of the neural network so that it performs over the training better and  better actions that lead to a higher score.
    #The step method will take batches of everything.
    def step(self, state, action, reward, next_state, done):
        batch_size = state.shape[0] #First step will be to get the batch size in a variable because ofcourse that will be one of the parameters we will need in all the upcoming computations.
        #All the inputs we have given are gonna be in the form of numpy arrays so first we have to convert them into tensors before anything.
        state = torch.tensor(state, dtype=torch.float32,device = self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32,device = self.device)
        reward = torch.tensor(reward, dtype=torch.float32,device = self.device)
        done = torch.tensor(done, dtype=torch.bool,device = self.device).to(dtype=torch.float32)#As we have to convert the bool to float forcomputations.
        action_values,state_value = self.network(state) #Getting the action-values and state_value using state.
        _, next_state_value = self.network(next_state) #we wont need the action_values for this so we will disregard them.
        #Now, we are gonna get the state_value by bellman equation too and for that we have this formula.
        target_state_value = reward + discount_factor * next_state_value *(1- done)
        #Now, we are gonna implement the advantage feature of our A3C Algorithm which is formulated like this.
        advantage = target_state_value - state_value
        #Now, we are gonna calculate the actor's loss and the critic's loss.
        #For calculating the actor's loss, we need entropy, and in order to calculate the entropy, we need the distribution of probabiliies over the action values and also the log probabilities over the same action values.
        probs = F.softmax(action_values, dim=-1)
        log_probs = F.log_softmax(action_values, dim=-1)
        #Now, we have everything we need to calculate the entropy and it goes by,
        entropy = -torch.sum(probs*log_probs, axis = -1)
        #Now, we need one more thing to calculate the actor's loss which is the log probabilities of the actions that are actually selected. 
        batch_idx = np.arange(batch_size)
        #We are gonna use this index to select the log probabilites of the actions that are taken in the batch.
        logp_actions = log_probs[batch_idx, action]
        actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        #We take our log proabilities of the actions multiplied by the advantage, now we are just gonna add to our advantage tensor a.detach to prevent the gradients from flowing into the critic network during the actors update and we are gonna take mean of that, and we are gonna add at the end an exploration feature, which will be the mean of the entropy multiplied by a verysmall number which will balance the importance of the entropy.
        critic_loss = F.mse_loss(target_state_value.detach(),state_value)
        #We get the critic loss by using the mse_loss function on target_state_value and state_value.
        #And, now we are gonna calculate the total loss.
        total_loss = actor_loss + critic_loss
        #Now we just need to back propagate it and before that, we need to reset the optimizer.
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()#updates model's parameters.
        
        #here, we will have our agent class instance created.
agent = Agent(number_actions)
        
        #Now, we are gonna evaluate our A3C agent on a certain number of epsiodes.
#We are gonna use this method to evaluate the score and its gonna take as argument env,agent and the number of epsiodes.
def evaluate(agent,env, n_episodes = 1):
    episodes_rewards = [] #We are gonna initialize this as an empty list and it'll contain the different total accumulated rewards for each of the epsidoes onw which we evaluate our agent.
    for _ in range(n_episodes):
        #We are gonna use a certain function from our environment object, which is of course the reset method. This reset method actually return the state which is ofcourse the initialized state, but also some other elements which we are gonna disregard again with an underscore.
        state, _ =  env.reset()
        #As, we want to get accumulated reward over each episode and since we are dealing with a certain episode.
        #We need to calculate a new local varaible that willcompute the total reward.
        total_reward = 0
        while True: #Using this while loop, we can break if the episode is done because we have our done variable.
            #The first thing the agent needs to do isofcourse, act.
            action = agent.act(state) #The action is played and we reach that next state.
            state, reward, done, info, _ = env.step(action[0]) #This is how we are gonna get our next states and other things that we need,by using the step method of the env variable. [0] is because action is still numpy array here,and assuming that we are evaluating the agent in a non batch modewhile  we need to add this index zero.
            total_reward += reward
            if done:
                break
        episodes_rewards.append(total_reward)#Now, we will add our total reward to our list.
        #Now, we need to exit the for loopand return the episode_rewards variable.
    return  episodes_rewards
    
#Finally, we are gonna implement the asynchronous part.
#Now, we are gonna create a class that will help us to manage multiple environments simultaneously.

class EnvBatch:
    def __init__(self, n_envs= 10):
        self.envs = [make_env() for  _ in range(n_envs)]#This will create the number of envs needed and put them in a list.
        
    #The next method we willcreate will reset all the environments at once.
    def reset(self):
        _states = []
        #We are gonna use a for loop here to iterate through the list and reset them.
        for env in self.envs:
            _states.append(env.reset()[0]) #This will reset all the envs at once and store the states in the list we created.
        return np.array(_states) #We need to create the   numpy array out of the list because openAI Gym requires this type of data.
    
    #We are gonnna make a final method of this Env Batch which will allow us to step in multiple environments simultaneously
    
    def step(self, actions):
        next_states, rewards, dones, infos, _ = map(np.array,zip(*[env.step(a) for env,a in zip(self.envs,actions)]))#Here, we are running two for loops together one for actions and one for envs, and then we are zipping them together and mapping them to form an array.
        #Before, returning everything, we need to check if any of these environment is finished, if done is true,because if thats the case, we gonna have to return that very environment first.
        for i in range(len(self.envs)):
            if dones[i]:
                next_states[i] = self.envs[i].reset()[0]
        return next_states, rewards, dones, infos
        

#Now, we are finally gonna start the training of our agent as we have everything we need.
import tqdm #It will used to show the progress in a bar.

#We'll create an instance of the EnvBatch class we'll make 10 environments with it.
env_batch =EnvBatch(number_environments)
batch_states = env_batch.reset() #This will reset and return all the states.

#Final training For Loop
#We are gonna implement a growing progress bar too.

with tqdm.trange(0,3001) as progress_bar:
    for i  in progress_bar:
        #As the first things the agents have to do is play an action on the batch_states.
        batch_actions = agent.act(batch_states)
        batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
        #Aswe are dealing with actually high rewards,we need to reduce the magnitude of these rewards in order to stabilize the training.
        batch_rewards *= 0.01
        agent.step(batch_states,batch_actions,batch_rewards, batch_next_states, batch_dones)
        batch_states = batch_next_states
        if i % 1000 == 0:
            print("Average Agent Reward : ", np.mean(evaluate(agent, env, n_episodes=10)))
            
# Part 3 - Visualizing the results

import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env):
  state, _ = env.reset()
  done = False
  frames = []
  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action[0])
  env.close()
  imageio.mimsave('kungfu.mp4', frames, fps=30)

show_video_of_model(agent, env)

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
                  
        
        
    
        
        
        
