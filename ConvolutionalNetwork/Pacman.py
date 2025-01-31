#First, we gonna import all the necessary libraries.
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from collections import deque
from torch.utils.data import DataLoader,TensorDataset

#We are gonna start by building the AI, Creating the Architecture of the Neural Network
#We basically build the brain of our AI, but not only the brain this time, but also the eyes because we are gonna bring into the Neural Network some Convolutional Layers, which will act as eyes in order to be able to see the input images from the game which are nothing less than the states.
#We are gonna create the network class just like before but this time, the constructor method will not take state_size as argument, because this time, the state is an image. Its not a simple one-dimensional vector.
#The state_size is now a multi dimensional array representing the images size in three dimensions with RGB.
class Network(nn.Module):
    def __init__(self,action_size, seed = 42):
        super(Network, self).__init__() #This is just in order to call the initializer of the base class, meaning the module class by the NN module to correctly activate the inheritance by setting up the internal state of the class.
        self.seed = torch.manual_seed(seed)
        #This is exactlynow that we start building the brain of our AI, including the eyes, the whole neural network.
        #The first operation is going to be a convolution. Because the eyes comes first before the fully connected layers of neurons in the brain.
        self.conv1 = nn.Conv2d(3,32,kernel_size=8,stride=4) #We created this by calling our NN module from which we are gonna call the conv2D class. So conv1 will be an instance of that conv2D class.
        #Its gonna take as arguments Input Channels and since we are working with RGB here,there are gonna be 3 input channels and second, the Output Channels and after experimentation, for this particular game it was found good to be 32.
        #Third, its gonna take as argument is the kernel_size, kernel_size refers to the dimensions of the filter or convolutional kernel applied to the input data. After experimentation, it was found that 8X8 was good for the particular game.
        #The last argument would be stride, stride refers to the number of pixels by which the filter(or kernel) is shifted over the input data when performing convolution. The value that was good for this modelwas 4 and it was found by experimentation.
        #The next thing we gonna define is a batch normalization layer for 2D Inputs.These help in faster and more stable training.
        #We are gonna call it bn1 for the batch normalization operation. This variable will be created as an instance of a new class, which is gonna be the batch norm 2D class.
        self.bn1 = nn.BatchNorm2d(32)#It is gonna take as arguments the number of features, just like we do the full connections, its basically the number of channels or feature maps in the previous layer and the previous layer is a convolutional layer of 32 feature maps.
        #Now, similarly, we are  gonna create more pairs of convolutionallayer and batch normalization layer.
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        #The number of output channels  of  the previous one is gonna be the number of input channels on this one. The number of output channels of this one is gonna be 64. What matter is that we graduallly increase the depth, meaning the number of channels, while decreasing the spatial dimensions.
        #So, the number of channels will be increased with the different convolutional layers while the kernel size and the stride will be decreasing.
        self.bn2 = nn.BatchNorm2d(64)#It will take the number  of output maps in the previous one as its argument.
        #Now, we  are gonna create the third pair.
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)#The arguments were found to be good after a lot of experimentation.
        self.bn3 = nn.BatchNorm2d(64)
        #Now, the final pair.
        self.conv4 = nn.Conv2d(64,128,kernel_size=3,stride=1)#As we are gradually increasing depth, here we take the output size as 128.
        self.bn4 = nn.BatchNorm2d(128)
        #And, here we are done with making some artificial eyes.
        #Now,the convolution layers are followed bythe full connection layers.
        self.fc1 = nn.Linear(10*10*128,512)#Now, the first argument is gonna be the number of output features resulting from flattening all the previous the previous convolutions. This is the flattening layer. How many neurons there are in this flattening layer.
        #The formulae for caluclating the number of neurons in the flattening layer is the (((input size - kernel size) + 2 * padding)/stride) + 1, Now you gonna have to apply this formula for each convolution layer,meaning you gonna have to applythis formula 4 times.
        #Now, lets move on to the second fully connected layer.
        self.fc2 = nn.Linear(512,256)#The values that are not explained are just found  using experimentation.
        #Now,we are gonna create the last fully connected  layer.
        self.fc3 = nn.Linear(256,action_size)
        
        #Now, that we have created the brain and eyes of our model, we are gonna define the forward method that is gonna forward propagate the signal from first the Pac-man images to the eyes to the fully connected layers.
        
    def forward(self,state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        #Now, after forward propagating the signal through the convolution layers, we are gonna flatten our resulting tensor.
        #We have to reshape x in order to get that flattening layer, in order to fit into the fully connected layers.
        x = x.view(x.size(0),-1)
        #Now, we will forward propagate this signal through the fully connected layers to the final output layer.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        #Now, the forward propagation is done and we have our output signal.
        
#Now, we are gonna start with the training of our AI.
#The first step is gonna be setting up the environment.
import gymnasium as gym
env = gym.make('MsPacmanDeterministic-v0', full_action_space = False,render_mode = 'human')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State Shape: ',state_shape)
print('State Size :', state_size)
print('Number of actions: ', number_actions)

#Now, initializing the hyperparameters.
learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99
#We removed replay_buffer_size because we will implementing experience replay in a much simpler way.
#We removed interpolation parameter because we wont be doing any soft updates in this particular model for the pacman.
#Now, we are gonna start by preprocessing the frames.
#In the deep q learning, we implemented experience replay after initializing the hyperparameters but we cant do the same here.
#Because, the input states are no longer input vectors of a small size, they are now input images of a big size, because there are many more dimensions. We cant store 100000 experiences like before.
#But we'll still do a simpler implementation of experience replay
#We have to preprocess the frames so that the input images can be converted into Pytorch Tensors that are accepted by the neural network of our AI.
#And for that, we are gonna import a new class, which is called image, and which is taken from the Python Imaging Library (PIL), which will allow us to load images from files and create new images.

from PIL import Image
from torchvision import transforms#We also need the transforms module from the TorchVision Library.
#Now, we gonna start defining our preprocessing function.

def preprocess_frame(frame):
    #It is gonna take as input the real frame coming from the game of Pac Man which will be converted to PyTorch Tensors.
    #As, the frameis in the form of a NumpyArray which represents exactly the image.
    #Our first step is to convert this NumPy array, which is our frame,into a PIL image object.
    frame = Image.fromarray(frame)#We use the image class's from array method to convert the array to the PIL Image Object.
    #Now, we will preprocess our frames but in the form of PIL object.
    #We are gonna define a new preprocess object which will be created as an instance of  the compose class and this compose class is taken from the transforms module.
    #We are gonna call the Compose class, which will take as input only one argument here which is a list that will contain the different transformations we are gonna do through our preproess project. 
    preprocess = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
    #We are first gonna resize our frames, as we can see  in the state shape above, the original dimesions of our frames is 210,160,3.
    #We need not only to lower the dimensions but also make squares, which is why we are gonna have after the image pre processing dimensions of 128 by 128.
    #So after the resizing, we are gonna call our transforms module again, from which we are gonna call this time that ToTensorClass.
    #This transformation will also normalize the frames, meaning that will also scale the pixel values to the range between zero and one.
    return preprocess(frame).unsqueeze(0)#Adding an extra dimension for batch.

#Now we gonna create the DQN Class (Deep Q Network), and we gonna name it Agent
#We dont have anything to specify in the parathesis this time.

class Agent():
    #Starting with the constructor method
    #First, we gonna self to refer to the object and then state_size and action_size as arguments for our class. 
    def __init__(self,action_size):
        #Again, we gonna use the torch library's device method to make sure we are using our gpu if it is available.
        self.device = torch.device("cuda : 0" if torch.cuda.is_available() else "cpu")
        #Now, state_size and the action_size are arguments  that we need to pass when creating an instance of our class and also, we still need to create the object variable.
        self.action_size = action_size
        #We are now gonna create two Q networks,the local one and the target one.
        #As, we have created our network class already, we just need to create two instances of that class.
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        #So, here we have created two instances of the Network class, the local one and the target one and  assigned them both to GPU.
        #Now, we have passed the arguments needed by class.
        #We are gonna create an optimizer as an object variable which is very important for the training of the AI.
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(),lr=learning_rate)#Then, we gonna call an attribute of this local Q Network, which is parameters which are exactly the weights of the network, what will be update step by step to predict better.
        #It also takes learning rate as an argument.
        #The adam class comes from the optim module, from which we are taking the adam class which take arguments, first,the local parameters meaning the weight of our local Q Network, so we are taking our local q network and we are basing it inside the adam optimizer.
        #Now, we still need to implement experience replay, so the only thing we are gonna do here is implement a deque here with max len = 10000
        self.memory = deque(maxlen = 10000)
    #Now, we are gonna implement the step method in the agent class which is gonna store experiences and decide when to learn from them.
    #It is gonna take as arguments, first of all, the object itself, and ofcourse, the experiences, but we are not gonna directly take whole experiences as an argument, we are gonna decompose it in its elements(state, action, reward, next_state, done)
    def step(self,state,action,reward,next_state,done):
        #As we removed the push method as we have not implemented the replay memory class in this network, we just need to simply do an append.
        #As, state and next_state are not preprocessed here, first we are gonna have to preprocess them for them to be ready to be fed tothe network.
        #As, they are just numpy arrays.
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        #Now, we'll simply just append all the necessaries in the deque.
        self.memory.append((state,action,reward,next_state,done))
        if len(self.memory) > minibatch_size:
            #Now, as we dont have a sample method, we are gonna implement it here. By using the simple sample function fromthe random library.
            experiences = random.sample(self.memory ,k = minibatch_size)
            self.learn(experiences,discount_factor) #Right here, we have just called the learn method and we gonna define it later in the same class.
                
    #Now, we gonna implement act method in this class, that will select an action based on a given state and a certain epsilon value for an epsilon greedy action policy.
    #It is gonna take three arguments, first self to refer to the object and state because it will select an action based on the current state or given state. 
    #And, as we gonna do epsilon greedy action selection policy, so its gonna take epsilon as the last argument. We gonna give it a default value to make it a float.
    def act(self, state, epsilon=0.):
        state = preprocess_frame(state).to(self.device)
        #Now, as we have frames now, we are gonna convert the state into the pytorch tensor using our preprocess_frame function.
        self.local_qnetwork.eval()
        #As, we are now in evaluation state, we can forward pass this state through our local q network.
        #But before that, we gonna make another check to make sure we are not in training mode anymore but in inference mode, meaning that we are doing predictions by using torch library's no_grad function which will make sure any gradient computation is disabled.
        with torch.no_grad():#we don't want gradients here because we are evaluating not training.
            action_values = self.local_qnetwork(state)
         #Now, since we are done by getting those action q values predictions.We are gonna back to the training mode.
        self.local_qnetwork.train()
        #E-greedy strategy for choosing actions.
        #If it is less than epsilon then choose a random action else take the best action from the model.
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())#Since, that selection is a simple process, we are gonna send that to the CPU.
            #np.argmax function selects the max out of all and as numpy expects data in a certain format, thats why data.numpy() to convert it into that format.
        else:
            return random.choice(np.arange(self.action_size)) 
            #We are gonna return a random action by using the choice function by the random library by using numpy's arange function which gives us the 0,1,2,3 indexes to randomly choose from.
    
    #Now, we are gonna implement the learn method.
    #It is not much different from the learn method in the deep q learning.
    #Remember, we were computing the q_targets, the next_q_targets and the expected q values, we were doing all this on stack of states and stacks of actions and stacks of next_states.
    #Since we dont have the replay memory class to give us stack of states,next_states,actions, dones,rewards we are gonna do that here.
    def learn(self, experiences, discount_factor):
        states,actions,rewards,next_states,dones = zip(*experiences)
        #Here we are unzipping the experiences into its elements, and the star before experiencs is needed  to make sure that each element gets zipped together. It is in the syntax.
        #As we can see, the elements are already inthe form of tensors.
        #So, how we gonna do this, we gonna do this by using the same code we used in sample method in the experience replay class.
        #The numpy's vstack function can also take tensors as input and convert them to numpy arrays and stack them and then we gonna use the torch's from numpy function to convert them back into tensors.
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        #Then, similarly we gonna make another variable actions, which is gonna contain the all the actions stacked and converted into tensors but this time we done need actions to be float as they are gonna be 0,1,2,3 so we gonna convert them into long integers.
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        #Now,we gonna create the variable rewards which is gonna be similar and the third element and a float.
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        #Next, new_states which are also gonna be floats.
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        #Next, dones are not gonna be directly floats, we gonna have to first set the data type to boolean which is uint8, and the way we are gonna do this, we are gonna add .astype before .float and in astype we gonna get the boolean uint8 using numpy.
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor *  next_q_targets * (1 - dones) #Formula for getting Q Targets.
        q_expected = self.local_qnetwork(states).gather(1,actions)#Because we are gonna gather all the respeced q values.We are just gonna add 1 and actions in the paranthesis.
        #Now, that we have everything we need, we are gonna calculate the loss.
        loss = F.mse_loss(q_expected,q_targets)#We gonna use the mse_loss function from the F library, which is computing the mean squared error loss between q_expected and q_targets.
        #Now, that we have the loss so now we can backpropagate the loss in order to update the model parameters to then update the new Q values leading to a better action selection policy.
        #So, thats what we gonna do but before that we gonna initialize the optimizer by zeroing out the gradient of that optimizer.Basically, we are gonna do a reset by using the object we created, self.optimizer.
        self.optimizer.zero_grad()#By using the zero grad  method, we reset all of the gradients to zero before starting backpropagation.
        #Finally, time for back propagation, we are gonna back propagate the loss that we just computed to compute the gradient of that loss with respect to the model parameters.
        loss.backward()#Thats all we need to backpropagate this loss.
        #The next step is to now perform a single optimization step to update the parameters of the model.
        #We are gonna do this by calling the step method with our optimizer.
        self.optimizer.step()
        
        #Now, we are gonna initialize the DCQN Agent
agent = Agent(number_actions)

#Here, we gonna start training the Agent
#Before starting the training, we are going to initialize the training parameters.
number_episodes = 2000#The maximum number of episodes over which we want to train our agent.
maximum_number_timesteps_per_episode = 10000#The maximum number of time steps per episode, meaning that we dont wanna let our AI try too hard. If it keeps staying in the air,if it stays in a certain stuck of the environment.To avoid that we set this.
#Defining the hyperparameters for the epsilon greedy
epsilon_starting_value = 1.0 #the starting value for the epsilon
#Then, we are gonna let the epsilon value decay, meaning we'll decrease it little by little to test other epsilon values.
epsilon_ending_value = 0.01 #The ending value for the epsilon
epsilon_decay_value = 0.995#The value of the epsilon is gonna decay by this value
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)#This variabe will contain that window of all the scores on the last a 100 episodes. We are gonna do that with a double ended queue in order to keep track of the scores of the last 100 episodes. We are gonna use the deque() function to implement the double ended queue.

#We are gonna implement the final training loop.
for episode in range(1, number_episodes + 1):
    #Resetting the state and getting initial action from the environment. First step is to reset the environment to its initial state at the beginning of the every epsiode.
    state, _ = env.reset() #We are gonna take our state, which here is a new local variable, and then add a comma followed by an underscore because then we will call our environment and reset the function. This returns not only the initial state, but also some other information like example, the initial observation, which is discarded here with this underscore.
    #The next step will be to initialize the score because the most important thing we'll have to measure here is the score. Score means the cumulative reward, which is why we have to initialize this to zero.
    score = 0
    #So, this over the episode will be the cumulative reward.
    #Now, we are gonna start another loop, Its gonna be a loop over the time steps, the time steps over the episode.
    #We are gonna call our time steps T very simply, these time steps are gonna be in the range from zero to the_maximum_number_of_time_steps_per_episode.
    for t in range(maximum_number_timesteps_per_episode):
        #The first thing that the agent is supposed to do at a specific time step is to select an action.
        action = agent.act(state, epsilon) #Introducting a new local variable, action.
        #And, we are gonna do this by using the attribute we created in the Agent class, so we are gonna use the instance we created followed by the .act method which selects an action in a given state of the environment and following an epsilon greedy policy.
        #Now, it ends up in a  new state of the environment because it played this action and it also has the reward.
       
        #Now we reached the new state and got a reward, and now we gonna put it into new variables.
        #We'll also have a boolean done, to specify if we are done or not.
        #Since, we are going to use the step function here, which will also return some other information that we dont need, so we will just discard this information with a first underscore and another one.
        next_state, reward, done,_,_ = env.step(action) #It is not the step method we defined, it runs one timestep of the environment's dynamics using the agent actions.
        #Now, we are gonna use the step method which also includes the learn method, now we are gonna use the tools to perform the training.
        agent.step(state,action,reward,next_state,done)
        #This step is exactly what will train our agent to land better and better on the moon.
        #Now, we need to update our state variable as we are in a new state.
        state = next_state
        #Now, we have the reward too, score is a cumulative  sum of rewards.
        score += reward
         #Now, we will take care of the done, we are just gonna check if at this specific time step where we are in the second for loop.
         #Our episode is done, so, if done, meaning if the episode is done at this specific time step, we'll simply do a break, so that we can finish the episode.
        if done:
            break
    #After the episode is finished, we are gonna append the score of the finished episode to that windown of the score on 100 episodes.
    scores_on_100_episodes.append(score)
    #Now, we are gonna decay the epsilon value for the epsilon greedy policy while ensuring that it doesn't go below the ending value of 0.01. So, we are gonna do this by taking the max.
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
    #Here, the training process is done, now we are gonna start the printing process and we are gonna make a dynamic print, meaning that we will get the Average Score for each episode, but in a dynamic way.
    #Meaning that episode by episode we will see the average of the cumulative reward evolving over the episode. But we are gonna make an overriding effect, meaning that each of these lines printed will be removed to give place to the next one.
    #And, ofcourse we will keep the average reward over 100 episodes every 100 episodes.
    #First we will print those average scores for each episode with that overriding effect.
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode,np.mean(scores_on_100_episodes)),end = "")#here, since we have to choose the number of decimals we want to have up to the comma, we are gonna enter this :2f, meaning that we will want the average score to be printed as a float with two decimals after the comma.
    #We are gonna get the average by using the mean function of the numpy library.
    #The trick to do the overriding effect is to simply add at the very beginning here inside the quote \r which is called a carriage return in programming, meaning that the cursor will return to the start of the line when printing until when you use it in a loop.
    #We also need to add the end argument which we'll have to set equal to a pair of quotes meaning that after printing this line, it wont automatically add a new line. 
    #Now, we are gonna check whether we are every 100 episodes. We are gonna use modulo again, you know that percentage giving us the rest of a Euclidean division of an integer by another.
    if episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode,np.mean(scores_on_100_episodes)))
    #Now, we are gonna take a new if condition, which this time will check if the mean of our scores over 100 episodes is larger than 200 and we have to add .0 to make it a float, well, if the average scores_on_100_episodes is larger than 200, we will say, win.
    if np.mean(scores_on_100_episodes) >= 500.0:
        print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,np.mean(scores_on_100_episodes)))
        #IF the average of the scores on 100 episodes is larger than 200, that means we actually started winning from this episode - 100 because this is a score over 100 episodes.
        #Since, we have an intelligent AI that is able to land on the moon, it is time to save the model's parameters and we do that using the torch library's save function which will take as input two arguments. First, the model parameters we wanna save, then second, a file in which we want those model parameters to be saved.
        torch.save(agent.local_qnetwork.state_dict(),'checkpoint.pth')
        break#To exit out of the training loop
    
# Part 3 - Visualizing the results

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
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'Pacman')

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
        
        
         
   


