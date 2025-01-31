import os #provides function for creating and removing a directory(folder), fetching its contents, changing and idetifying the current directory, etc.
import random #for random parameters, random variable generators
import numpy as np #in order to work with arrays and mathematics
import torch #Because we'll build and train our libraries with pytorch
import torch.nn as nn #neural network module from the torch library
import torch.optim as optim #for the optimizer when training the AI
import torch.nn.functional as F #because we'll use some functions during the training
import torch.autograd as autograd #The torch.autograd module is used to calculate gradients of tensors with respect to other tensors.
from torch.autograd import Variable #In order to create some torch variables
from collections import deque, namedtuple

#Now we will create the Architecture of our Model
class Network(nn.Module):
    #Observation Space - The state is an 8 Dimensional vector : The coordinates of the lander in x and y, its linear velocities in x and y, its angle, its angular velocities and two booleans that represent whether each leg is in contact with the ground or not.
   #Ovservation space is the 8 dimensional vector describing the state of the agent or AI at each time step
    #Action Space - 0 : Do nothing, 1 : fire left orientation engine, 2 : fire main engine, 3 : fire right orientation engine
    #Seed is used to initialize the random number generator and we are adding a default seed of 42, just to fix a seed for the randomness.
    def __init__(self, state_size, action_size, seed = 42):
        super(Network, self).__init__()
    #In super, we enter Network and self as arguments just in order to activate the inheritance. 
    #First, we gonna activate the seed
        self.seed = torch.manual_seed(seed)
    
    #Then, we'll create the first fully connected layer, which is gonna include input layer(which has same number of neurons as the state_size which is 8) and a hidden layer(which has 64 neurons as they are found to be optimal by trial and error.)
        self.fc1 = nn.Linear(state_size,64)
    #Then, we'll create another fully connected layer, which will need us to specify the number of neurons from the previous hidden layer and the number of neurons of the new hidden layer which is again found to be good at 64 neurons
        self.fc2 = nn.Linear(64,64)
    #Then, we'll create the final connected layer, which is gonna connect the last hidden layer witht the output layer with the size action_size which is 4
        self.fc3 = nn.Linear(64, action_size)
          
    #Now, we are gonna build the forward method inside the network class, which is gonna forward propagate the signal from the input layer to the output layer through our two fully connected layers.
    #It is going to take as input first self and then state as arguments because its going to propogate the signal from the state to the output layer in order to play a certain action at a ceratin time stamp
    def forward(self,state):
        #we are gonna use our fc1 object because this is an instance of the linear class, therefore returning the first fully connected layer.
        #Its gonna take input "state" which is the current state at a certain time step in the environment.
        #As it returns the first fully connected layer, we gonna store that output in the variable x.
       x = self.fc1(state)
       #then we are gonna activate this signal with the relu function, the rectifier activation function.
       #relu function introduces non linearity. It is defined as f(x) = max(0,x), where x is the input to the function. In other words, if the input is positive, it returns the input value; if it is negative, it returns zero.
       #and we are gonna import it using the function module we imported.
       x = F.relu(x)  
       #we are gonna do the second part of the forward propagation,meaning from the first fully connected layer to the second fully connected layer.
       x = self.fc2(x)
       x = F.relu(x)
       #The second fully connected layer takes the input from the first fully connected layer and then we activate the signal of the second fully connected layer with the relu function.
       return self.fc3(x)
       #Then we call our third full connection fc3, which takes as input x which is now fully activated with the relu function.
       
#Training the AI
#Setting up the Environment
import gymnasium as gym
#Now we will create an variable which is gonna be the environment itself, and we gonna use the library gymnasium to import our lunar landing environment.
env = gym.make('LunarLander-v2', render_mode = "rgb_array")
#Now, we are gonna define some important variables while setting up the environment
#First, we gonna create the variable state_shape which we gonnna get from our env variable itself.
state_shape = env.observation_space.shape
print("Observations shape: ", state_shape)
#Then, we gonna create the variable state_size which is basically gonna consist the number of inputs, which is the first element of the shape of the obervation space.
state_size = env.observation_space.shape[0]
print("State size :", state_size)

#We gonna create the variable which is gonna specify the number of actions that can be taken and we gonna import the action_size from the env variabe itself.
number_actions = env.action_space.n
print("Actions Size : ", number_actions)

#Now, we gonna specify the hyperparameters
learning_rate = 5e-4 #5 * 10^-4
#Suitable value found after trial and error

minibatch_size = 100
#Refers to the number of observations used in one step of the training to update the model parameters.

discount_factor = 0.99
#The discount factor determines the importance of future rewards. A factor of 0 will make the agent short-sighted by only considering current rewards, while a factor approaching 1 will make it strive for a long-term high reward.

replay_buffer_size = int(1e5)
#It is the experience replay, thats the size of the memory of the AI. How many experiences including the state, action, reward, etc in the memory of the agent, the training to sample and break the correlations in the observation sequences. The purpose of this is to stabilize and improve the training process.

interpolation_parameter = 1e-3
#Interpolation parmeter used for the training, so thats the parameter that will be used in the sub update of the target networks. Interpolation is estimating or measuring an unknown quantity between two known quantities.

#Implementing Experience Replay
#Usually the name given to the class to implement Experience Replay is Replay Memory. we are specifying object in the paranthesis meaning that there is no inheritance this time.
class ReplayMemory(object):
    #First, we gonna start the constructor method to initialize the replay memory object. Constructor method which is the __init__ method, which will take self to refer to the object and another argument called capacity (the capacity of the memory.)
    def __init__(self, capacity):
        #first, we gonna implement a line of code that will be useful to use the GPU by calling th torch library from where we gonna call the device function in which we gonna specify ("cuda : 0") meaning we are gonna use the GPU.
        self.device = torch.device("cuda : 0" if torch.cuda.is_available() else "cpu")
        #This means, if torch.cuda. , gpu is available, then it will use the GPU, else it will use the CPU.
        #Now, to create our capacity variable, we gonna call our object variable which we gonna initialize to our capacity argument here which will be specified later on when we create the instance of this class.        
        self.capacity = capacity #Capacity is the maximum size of the memory buffer.
        #Then we gonna create another variable, which is gonna be memory, which is gonna be an empty list which will store the experiences, each containing the state, the action, etc.
        self.memory = []
        
    #Now, we gonna create a push method which will push an experience into the memory buffer. As, we have chosen replay_buffer_size to be 100000, this is the method thats gonna add the experiences into that memory while also checking that we dont exceed the memory.
    #We are gonna give two arguments, one is gonna be the object itself and the other one is gonna be the event or experience which is gonna contain the state, the action, the next state, the reward and that boolean saying if we are done or not.
    def push(self,event):
        #Now we gonna take our memory variable which was defined in the constructor method, and we gonna use the append function just to append the event into the list.
        self.memory.append(event)
        #Now we gonna make sure that the memory buffer doesnot exceeds its capacity after we appended that event. So, we gonna add an if condition which is gonna check the length of the memory is larger than the capactiy, we gonna delete the oldest event.
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    #Now, we gonna implement the last method of this replay memory class which is gonna be the sample method which will randomly select a batch of experiences from the memory buffer.
    #This method sample is gonna take two arguments, one which is gonna be the object itself and the other is gonna be the batch_size which are gonna be the number of experieces or events that are gonna be sampled in the batch.
    def sample(self,batch_size):
        #First, we gonna create a new local variable which we gonna call experiences, which will contain the experiences sampled in the batch. For which, we gonna use the random.sample method which is gonna take a random sample from the memory variable, and another argument which is gonna take the batch_size.
        experiences = random.sample(self.memory, k = batch_size)
        #Now, we gonna extract different elements of each sampled experience(the action, the state, the next state, etc.) So, we gonna extract them and stack them one by one. We gonna do this by using the numpy library from which we gonna call the vstack function that will stack the elements in the sampled experiences together. And we gonna use for loop to iterate and extract the elements for stacking.
        #But, we must check each experience if it is existent and for that we gonna add an if statement that will check if it is none or not.
        #Then after stacking and checking, we also need to convert this stack into PyTorch Tensors because when we are gonna use the Pytorch to train and especially the neural network when updating the model parameters through back propagation. We gonna use the torch library's method for that which is torch.from_numpy.
        #We also need to make sure that the data type of these tensors is float in order to give a certain format expected by the torch library functions. For that, we gonna add the .float() function at the end of the statement.
        #Because we also have our device, we are just gonna add at the end .to(self.device), thats just in order to move this stack into the designated computing device whether it is CPU or GPU.
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        #Then, similarly we gonna make another variable actions, which is gonna contain the all the actions stacked and converted into tensors but this time we done need actions to be float as they are gonna be 0,1,2,3 so we gonna convert them into long integers.
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None ])).long().to(self.device)
        #Now,we gonna create the variable rewards which is gonna be similar and the third element and a float.
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        #Next, new_states which are also gonna be floats.
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        #Next, dones are not gonna be directly floats, we gonna have to first set the data type to boolean which is uint8, and the way we are gonna do this, we are gonna add .astype before .float and in astype we gonna get the boolean uint8 using numpy.
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        #Now, we gonna return return states, next_states, actions, rewards, dones in the same order.
        return states, next_states, actions, rewards, dones
    
#Now we gonna create the DQN Class (Deep Q Network), and we gonna name it Agent
#We dont have anything to specify in the parathesis this time.

class Agent():
    #Starting with the constructor method
    #First, we gonna self to refer to the object and then state_size and action_size as arguments for our class. 
    def __init__(self, state_size,action_size):
        #Again, we gonna use the torch library's device method to make sure we are using our gpu if it is available.
        self.device = torch.device("cuda : 0" if torch.cuda.is_available() else "cpu")
        #Now, state_size and the action_size are arguments  that we need to pass when creating an instance of our class and also, we still need to create the object variable.
        self.state_size = state_size
        self.action_size = action_size
        #We are now gonna create two Q networks,the local one and the target one.
        #As, we have created our network class already, we just need to create two instances of that class.
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size,action_size).to(self.device)
        #So, here we have created two instances of the Network class, the local one and the target one and  assigned them both to GPU.
        #Now, we have passed the arguments needed by class.
        #We are gonna create an optimizer as an object variable which is very important for the training of the AI.
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(),lr=learning_rate)#Then, we gonna call an attribute of this local Q Network, which is parameters which are exactly the weights of the network, what will be update step by step to predict better.
        #It also takes learning rate as an argument.
        #The adam class comes from the optim module, from which we are taking the adam class which take arguments, first,the local parameters meaning the weight of our local Q Network, so we are taking our local q network and we are basing it inside the adam optimizer.
        #Now, we need to implement the memory of the AI in this DQN Class using the memory replay class.
        self.memory = ReplayMemory(replay_buffer_size)
        #Replay memory class takes capacity as argument which we have already specified in the hyperparameters as replay_buffer_size.
        #We need to initialzie another time step, time step counter, for deciding in which moment we are gonna learn and update the network parameters.We gonna call it t_step.
        self.t_step = 0
        
    #Now, we are gonna implement the step method in the agent class which is gonna store experiences and decide when to learn from them.
    #It is gonna take as arguments, first of all, the object itself, and ofcourse, the experiences, but we are not gonna directly take whole experiences as an argument, we are gonna decompose it in its elements(state, action, reward, next_state, done)
    def step(self,state,action,reward,next_state,done):
        #The first thing its gonna do is to store the experience in the Replay memory.Since, we already created our memory object as an instance of the ReplayMemory Class, we are gonna call our self.memory object.
        #As, we have already implemented our push method in that class, we are gonna call it to store the experience from the self.memory object.
        self.memory.push((state,action,reward,next_state,done))
        #The step method does two things, first it stores the experience and then it decides when to learn from them. 
        #So, for the learning part, we'll need to increment the time step counter and reset it every four steps, so that we can learn every four steps which we gonna implement in the learn function.
        self.t_step = (self.t_step + 1)%4 #Here, we have incremented it by 1 and divided it by 4 to check if it is divisible.If the remainder is 0, it learns.
        if self.t_step == 0:
            #First, we check if the 4 timestamps have passed also, and also we check if the number of experiences in the memory is already larger than this minibatch_size of 100, so we are gonna continue this if statement with another if statement and because self.memory is the attribute of the Replay Memory class, we write self.memory.memory.
            if len(self.memory.memory) > minibatch_size:
                #So, for that, we first gonna get our experiences in a local variable also called experiences, for that we gonna use our sample method that we created in the ReplayMemory class.
                experiences = self.memory.sample(100)
                self.learn(experiences,discount_factor) #Right here, we have just called the learn method and we gonna define it later in the same class.
                
    #Now, we gonna implement act method in this class, that will select an action based on a given state and a certain epsilon value for an epsilon greedy action policy.
    #It is gonna take three arguments, first self to refer to the object and state because it will select an action based on the current state or given state. 
    #And, as we gonna do epsilon greedy action selection policy, so its gonna take epsilon as the last argument. We gonna give it a default value to make it a float.
    def act(self, state, epsilon=0.):
        #first, we gonna start with the state by converting it into a tensor for it to be used by the neural network later on.
        #Important for reinforcement learning, we need to adhere an extra dimension to our state vector, right now, we have 8 dimensions of the state corresponding to the coordinates of the spaceship,but, we need to add an extra dimension which will correspond to the batch,meaning, this dimension will say which batch does this state belongs to.
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)#Now, using this unsqueeze method, first dimension of this state will correspond to the batch.
        #Now, our state is to updated as the torch tensor with an extra dimension added and pushed in the gpu.
        #Now, because we are about to forward pass the state through the local Q network to get the action values, first we are gonna set it to evaluation mode. We gonna do this by .eval function which comes nn.module class which the Network class inherits.
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
    
    #Now, we gonna implement the learn method of the agent class that will update agent's q values based on sampled experiences.
    def learn(self, experiences, discount_factor):
        #its gonna take arguments self to refer to the object, experiences and the discount_factor.
        states,next_states,actions,rewards,dones = experiences
        #We are gonna create these new local variables here and they are gonna be equal to the experiences which is the tuple of exactly these elements. We are basically just unpacking the elements of the experiences.
        #The next step is to get the maximum predicted q values for the next states from the target network and that is because we are gonna compute the q targets for the current states,which include those maximum predicted q values in the formula.
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        #The first thing we gonna do is forward propagate the next state from our target q network.
        #This gives us the action values target q network propagating the next state and therefore now, since we want to take the maximum of those action values, we are gonna use the max function to get the maximum, but before that we actually have to detach the action values in the tensor in order to then get the maximum of them. The detach function actually detaches the resulting tensor from the computation graph(meaning we wont be tracking gradients for this tensor during the back propagation.)
        #And then, in this max function we have to input a one here because we need the maximum value along dimension one which corresponds to the action dimension in the tensor. We also need to add here a square bracket zero because after giving max(1) we get two tensors,a tensor of maximum values and a tensor of indices corresponding to the actions for these maximum values. And to select the maximum action values tensor we add 0 at the end.
        #We gonna have to add a .unsqueeze again at index 1 because we'll need to add again the dimensions of the batch.
        #And, this gets us the maximum predicted Q values for the next states from the target network. 
        #Now, we have that next Q target, we can compute the Q targets for our current state.
        q_targets = rewards + discount_factor *  next_q_targets * (1 - dones) #Formula for getting Q Targets.
        #Next step is to get the expected Q values from the local q network this time. So, this time we are gonna forward propagate the state from our local Q network as opposed to propagating the next state.
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
        #The final step is to update the target network parameters with those of the local network.
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)
        
    #Final method implementation of the agent class is gonna be the soft_update method, that will softly update the target's network parameters.
    #And it's gonna take as parameters self to refer to the object and its gonna take our local q networks, target q networks and its gonna take the interpolation parameter.
    def  soft_update(self, local_model, target_model, interpolation_parameter):
        #So, first its going to loop through the parameters of the local Q network and the target q network.
        #The trick to get them both at the same time is to use the zip function which will take the params of both the models as arguments.
        #We can get the parameters by using the parameters() function of the nn.module which is inherited by the Network class.
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            #It will softly update the target model parameters using the weighted average of the target and local parameters.
            #To update the params we are actually gonna use the copy function to update the parameters of the target Q network. 
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)#this is the formula for updating the params softly.
            
#Summary of the Agent Class
#We created this agent class that defines the behavior of an agent that interacts with out space environment using a deep Q Network.
#And, while it is interacting with the environment, while the agent maintains two Q networks, first, the local q network and then the target Q network.
#This Double Q network setup will stabilize the learning process that we implemented here.
#At the same time, while we have the soft update method that will update the target Q network parameters by blending them with those of the local Q Network with that formula we implemented here.THe purpose of this is to prevent abrupt changes, which could destabilize the training.
#We also implemented the act method that will help the agent choose an action based on its current understanding of the optimal policy. Those actions will be returned from the local Q network that will forward propagate the state to return the action values and the following an epsilon greedy policy, it'll return the final action. The fact that we can select random actions allows the agent to explore some more actions which could potentially lead to a better result at the end.
#It uses two separate networks: one for selecting actions (the local network), and another for estimating the future rewards (the target network).
#Then finally, the learn method uses experiences that are sampled from the replay memory in order to update the local Q networks q values towards the target q values.

#Now, we are gonna initialize our DQN Agent
agent = Agent(state_size,number_actions)

#Here, we gonna start training the Agent
#Before starting the training, we are going to initialize the training parameters.
number_episodes = 2000#The maximum number of episodes over which we want to train our agent.
maximum_number_timesteps_per_episode = 1000#The maximum number of time steps per episode, meaning that we dont wanna let our AI try too hard. If it keeps staying in the air,if it stays in a certain stuck of the environment.To avoid that we set this.
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
        #For each step, the reward:

        # is increased/decreased the closer/further the lander is to the landing pad.

        # is increased/decreased the slower/faster the lander is moving.

        # is decreased the more the lander is tilted (angle not horizontal).

        # is increased by 10 points for each leg that is in contact with the ground.

        # is decreased by 0.03 points each frame a side engine is firing.

        # is decreased by 0.3 points each frame the main engine is firing.

        # The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

        # An episode is considered a solution if it scores at least 200 points.
        
        #These are the reward policies for the agent.
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
    if np.mean(scores_on_100_episodes) >= 200.0:
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

show_video_of_model(agent, 'LunarLander-v2')

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