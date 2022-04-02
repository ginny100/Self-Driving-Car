# AI (brain) for Self Driving Car

# Importing the libraries
import numpy as np # playing with arrays
import random # playing with random samples from different batches of experiences
import os # useful for loading the model
import torch # becasue we're using pytorch - the recommended library for AI as it can handle dynamic graphs
import torch.nn as nn # nn is the neural network module including deep q-learning
import torch.nn.functional as F # functional is a package containing all the different functions we use to implement the neural network
import torch.optim as optim # some optimizer to perform stochastic gradient descent
import torch.autograd as autograd # a variable class to convert sensor into a variable containing the sensor and the gradient
from torch.autograd import Variable

# Creating the architecture of the Neural Network
class Network(nn.Module):
    # The init function
    def __init__(self, input_size, nb_action): # number of input neurons, number of output neurons corresponding to number of possible actions
        super(Network, self).__init__() # Inheritance trick to use all the tool of the module
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) # Establish the 1st full connection (30 neurons in the hidden layer, this is a good number achieved from different experience)
        self.fc2 = nn.Linear(30, nb_action) # Establish the 2nd full connection (30 neurons from the prev full connection now becomes the input size of the 2nd full connection)
    
    # The forward function used to active the neurons in the neural network using the rectifier activation function and eventually return the Q-values
    def forward(self, state):
        x = F.relu(self.fc1(state))  # Activate the hidden neuron
        q_values = self.fc2(x) # Q-values for each possible action
        return q_values
    
# Implementing Experience Replay
class ReplayMemory(object):
    # The init function
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] # Should always maintain the same capacity
    
    # The push function
    def push(self, event):
        # Appends new events (transitions) to the memory
        self.memory.append(event)
        # Makes sure that the memory always maintains the same capacity
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    # The sample function
    def sample(self, batch_size):
        # Get the samples from our memory and format them
        samples = zip(*random.sample(self.memory, batch_size)) # From the random lib, we're gonna use the sample function
        # The lambda func takes the samples and concatenate them with respect to the 1st dimension
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q-Learning
class Dqn():
    # The init function
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = [] # A sliding window of the evolving means of the last rewards
        self.model = Network(input_size, nb_action) # Our neural network
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) # We use the Adam class in this project, but we can choose different classes too. lr is the learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # Create a Tensor class - a fake dimension corresponding to the batch. By inputting 0 to unsqueeze(), the fake dimension now becomes the 1st dimension
        self.last_action = 0
        self.last_reward = 0
    
    # The select right action function
    def select_action(self, state):
        # We have 1 input state and 3 possible actions, so we have 3 Q-values
        # We're going to generate a distribution of probabilities with respect to these Q-values
        '''
        Convert state from a torch.Tensor() to a Torch Variable.
        Set volatile to True to exclude the gradient associated to this input state from the graph of all computations of the nn module.
        7 is the Temperature param. The higher is the Temparature param, the higher will be the probability of the winning Q-value.
        Set the Temperature to 0 if you don't want to activate the AI
        '''
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 100)
        # Take a random draw from the above distribution to get our final action
        action = probs.multinomial(1)
        return action.data[0, 0]
    
    # The learn function
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Get the model outputs of the input state of the batch_state
        '''
        With the gather() function, each time, we'll gather the best action to play for each of the input state of the batch state
        Note that the batch_state here has the fake dimension corresponding to the batch, while the batch_action doesn't have it.
        Change the datatype of batch_action so that it'll be suitable for the gather() function later
        Therefore, we need to add unsqueeze() to the batch_action. 1 corresponds to the fake dimension of the actions.
        Finally, we need to kill the fake batch with squeeze() as we're now out of the neural network. We have our outputs, but we don't want them in a batch, we want them in a simple vector of outputs
        '''
        outputs = self.model(batch_state).gather(1, batch_action.type(torch.int64).unsqueeze(1)).squeeze(1)
        # Get the maximum of the Q-values of the next state
        '''
        Use detach() to detach all the outputs of the model bc we have several states in this batch_next_state
        We pass 1 in the max() function bc the action is represented by the index 1, and we want to get the best action
        The next state is represented by the index 0, therefore we do [0]
        '''
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # Calculate the target
        target = self.gamma * next_outputs + batch_reward
        # Calculate the temporal difference loss
        td_loss = F.smooth_l1_loss(outputs, target) # smooth_l1_loss() is the recommend loss function for Deep Q-Learning
        # Update the weight with stochastic gradient descent
        self.optimizer.zero_grad()
        # Back propagate the loss into the network
        td_loss.backward(retain_graph = True)
        # Update the weight according to the propagation
        self.optimizer.step() # The step() will update the weight
    
    # The update function
    def update(self, reward, new_signal):
        # Make an update when reaching a new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) # Convert our new_signal into a torch.Tensor() to get the new_state
        # Update the memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # Play an action
        action = self.select_action(new_state)
        # Make sure that the memory contains more than 100 samples, so that the AI can learn from them
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            # Learn from the random batches
            #self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            self.learn(batch_state, batch_next_state, batch_action, batch_reward)
        # Update the last action
        self.last_action = action
        # Update the last state
        self.last_state = new_state
        # Update the reward
        self.last_reward = reward
        # Update the reward_window, make sure that it has a fixed size
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        # Return the action we play when reaching a new state
        return action
    
    # The score button that calculates the mean of the reward window
    def score(self):
        return sum(self.reward_window) /(len(self.reward_window) + 1.0)
    
    # The save button that saves the brain of the car so that we can use it after we close and reopen the app
    def save(self):
        # Save the last updated weight of our model
        torch.save({'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')
    
    # The load button
    def load(self):
        if os.path.isfile('last_brain.pth'):
            # If the last brain exists
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done!')
        else:
            # If the last brain does not exist
            print('no checkpoint found...')