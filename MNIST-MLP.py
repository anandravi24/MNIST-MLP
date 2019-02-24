#################################################################################################################
# Author : Anand Ravi                                                                                           #
# Program : Implementation of Multi-Layer Perceptron                                                            #
# Arguments :  < Network Configuration [Seperated by comma]> < Epochs > < Minibatch Size > < Learning Rate >    #
# Example: python3 MNIST-MLP.py 200,100,50 50 100 0.01                                                          #
# Email : anandrav@usc.edu                                                                                      #
#################################################################################################################

import h5py
import numpy as np
import sys
import argparse

parser= argparse.ArgumentParser()
parser.add_argument("Network_Configuration", help="Enter the number of neurons in hidden each layer as tuple {Seperated by comma}",type=str)
parser.add_argument("Epochs", help="Enter the number of epochs",type=int)
parser.add_argument("Minibatch", help="Enter the minibatch size",type=int)
parser.add_argument("Learning_Rate", help="Enter the learning_rate",type=float)
args=parser.parse_args()


class Relu:
    ''' Class contains Relu functions in forward and backward direction '''
    
    def forward(self,x):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask
 
    def backprop(self,x):
        return self.mask


class Layer:
    
    def __init__(self,incoming_neurons,outgoing_neurons):
        ''' Constructor to initialze all the member variables of the layer object '''
        
        self.outgoing_neurons=outgoing_neurons
        dimension_weight = (outgoing_neurons,incoming_neurons)
        self.weight = np.random.normal(0,0.1,dimension_weight)
        self.bias = np.zeros((outgoing_neurons,1))

        self.state = np.zeros((outgoing_neurons,)) 
        self.activated = np.zeros((outgoing_neurons,))
        
        self.gradient_weight = np.zeros(dimension_weight)
        self.gradient_bias = np.zeros((outgoing_neurons,))
        self.gradient = np.zeros((outgoing_neurons,))
        
    def forward(self,previous_activation,activation):
        ''' Forward Pass Function '''
        
        self.state = self.weight @ previous_activation + self.bias
        self.activated = activation.forward(self.state)
    
    def backprop(self,previous_gradient,previous_weight,activation):
        ''' Gradient Calculation for Back Propogation '''
        
        self.gradient=activation.backprop(self.state) * (previous_weight.T @ previous_gradient)
    
    def update(self, activation_previous ,eta, ):
        ''' Updating the weight '''
        
        self.gradient_weight = self.gradient @ activation_previous.T
        self.weight -= eta * self.gradient_weight/minibatch
        self.bias -= eta * np.reshape(self.gradient.sum(axis = 1),(self.outgoing_neurons,1))/minibatch


''' Cross Entropy loss Calculation '''

def cross_entropy(target,predicted):
    temp=-target.T*np.log(predicted)
    loss=np.sum(temp)/len(target)
    return loss



if(len(sys.argv)<4):
    sys.exit('Insufficient arguments \nEnter 4 arguments: <Network Configuration>, <Epochs>, <Minibach>, <Learning Rate>')


''' Network Configuration '''

network_configuration = tuple(map(int, args.Network_Configuration.split(',')))
size_of_input = 784
number_of_classifications = 10
network = []


layer_one = Layer(size_of_input,network_configuration[0])
network.append(layer_one)
activation_one = Relu()
network.append(activation_one)
for i in range(1,len(network_configuration)):
    layer = Layer(network_configuration[i-1],network_configuration[i])
    network.append(layer)
    activation = Relu()
    network.append(activation)
layer_output = Layer(network_configuration[len(network_configuration)-1],number_of_classifications)
network.append(layer_output)




''' Loading Data '''

dataset_file = "mnist_traindata.hdf5"
data = h5py.File(dataset_file,'r')
ks_data = list(data.keys())
ks_data
x_data = data.get(ks_data[0]).value
y_data = data.get(ks_data[1]).value

x_data_train = x_data[:50000,:]
y_data_train = y_data[:50000,:]

x_data_val = x_data[50000:59999,:]
y_data_val = y_data[50000:59999,:]



''' Setting up the Hyperparameters '''
epochs = args.Epochs
minibatch = args.Minibatch
learning_rate = args.Learning_Rate




for i in range(epochs):
    for batch_counter in range(int(len(x_data_train)/minibatch)):
        
        ''' Making Minibatches of desired sizes'''
        
        x_data_batch = x_data_train[batch_counter * minibatch : batch_counter * minibatch + minibatch,:]
        y_data_batch = y_data_train[batch_counter * minibatch : batch_counter * minibatch + minibatch,:]
        
        
        '''Forward Propogation'''
        
        network[0].state =  network[0].weight @ x_data_batch.T + network[0].bias
        network[0].activated = network[1].forward(network[0].state)
        
        for layer_counter in range(2,len(network)-1,2):
            previous_activated = network[layer_counter-2].activated
            activation = network[layer_counter+1]
            network[layer_counter].forward(previous_activated,activation)
        
        network[-1].state = network[-1].weight @ network[-3].activated + np.reshape(network[-1].bias,(10,1))
        
        temp = network[-1].state - network[-1].state.max()
        temp2 = np.exp(temp)
        network[-1].activated = temp2 / np.sum(temp2, axis=0, keepdims=True)
        
        
        '''Backward Propogation'''
        
        network[-1].gradient = (network[-1].activated - y_data_batch.T)/len(x_data_batch)
        network[-1].gradient_weight = network[-1].gradient @ network[-3].activated.T
        network[-1].gradient_bias = network[-1].gradient.sum(axis = 0)
        
        for layer_counter in range(len(network)-3,-1,-2):
            activation=network[layer_counter+1]
            previous_gradient=network[layer_counter+2].gradient
            previous_weight=network[layer_counter+2].weight
            network[layer_counter].backprop(previous_gradient,previous_weight,activation)
        
        
        ''' Updating Weight '''
        
        network[0].gradient_weight = network[0].gradient @ x_data_batch
        network[0].weight -= learning_rate * network[0].gradient_weight
        network[0].gradient_bias = network[0].gradient.sum(axis = 1)
        network[0].bias -= learning_rate * np.reshape(network[0].gradient.sum(axis=1),(network[0].outgoing_neurons,1))/minibatch
        
        
        for layer_counter in range(2,len(network)-1,2):
            previous_activation = network[layer_counter-2].activated
            network[layer_counter].update(previous_activation,learning_rate)            
        
    
    ''' Validation Loss Calculation after every Epoch'''
    
    
    network[0].state =  network[0].weight @ x_data_val.T + network[0].bias
    network[0].activated = network[1].forward(network[0].state)
    for layer_counter in range(2,len(network)-1,2):
        previous_activated = network[layer_counter-2].activated
        activation = network[layer_counter+1]
        network[layer_counter].forward(previous_activated,activation)
        
        network[-1].state = network[-1].weight @ network[-3].activated + np.reshape(network[-1].bias,(10,1))
        
        temp = network[-1].state - network[-1].state.max()
        temp2 = np.exp(temp)
        network[-1].activated = temp2 / np.sum(temp2, axis=0, keepdims=True)
    
    val_acc=(np.argmax(y_data_val, axis = 1) == np.argmax(network[-1].activated, axis = 0)).sum()
    print('Epoch........',i,'  Validation Accuracy.............',val_acc/len(y_data_val))

        


        
        

