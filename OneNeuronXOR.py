import numpy as np
import random
import matplotlib.pyplot as plt

numIterations = 100 # Number of Iterations used
eta=1 # Multiplying Factor
theta = 0.01 # Error Limit

inputs = [(0,0),(0,1),(1,0),(1,1)] # Inputs
outputs = [(0),(1),(1),(0)] # Target Outputs

losses = np.zeros((numIterations, 1)) # Array to contain loss values

num_ip_neurons = 1  # No. of Input Neurons
num_ips_ip_layer = 2  # No. of Inputs to Input Neurons
num_op_neurons = 0  # No. of Output Neurons
num_ips_op_layer = 0  # No. of Inputs to each Neuron

def sigmoid(x):  # Sigmoid function
    return 1 / (1 + np.exp(-x))

def derivSigmoid(x):  # Derivative of Sigmoid Function
    return (sigmoid(x) * (1 - sigmoid(x))) #sigmoid*(1-sigmoid)

##################################################################################################################################
##                                              Neuron Class                                                                    ##
##################################################################################################################################
class neuron:
    def __init__(self,numInputs):
        # np.random.seed(1)
        self.weights = np.random.rand(numInputs)
        self.weights = np.array([(self.weights)])
        self.bias = random.random()
        self.name = None
        self.polyVar1 = np.random.random()
        self.polyVar2 = np.random.random()

##################################################################################################################################
##                                              Neural Network Class                                                            ##
##################################################################################################################################
class myANN:
    def __init__(self, num_ip_neurons, num_op_neurons):
        self.num_ip_neurons = num_ip_neurons
        self.num_op_neurons = num_op_neurons
        self.ip_neurons = []
        self.op_neurons = []
        # self.w3 = random.random()
        # self.w4 = random.random()

    def addIPNeurons(self,  num_ip):
        tempNeuron = neuron(num_ip)
        self.ip_neurons.append(tempNeuron)
        return tempNeuron

    def addOPNeurons(self,  num_ip):
        tempNeuron = neuron(num_ip)
        self.op_neurons.append(tempNeuron)
        return tempNeuron

    def polyFunction(self):
        self.w3 = random.random()
        self.w4 = random.random()
        return self.w3,self.w4

##################################################################################################################################
##                                  Creation of Neural Network and Neurons                                                      ##
##################################################################################################################################


myNetwork = myANN(num_ip_neurons,num_op_neurons)
myNeurons = ['inNeuron1', 'inNeuron2', 'opNeuron']
j = 0
for temp in range(myNetwork.num_ip_neurons):
    tempNeuron = myNetwork.addIPNeurons(num_ips_ip_layer)
    tempNeuron.name = myNeurons[j]
    j = j + 1
for temp in range(myNetwork.num_op_neurons):
    tempNeuron = myNetwork.addOPNeurons(num_ips_op_layer)
    tempNeuron.name = myNeurons[j]
    j = j + 1
print("Initial weights and bias for neurons are as:")
for i in range(myNetwork.num_ip_neurons):
    print(f"{myNetwork.ip_neurons[i].name} -- weights : {myNetwork.ip_neurons[i].weights} -- bias: "
          f"{myNetwork.ip_neurons[i].bias}")
for i in range(myNetwork.num_op_neurons):
    print(f"{myNetwork.op_neurons[i].name} -- weights : {myNetwork.op_neurons[i].weights} -- bias: "
          f"{myNetwork.op_neurons[i].bias}")
print()
x , y = myNetwork.polyFunction()
print(myNetwork.ip_neurons[0].polyVar1,myNetwork.ip_neurons[0].polyVar2)



##################################################################################################################################
##                                              Training Algorithm                                                              ##
##################################################################################################################################


def train(inputs, outputs, numIterations):
    for itr in range(numIterations):
        for i in range(len(inputs)):
            ip = inputs[i]
            y = outputs[i]
            w1 = myNetwork.ip_neurons[0].weights[0][0]
            w2 = myNetwork.ip_neurons[0].weights[0][1]
            w3 = myNetwork.ip_neurons[0].polyVar1
            w4 = myNetwork.ip_neurons[0].polyVar2
            b = myNetwork.ip_neurons[0].bias
            p = (ip[0]*w1 + ip[1]*w2) + b
            polFunc = p*(p*w3 + w4)

            a = sigmoid(polFunc)
            loss = (a - y)**2
            term1 = 2*(a-y)
            term2 = derivSigmoid(polFunc)
            term = term1 * term2
            delta_w4 = term * p * eta
            delta_w3 = term * (p**2) * eta
            delta_b = term * (2*p*w3 + w4) * eta
            delta_w1 = term * (2*p*w3 + w4) * ip[0]*eta
            delta_w2 = term * (2 * p * w3 + w4) * ip[1] * eta

##################################################################################################################################
##                                              Updating Parameters                                                             ##
##################################################################################################################################
            myNetwork.ip_neurons[0].polyVar1 -= delta_w3
            myNetwork.ip_neurons[0].polyVar2 -= delta_w4
            myNetwork.ip_neurons[0].bias -= delta_b
            myNetwork.ip_neurons[0].weights[0][0] -= delta_w1
            myNetwork.ip_neurons[0].weights[0][1] -= delta_w2

##################################################################################################################################
##                                                    Training                                                                  ##
##################################################################################################################################

train(inputs,outputs,numIterations)

##################################################################################################################################
##                                                   Prediction                                                                 ##
##################################################################################################################################

def predict(ip,w3,w4):
    w1 = myNetwork.ip_neurons[0].weights[0][0]
    w2 = myNetwork.ip_neurons[0].weights[0][1]
    b = myNetwork.ip_neurons[0].bias
    p = (ip[0] * w1 + ip[1] * w2) + b
    polFunc = p * (p * w3 + w4)

    a = sigmoid(polFunc)
    print (a)
    if a > 0.9:
        return 1
    else:
        return 0

print()
print('Predicted XOR truth table\n')
print(f" X | Y  |  Out")
print(f" 0 | 0  |  {predict((0,0),myNetwork.ip_neurons[0].polyVar1,myNetwork.ip_neurons[0].polyVar2)}")
print(f" 0 | 1  |  {predict((0,1),myNetwork.ip_neurons[0].polyVar1,myNetwork.ip_neurons[0].polyVar2)}")
print(f" 1 | 0  |  {predict((1,0),myNetwork.ip_neurons[0].polyVar1,myNetwork.ip_neurons[0].polyVar2)}")
print(f" 1 | 1  |  {predict((1,1),myNetwork.ip_neurons[0].polyVar1,myNetwork.ip_neurons[0].polyVar2)}")
print()