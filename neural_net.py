import numpy as np
import random

#****************** Connection ************************
class Connection:
	def __init__(self):
		self.weight = random.random()
		self.deltaWeight = random.random()



#****************** class Neuron ***********************
class Neuron:

	eta = 0.15
	alpha = 0.5 #could be anything between zero and one

	def __init__(self, numOutputs, myIndex):
		self.m_myIndex = myIndex
		self.m_gradient = 0.0
		self.m_outputVal = 0.0

		self.m_outputWeights = []
		for c in range(0, numOutputs):
			self.m_outputWeights.append(Connection())

	def feedForward(self, prevLayer):
		sum = 0.0
		#sum the previous layer's outputs, include bias neuron
		for n in range(0, len(prevLayer)):
			sum += prevLayer[n].m_outputVal * prevLayer[n].m_outputWeights[self.m_myIndex].weight

		self.m_outputVal = self.transferFunction(sum);

	@staticmethod
	def transferFunction(x):
		#tanh(x) transfer function
		return np.tanh(x)

	@staticmethod
	def transferFunctionDerivative(x):
		return (1 - x*x) #actual derivative is 1 - tanh(x)^2

	def sumDOW(self, nextLayer):
		sum = 0.0
		for n in range(0, len(nextLayer)-1):
			sum += self.m_outputWeights[n].weight * nextLayer[n].m_gradient 
		return sum

	def calcOutputGradients(self, targetVal):
		delta = targetVal - self.m_outputVal
		self.m_gradient = delta * Neuron.transferFunctionDerivative(self.m_outputVal)

	def calcHiddenGradients(self, nextLayer):
		dow = self.sumDOW(nextLayer)
		self.m_gradient = dow * Neuron.transferFunctionDerivative(self.m_outputVal)

	def updateInputWeights(self, prevLayer):
		print "Previous layer length ", len(prevLayer)
		for n in range(0, len(prevLayer)):
			neuron = prevLayer[n]
			oldDeltaWeight = neuron.m_outputWeights[self.m_myIndex].deltaWeight
			newDeltaWeight = Neuron.eta * neuron.m_outputVal * self.m_gradient \
							+ Neuron.alpha * oldDeltaWeight

			neuron.m_outputWeights[self.m_myIndex].deltaWeight = newDeltaWeight
			neuron.m_outputWeights[self.m_myIndex].weight += newDeltaWeight
		return prevLayer

#******************** class Net *************************
class Net:
	def __init__(self, topology):
		#class members
		self.m_error = 0.0
		self.m_recentAverageError = 0.0
		numLayers = len(topology)
		#array of form [layer index][neuron index]
		self.m_layers = [];


		for layerNum in range(0, numLayers):

			if (layerNum == numLayers-1):
				numOutputs = 0  
			else: 
				numOutputs = topology[layerNum+1]

			layer = []
			for neuronNum in range(0, topology[layerNum]+1):
				layer.append(Neuron(numOutputs, neuronNum)) 
				print "Made a Neruon"

			self.m_layers.append(layer)
			self.m_layers[layerNum][-1].m_outputVal = 1.0

	def feedForward(self, inputVals):
		assert (len(inputVals) == len(self.m_layers[0])-1)

		for i in range(0, len(inputVals)):
			self.m_layers[0][i].m_outputVal = inputVals[i]

		for layerNum in range(1, len(self.m_layers)):
			prevLayer = self.m_layers[layerNum -1]
			for n in range(0, len(self.m_layers[layerNum])-1):
				self.m_layers[layerNum][n].feedForward(prevLayer)



	def backProp(self, targetVals):
		#calculate overall net error (rms)
		outputLayer = self.m_layers[-1] #handle to output layer
		self.m_error = 0.0

		for n in range(0, len(outputLayer)-1):
			delta = targetVals[n] - outputLayer[n].m_outputVal;
			self.m_error += delta*delta

		self.m_error /= len(outputLayer) -1
		self.m_error = np.sqrt(self.m_error)

		#implement a recent average measurment
		self.m_recentAverageError = (self.m_recentAverageError * self.m_recentAverageError + self.m_error)/(self.m_recentAverageError+1.0)

		#print self.m_layers[-1][0].m_gradient
		#calculate output layer gradient
		for n in range(0, len(outputLayer)-1):
			outputLayer[n].calcOutputGradients(targetVals[n])
		#print self.m_layers[-1][0].m_gradient

		#calculate gradients on hidden layer
		for layerNum in range (len(self.m_layers) - 2, 0, -1):
			print layerNum
			hiddenLayer = self.m_layers[layerNum]
			nextLayer = self.m_layers[layerNum + 1]

			for n in hiddenLayer:
				n.calcHiddenGradients(nextLayer)

		#for all layers from outputs to first hidden layer
		#update connection weights
		for layerNum in range(len(self.m_layers)-1, 0, -1):
			layer = self.m_layers[layerNum]
			prevLayer = self.m_layers[layerNum-1]
			for n in range(0, len(layer)-1):
				prevLayer = layer[n].updateInputWeights(prevLayer)


	def getResults(self):
		resultVals = []

		for n in range(0, len(self.m_layers[-1])-1):
			resultVals.append(self.m_layers[-1][n].m_outputVal)

		return resultVals


def main():
	topology = np.array([2, 4, 1])
	myNet = Net(topology)
	inputVals = np.array([1, 0])
	targetVals = np.array([1])
	trainingData = np.array([inputVals])
	trainingAns = np.array([targetVals])
	
	for i in range(0, 100):
		trainingData = np.append(trainingData, [inputVals], axis=0)
		trainingAns = np.append(trainingAns, [targetVals], axis=0)
	
	for i in range (0, trainingData.shape[0]):

		myNet.feedForward(trainingData[i])
		myNet.backProp(trainingAns[i])		
		resultVals = myNet.getResults()
		print "Results", resultVals
		print "Error: ", myNet.m_error


main()