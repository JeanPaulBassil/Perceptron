import numpy as np

class Perceptron:
    def __init__(self, inputSize, learningRate=0.1, epochs=100):
        self.weights = np.zeros(inputSize + 1)
        self.learningRate = learningRate
        self.epochs = epochs
        
    def predict(self, inputs):
        sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if sum > 0 else -1
    
    def train(self, trainingInput, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(trainingInput, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learningRate * (label - prediction) * inputs
                self.weights[0] += self.learningRate * (label - prediction)
                

                