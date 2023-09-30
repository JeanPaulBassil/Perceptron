from perceptron import Perceptron
import numpy as np

perceptron = Perceptron(inputSize=2)

trainingInputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

labels = np.array([-1, -1, -1, 1])

perceptron.train(trainingInput=trainingInputs, labels=labels)

print(perceptron.predict(np.array([0, 0])))
print(perceptron.predict(np.array([1, 1])))
