import numpy as np


class Layer:
    def __init__(self, NInput, NOutput, bias=True):
        self.NInput = NInput
        self.NOutput = NOutput
        self.useBias = bias
        self.initializeWeights()

    def initializeWeights(self):
        """
        Initializes the weights and biases with small random values.
        """
        # create a weights matrix, of Input x Output size.
        self.weights = np.random.randn(self.NInput, self.NOutput)

        # if use Bias, then create another
        if self.useBias:
            self.bias = np.zeros((1, self.NOutput))
