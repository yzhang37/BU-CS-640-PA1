import numpy as np


class Network:
    def __init__(self, layers, activationList, dActivationList, loss, dLoss, is_one_hot=False):
        """
        :param layers: List[Layers]
               This should be a list of Layers objects.
        :param activationList: List[functions].
               a list of activation functions (e.g. sigmoid).
        :param dActivationList: List[functions].
               a list of derivative functions (e.g. dSigmoid).
        :param loss: function.
               a loss function (e.g. enclideanLoss).
        :param dLoss: function.
               a loss derivative function (e.g. dEuclideanLoss).
        """
        assert len(layers) == len(activationList) == len(dActivationList)
        self.layers = layers
        self.activationList = activationList
        self.dActivationList = dActivationList
        self.loss = loss
        self.dLoss = dLoss
        self.isOneHot = is_one_hot

        self.gradients = []
        self.grad_bias = []
        self.grad_zero()

    def grad_zero(self):
        """
        Make the gradients zero
        """
        self.gradients.clear()
        self.grad_bias.clear()
        for layer in self.layers[::-1]:
            self.gradients.append(np.zeros_like(layer.weights))
            if layer.useBias:
                self.grad_bias.append(np.zeros_like(layer.bias))
            else:
                self.grad_bias.append(None)

    def fit(self, X, Y, learningRate, regLambda, regMethod="l2"):
        """
        Fit the model with input features and targets.
        Parameters
        ----------
        X, Y : array-like
            X contains the input features while Y contains the target values.
        learningRate, regLambda : float
            Basic hyperparameters for the model.
        """

        if regMethod not in ("l1", "l2"):
            raise Exception("invalid reg method.")

        # First, initialize zero gradients.
        self.grad_zero()

        # Next, use the forward and backprog functions to accumulate gradients
        # sample by sample.
        # Hint:
        # For each sample, do:
        #     Forward pass once, using the forward function defined below.
        #     Backward propagation, using the backprog function defined below.

        for x, y in zip(X, Y):
            all_layers = self.forward(x)
            self.backprog(all_layers, y)

        # Lastly, update weights and biases using the gradients; don't forget to
        # take the mean before updating the weights. Note that both the learning
        # rate and the regularization should appear here.

        num_sample = X.shape[0]
        c = learningRate / num_sample
        for rev_i, layer in enumerate(self.layers[::-1]):
            if layer.useBias:
                layer.bias -= c * self.grad_bias[rev_i]
            if regMethod == "l1":
                layer.weights = layer.weights - c * regLambda * np.sign(layer.weights) - c * self.gradients[rev_i]
            elif regMethod == "l2":
                layer.weights = (1 - c * regLambda) * layer.weights - c * self.gradients[rev_i]

    def predict(self, X):
        """
        Predict the outcome of given input features. This should be as easy as
        Parameters
        ----------
        X : array-like
        """
        y_output = []
        for x in X:
            y = (self.forward(x)[-1]).flatten()
            if y.shape[0] == 1:
                y = y[0]
            y_output.append(y)
        ans = np.array(y_output)
        if self.isOneHot:
            pred_argmax = np.argmax(ans, axis=1)
            pred = np.zeros_like(ans)
            for i, argmax in enumerate(pred_argmax):
                pred[i, argmax] = 1
            ans = pred
        return ans

    def forward(self, x):
        """
        Performs the forward propagation for ONE sample.
        x  : array-like
            Note that this is just ONE single sample.
        """
        working = np.atleast_1d(x)[np.newaxis, :]
        all_layers = [working]
        for i, (layer, act_func) in enumerate(zip(self.layers, self.activationList)):
            working = act_func(np.dot(working, layer.weights) + layer.bias)
            all_layers.append(working)
        return all_layers

    def backprog(self, all_layers, y):
        """
        Performs the backward propagation. This is the core part of the model.
        Parameters
        ----------
        None, but feel free to add anything you like.
        """
        yPred = all_layers[-1]
        # First, compute the derivative of the loss and the output activation, as beta
        beta = self.dLoss(yPred, y)

        rev_hidden = all_layers[::-1]
        rev_dActLists = self.dActivationList[::-1]
        # Then, compute the gradient layer by layer in the reverse order.
        for rev_i, layer_i in enumerate(self.layers[::-1]):
            # dAct_layer_i is, the dActivation of the current layer_(i+1)
            beta = rev_dActLists[rev_i](rev_hidden[rev_i], beta)

            if layer_i.useBias:
                self.grad_bias[rev_i] += beta

            new_gradient = np.dot(rev_hidden[rev_i + 1].T, beta)
            self.gradients[rev_i] += new_gradient
            beta = np.dot(beta, layer_i.weights.T)

    def getLoss(self, X, Y, regLambda=0.0, regMethod="l2"):
        """
        Compute and return the loss given X and Y. This function is helpful for
        you to visualize the training process.
        """
        if regMethod not in ("l1", "l2"):
            raise Exception("invalid reg method.")
        y_pred = self.predict(X)
        num_samples = X.shape[0]
        _loss = self.loss(y_pred, Y) / num_samples
        _reg = 0.0
        if regLambda != 0:
            if regMethod == "l1":
                for layer in self.layers:
                    _reg += np.sum(np.abs(layer.weights))
            elif regMethod == "l2":
                for layer in self.layers:
                    _reg += np.sum(np.square(layer.weights))
                _reg /= 2.0
        return _loss + regLambda * _reg / num_samples
