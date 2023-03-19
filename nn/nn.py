# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    #note that Union was originally called with parentheses and threw a syntax error
    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):  #<-- winter quarter sad face close parenthesis

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # dictionary of activation functions without leading underscores
        activation_dict = {"sigmoid":self._sigmoid}

        #print("matmul shapes:")
        #print(W_curr.shape)
        #print(A_prev.shape)
        #print(b_curr.shape)

        # print(W_curr)
        # print(A_prev)
        # print(np.matmul(W_curr, np.transpose(A_prev)))
        # print(np.tile(b_curr, A_prev.shape[0]))

        #linear transform of the inputs
        Z_curr = np.transpose(np.matmul(W_curr, np.transpose(A_prev)) + np.tile(b_curr, A_prev.shape[0]))

        #feed proto-outputs through activation function
        A_curr = np.array([activation_dict[activation](i) for i in Z_curr])

        return (A_curr, Z_curr)

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """

        #fix list formatting; adding matrices will not work otherwise
        if len(X.shape) == 1:
            X = np.reshape(X, (1, X.shape[0]))

        #The input to the next layer, starting with the function's argument.
        #at each step of the loop below, this is set to the single_layer_pass output
        last_output = X

        #linear and activation matrices for each layer
        cache = {"A0": X}

        for idx in range(int(len(self._param_dict)/2)):
            layer = idx + 1
            #feed the output of the previous layer through the current layer
            print(self.arch[idx])
            single_layer_pass = self._single_forward(self._param_dict["W"+str(layer)], self._param_dict["b"+str(layer)], last_output, self.arch[idx]['activation'])
            #store the output of the current layer to be fed into the next layer
            last_output = single_layer_pass[0]
            #add entries to the dictionary
            cache["A" + str(layer)] = single_layer_pass[0]
            cache["Z" + str(layer)] = single_layer_pass[1]

        return (last_output, cache)


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        activation_dict = {"sigmoid":self._sigmoid_backprop}

        ###dL/dA_prev = dL/dA_curr * dA_curr/dA_prev = (dL/dA_curr * dA_curr/dZ_curr) * dZ_curr/dA_prev

        #derivative of the loss function L with respect to the transformed inputs Z
        #dimensions of len(z)
        print("dldb")

        #print(dA_curr)
        #print(Z_curr)
        #print(b_curr)

        #print(len(dA_curr))

        #'np.array' is used to convert from a list of 1-element arrays to one array
        dLdZ = np.array(activation_dict[activation_curr](dA_curr, np.transpose(Z_curr)))
        #print(dLdZ)
        #print(np.array(dLdZ))
        #^ this equals the derivative of L with respect to the constants b (dL/db) since dZ/db = 1

        #print("dlda")

        #print(np.transpose(W_curr))
        #print(dLdZ)

        #derivative of the loss function with respect to the inputs A_in
        #dimensions of len(A_in)
        dLdA_in = np.matmul(np.transpose(W_curr), dLdZ)
        #print(dLdA_in)
        #print(dA_curr)

        print("dldw")

        #print(np.transpose(dLdZ))
        #print(A_prev)

        #mathematically I believe this should be the transpose of dL/dZ, but it comes pre-transposed here due to quirks
        # of how my numpy matrices are oriented
        dLdW = np.matmul(dLdZ, A_prev)
        #print(W_curr.shape)
        #print(dLdW.shape)

        return (dLdA_in, dLdW, dLdZ)


    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):

        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """

        nlayers = len(self.arch)

        grad_dict = {}

        #derivative of the loss with respect to the activation output
        dLdA_out = self._binary_cross_entropy_backprop(y, y_hat)

        for idx in range(nlayers):
            invidx = nlayers-idx-1

            backprop_s = self._single_backprop(
                W_curr = self._param_dict['W' + str(invidx+1)],
                b_curr = self._param_dict['b' + str(invidx+1)],
                Z_curr = cache["Z" + str(invidx+1)],
                #case to compare to inputs (or add a dummy entry to cache containing the inputs)
                A_prev = cache["A" + str(invidx)],
                dA_curr = dLdA_out,
                activation_curr = self.arch[invidx]["activation"])

            grad_dict["dW"+str(invidx)] = backprop_s[1]
            grad_dict["db"+str(invidx)] = backprop_s[2]

            dLdA_out = backprop_s[0]


        pass

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        per_obs_loss_train = []
        per_obs_loss_val = []

        for i, xtr in enumerate(X_train):
            print(i)
            #if i == 1:
            #    import sys
            #    sys.exit(0)

            #predict from the current observation
            fpass = self.forward(xtr)
            #calculate the training loss
            per_obs_loss_train.append(self._binary_cross_entropy(y_train[i], fpass[0]))

            #calculate the gradient and update the weights
            grad = self.backprop(y_train[i], fpass[0], fpass[1])
            self._update_params(grad)

            #compute validation loss
            per_obs_loss_val.append(self._binary_cross_entropy(y_val, self.predict(X_val)))



    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """

        #return the output of one forward pass through the network
        return self.forward(X)[0]

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        #using a logistic sigmoid function
        return [1/(1+np.exp(-i)) for i in Z]


    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        return [dA[i]*np.exp(-Z[i])/(1+np.exp(-Z[i]))**-2 for i in range(len(dA))]

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        pass

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        pass

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        n = y.shape[0]

        # compute the binary cross entropy loss function. First term is added for true observations, second term is added for false ones.
        #from hw7
        bce_loss = -sum([y_hat[x] * np.log(y_hat[x]) + (1 - y_hat[x]) * (np.log(1 - y_hat[x])) for x in range(n)]) / n
        return bce_loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to y_hat.
        """
        # dividing by y.shape[0] should make the learning rate choice less sensitive to batch size
        # np.matmul(y_hat - y, X)  # / y.shape[0]
        #from hw7
        return [(y_hat[i] - y[i])/(y_hat[i]*(1 - y_hat[i])) for i in range(len(y))]

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        pass

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        pass


#---------------------------------------------------------------------------
#--------------------------------  testing  --------------------------------
#---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#----------------------------------------------
# generate data
# generate a pair of 2d gaussians and label them and see if the method can distinguish the points

#distributions for classes 0 and 1
n0 = 500
mean0 = [0,0]
cov0 = [[1,0],[0,1]]
n1 = 600
mean1 = [2,3]
cov1 = [[1,1],[1,.5]]
#number of input features
nf = len(mean1)

#generate data and labels
#ignore the error about positive semidefiniteness
d0 = np.random.multivariate_normal(mean0, cov0, n0)
d1 = np.random.multivariate_normal(mean1, cov1, n1)

#combine the data points
data = np.concatenate((d0,d1))

#label each gaussian and combine the labels
labels = np.reshape(np.concatenate((np.zeros(n0),np.ones(n1))), (n0+n1,1))

#combine data points and their labels
data_labels = np.concatenate((data,labels), axis=1)

#shuffle combined data and labels before use
np.random.shuffle(data_labels)

#plot the data
plot = False
if plot:
    plt.scatter(data_labels[:,0], data_labels[:,1], c=data_labels[:,2])
    plt.show()

#prepare data and labels for testing
n_train = 700
x_train = data_labels[0:n_train,0:nf]
y_train = np.reshape(data_labels[0:n_train,nf], (n_train, 1))
x_val = data_labels[n_train:,0:nf]
y_val = np.reshape(data_labels[n_train:,nf], (n0+n1-n_train, 1))

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

#----------------------------------------------
#initialize neural network
test_nn = NeuralNetwork(
    [{'input_dim': nf, 'output_dim': 2, 'activation': 'sigmoid'}, {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
    lr=.1,
    seed=0,
    batch_size=10,
    epochs=10,
    loss_function="sigmoid"
    )

test_nn.fit(x_train, y_train, x_val, y_val)



#print(test_nn.predict(x_train[0]))
#print(y_train[0])
#print("bce")
#print(test_nn._binary_cross_entropy(y_train[0], test_nn.predict(x_train[0])))

#trimmings
# print(f"z curr shape: {Z_curr.shape}")
# print(f"w curr shape: {W_curr.shape}")
# print(f"a prev shape: {A_prev.shape}")
# print(f"b curr shape: {b_curr.shape}")
# print(np.matmul(W_curr, A_prev).shape)