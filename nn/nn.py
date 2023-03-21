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

        #initialize a dictionary of the activation and loss functions and their gradients
        # dictionary of activation functions without leading underscores
        self._activation_dict = {"sigmoid":self._sigmoid, "relu":self._relu}
        self._activation_backprop_dict = {"sigmoid":self._sigmoid_backprop, "relu":self._relu_backprop}
        self._loss_dict = {"bce":self._binary_cross_entropy, "mse":self._mean_squared_error}
        self._loss_backprop_dict = {"bce":self._binary_cross_entropy_backprop, "mse":self._mean_squared_error_backprop}


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

        #linear transform of the inputs
        Z_curr = np.transpose(np.matmul(W_curr, np.transpose(A_prev)) + np.tile(b_curr, A_prev.shape[0]))

        #feed transformed inputs through activation function
        A_curr = np.array([self._activation_dict[activation](i) for i in Z_curr])

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
        #this is needed only for single vector inputs
        if len(X.shape) == 1:
            X = np.reshape(X, (1, X.shape[0]))

        #The input to the next layer, starting with the function's argument.
        #at each step of the loop below, this is set to the single_layer_pass output
        last_output = X

        #linear and activation matrices for each layer
        #add the inputs at the first element since the final gradient is taken with respect to them
        cache = {"A0": X}

        for idx in range(len(self.arch)):
            layer = idx + 1
            #feed the output of the previous layer through the current layer
            single_layer_pass = self._single_forward(self._param_dict["W"+str(layer)], self._param_dict["b"+str(layer)], last_output, self.arch[idx]['activation'])
            #store the output of the current layer to be fed into the next layer
            last_output = single_layer_pass[0]
            #print(single_layer_pass[0].shape)
            #add entries to the dictionary; used in backpropagation
            cache["A" + str(layer)] = single_layer_pass[0]
            cache["Z" + str(layer)] = single_layer_pass[1]

        #print(f"forward pass output: {last_output.shape}")
        # import sys
        # sys.exit(0)
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

        #b_curr seems to be unused

        #derivative of the loss function L with respect to the transformed inputs Z
        #dimensions of len(z) (1d vector)
        #this equals the derivative of L with respect to the constants b (dL/db) since dZ/db = 1
        #'np.array' is used to convert from a list of 1-element arrays to one array

        # print("shapes")
        # print(dA_curr.shape)
        # print(Z_curr.shape)

        dLdZ = self._activation_backprop_dict[activation_curr](dA_curr, np.transpose(Z_curr))

        #print(dLdZ.shape)
        #print(W_curr.shape)

        #derivative of the loss function with respect to this neuron's input A_in (input as in during forward propagation)
        #dimensions of len(A_in) (1d vector)
        dLdA_in = np.matmul(np.transpose(W_curr), dLdZ)

        # print(dLdA_in.shape)

        #mathematically I believe this should use the transpose of dL/dZ, but it comes pre-transposed here due to quirks
        # of how my numpy matrices are oriented
        #dimensions of shape(W) (2d matrix)
        dLdW = np.matmul(dLdZ, A_prev)


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

        #store the gradients with respect to the weights in each layer for gradient descent
        grad_dict = {}

        #Initialized to the derivative of the loss with respect to the final prediction
        # this is the initial derivative from which backpropagation begins
        #This subsequently stores the derivative of the loss with respect to the input of each layer,
        # which is equal to the derivative of the loss with respect to the output of the layer above

        # print("ys")
        # print(y.shape)
        # print(y_hat.shape)

        dLdA_out = np.transpose(self._loss_backprop_dict[self._loss_func](y, y_hat))

        # print(dLdA_out)
        # print(dLdA_out.shape)

        for idx in range(nlayers):
            invidx = nlayers-idx-1

            backprop_s = self._single_backprop(
                W_curr = self._param_dict['W' + str(invidx+1)],
                b_curr = self._param_dict['b' + str(invidx+1)],
                Z_curr = cache["Z" + str(invidx+1)],
                A_prev = cache["A" + str(invidx)],
                dA_curr = dLdA_out,
                activation_curr = self.arch[invidx]["activation"])

            # print(backprop_s[1])
            # print(backprop_s[0])

            grad_dict["dW"+str(invidx+1)] = np.array(backprop_s[1])
            grad_dict["db"+str(invidx+1)] = np.array(backprop_s[2])

            dLdA_out = backprop_s[0]

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """

        #for each layer of the neural network, subtract the gradients of the loss function with respect to W and b
        # from W and b respectively to perform gradient descent
        for idx0 in range(len(self.arch)):
            idx = idx0+1
            self._param_dict["W"+str(idx)] -= grad_dict["dW"+str(idx)]*self._lr

            self._param_dict["b"+str(idx)] -= grad_dict["db"+str(idx)]*self._lr


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

        #training and validation losses for each epoch
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        #loop over all the epochs
        for epochx in range(self._epochs):

            print(f"epoch: {epochx}")

            # training and validation losses for each observation
            per_obs_loss_train = []
            per_obs_loss_val = []

            #grad dicts for each observation in a batch
            grad_dicts = []

            #loop over all training examples
            for i, xtr in enumerate(X_train):
                #print(f"x_train element: {xtr}")
                #predict from the current observation
                fpass = self.forward(xtr)
                #calculate the training loss
                #print(f"fpass[0].shape =  {fpass[0][0].shape}")

                train_i_2 = np.reshape(y_train[i], (1,len(y_train[i])))
                #print(train_i_2.shape)

                per_obs_loss_train.append(self._loss_dict[self._loss_func](train_i_2, fpass[0]))

                #calculate the gradient and update the weights
                grad_dicts.append(self.backprop(train_i_2, fpass[0], fpass[1]))

                #compute validation loss using the updated weights
                per_obs_loss_val.append(self._loss_dict[self._loss_func](y_val, self.predict(X_val)))
                #per_obs_loss_val += [self._loss_dict[self._loss_func](y_val[obs_i], self.predict(X_val[obs_i])) for obs_i in range(len(X_val))]

                #when the batch is done, update the parameters and reset the list of grad_dicts used to update them
                if i != 0 and i % self._batch_size == 0:
                    for grad_dict in grad_dicts:
                        self._update_params(grad_dict)
                    grad_dicts = []

                import sys
                #sys.exit(0)

            per_epoch_loss_train.append(np.nanmean(per_obs_loss_train))
            per_epoch_loss_val.append(np.nanmean(per_obs_loss_val))

        #print(per_epoch_loss_train)

        return (per_epoch_loss_train, per_epoch_loss_val)


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
        #the logistic function is applied elementwise to z
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
        #elementwise product of (the gradient of the logistic curve with respect to z) and
        # (the derivatives of the loss function with respect to the logistic curve values A)
        # [where A is the vector obtained by applying the logistic curve elementwise to z]
        # dL/dz = dL/dA o dA/dz             ['o' is the elementwise product]
        return [dA[i]*np.exp(-Z[i])*(1+np.exp(-Z[i]))**-2 for i in range(len(dA))]

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
        #print([zi if zi >= 0 else 0 for zi in Z])
        return [zi if zi >= 0 else 0 for zi in Z]

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
        #the index of 0 on dA is to get rid of useless internal arrays
        return np.reshape([dA[i][0] if Z[i] >= 0 else 0.0 for i in range(len(Z))], (len(Z),1))

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
        n_obs = y_hat.shape[0]
        n_feat = y_hat.shape[1]

        #print("bce-debug")
        #print(f"y shape: {y.shape}")
        #print(f"y_hat shape: {y_hat.shape}")

        # compute the binary cross entropy loss function. First term is added for true observations, second term is added for false ones.
        bce_loss = [-sum([y[i_obs][x] * np.log(y_hat[i_obs][x]) + (1 - y[i_obs][x]) * (np.log(1 - y_hat[i_obs][x]))
                          if (y_hat[i_obs][x] != 1 and y_hat[i_obs][x] != 0)
                          else np.sign(y_hat[i_obs][x] - y[i_obs][x]) * (10 ** 6)
                          for x in range(n_feat)]) / n_feat for i_obs in range(n_obs)]
        #print(f"bce_loss: {bce_loss}")
        #import sys
        #sys.exit(0)
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

        # print("bce-debug")
        # print(f"y shape: {y.shape}")
        # print(f"y_hat shape: {y_hat.shape}")

        #Note that this is the bce loss derivative with respect to y_hat,
        # not with respect to the weights W used to calculate y_hat.
        #Beware that these may be very large where y_hat is near 0 or 1.
        # As far as I can tell this is a natural property of the BCE function.
        return [[(y_hat[i_obs][i] - y[i_obs][i])/(y_hat[i_obs][i]*(1 - y_hat[i_obs][i])) if (y_hat[i_obs][i] != 1 and y_hat[i_obs][i] != 0)
                 else np.sign(y_hat[i_obs][i] - y[i_obs][i])*(10**6)
                 for i in range(y_hat.shape[1])] for i_obs in range(y_hat.shape[0])]

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

        return [np.sum([(y_hat[i_obs][i]-y[i_obs][i])**2
                for i in range(y_hat.shape[1])])/y_hat.shape[1]
                for i_obs in range(y_hat.shape[0])]

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

        return [[2*(y_hat[i_obs][i] - y[i_obs][i])
                for i in range(y_hat.shape[1])]
                for i_obs in range(y_hat.shape[0])]


#---------------------------------------------------------------------------
#--------------------------------  testing  --------------------------------
#---------------------------------------------------------------------------

import matplotlib.pyplot as plt

run_internal_nn_test = False

if not run_internal_nn_test:

    import sklearn.datasets
    digits = sklearn.datasets.load_digits()

    #initialize neural network
    test_nn = NeuralNetwork(
        [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'}, {'input_dim': 16, 'output_dim': 16, 'activation': 'relu'}, {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
        lr=.00005,
        seed=0,
        batch_size=1,
        epochs=10,
        loss_function="mse" #do not use bce here; it is not numerically stable
        )

    n_train = 200
    n_val = 50

    digit_data = digits.data
    np.random.shuffle(digit_data)

    train = digit_data[0:n_train]
    val = digit_data[n_train:n_train+n_val]

    # print(val[0])
    # print(np.reshape(val[0], (8,8)))

    fit = test_nn.fit(train, train, val, val)

    #validation loss
    plt.plot([i for i in range(len(fit[1]))], fit[1])
    plt.show()

    #training loss
    #plt.plot([i for i in range(len(fit[0]))], fit[0])
    #plt.show()

    val_mat = np.reshape(val[0], (8,8))

    plt.gray()
    plt.matshow(val_mat)
    plt.show()

    pred_mat = np.reshape(test_nn.predict(val[0]), (8,8))

    plt.matshow(pred_mat)
    plt.show()


if run_internal_nn_test:

    import warnings
    warnings.filterwarnings("ignore")

    #----------------------------------------------
    # generate data
    # generate a pair of 2d gaussians and label them and see if the method can distinguish the points

    #distributions for classes 0 and 1
    n0 = 550
    mean0 = [0,0]
    cov0 = [[1,0],[0,1]]
    n1 = 650
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
    n_train = 600
    x_train = data_labels[0:n_train,0:nf]
    y_train = np.reshape(data_labels[0:n_train,nf], (n_train, 1))
    x_val = data_labels[n_train:,0:nf]
    y_val = np.reshape(data_labels[n_train:,nf], (n0+n1-n_train, 1))

    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_val.shape)
    # print(y_val.shape)

    #----------------------------------------------
    #initialize neural network
    test_nn = NeuralNetwork(
        [{'input_dim': nf, 'output_dim': 2, 'activation': 'relu'}, {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=.1,
        seed=0,
        batch_size=10,
        epochs=10,
        loss_function="mse"
        )

    fit = test_nn.fit(x_train, y_train, x_val, y_val)

    #print(test_nn._param_dict["W1"])
    #print(test_nn._param_dict["b1"])
    #print(test_nn._param_dict["W2"])
    #print(test_nn._param_dict["b2"])

    #plot validation loss
    plt.plot([i for i in range(len(fit[1]))], fit[1])
    plt.show()

    plt.scatter(x_val[:,0], x_val[:,1], c=y_val)
    plt.show()

    val_pred = test_nn.predict(x_val)
    plt.scatter(x_val[:,0], x_val[:,1], c=val_pred)
    plt.show()

    plt.hist2d(y_val.flatten(), val_pred.flatten(), bins=(2,10))
    plt.show()


else:
    print("skipping internal nn test")
