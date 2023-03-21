# TODO: import dependencies and write unit tests below
from nn import nn
from nn import preprocess
import numpy as np

# initialize neural network
test_nn = nn.NeuralNetwork(
    [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
     {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
    lr=.1,
    seed=0,
    batch_size=10,
    epochs=10,
    loss_function="mse"
)

def test_single_forward():
    pass

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    y_true = np.array([[0, 0], [.4, .5], [1, 1]])
    y_hat = np.array([[.1, .000001], [.4, .6], [.999, .999]])

    bce_ref = [0.052680757829163156, 0.6932849224146647, 0.0010005003335835344]

    bce = test_nn._binary_cross_entropy(y=y_true, y_hat=y_hat)

    print(bce)
    tolerance = 0.00000001

    assert np.array([bce[i] - bce_ref[i] < tolerance for i in range(len(bce_ref))]).all()

def test_binary_cross_entropy_backprop():
    y_true = np.array([[0, 0], [.4, .5], [1, 1]])
    y_hat = np.array([[.1, 0], [.4, .6], [1, 1.2]])

    backprop_ref = [[1.111111111111111, 0.0], [0.0, 0.4166666666666666], [0.0, -0.8333333333333334]]

    backprop = test_nn._binary_cross_entropy_backprop(y=y_true, y_hat=y_hat)

    tolerance = 0.00000001

    a = backprop
    b = backprop_ref

    bools = [a[j][i] - b[j][i] < tolerance for i in range(len(a[0])) for j in range(len(a))]

    assert np.array(bools).all()

def test_mean_squared_error():
    y_true = np.array([[0, 0], [.4, .5], [1, 1]])
    y_hat = np.array([[.1, 0], [.4, .6], [1, 1.2]])

    mse_ref = [0.005000000000000001, 0.0049999999999999975, 0.01999999999999999]

    mse = test_nn._mean_squared_error(y=y_true, y_hat=y_hat)

    tolerance = 0.00000001

    assert np.array([mse[i]-mse_ref[i] < tolerance for i in range(len(mse_ref))]).all()

def test_mean_squared_error_backprop():

    y_true = np.array([[0,0],[.4,.5],[1,1]])
    y_hat = np.array([[.1,0],[.4,.6],[1,1.2]])

    backprop_ref = [[0.2, 0.0], [0.0, 0.19999999999999996], [0.0, 0.3999999999999999]]

    backprop = test_nn._mean_squared_error_backprop(y=y_true, y_hat=y_hat)

    tolerance = 0.00000001

    a = backprop
    b = backprop_ref

    bools = [a[j][i]-b[j][i] < tolerance for i in range(len(a[0])) for j in range(len(a))]

    assert np.array(bools).all()

def test_sample_seqs():

    test_str = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
    test_labels = [True, False, True, False, True, False, False, False, True, False, True, False, False, False,
                   False, False]
    processed_seqs = preprocess.sample_seqs(test_str, test_labels)

    assert len(processed_seqs[0]) == len(test_str) and len(processed_seqs[1]) == len(test_labels) and sum(processed_seqs[1])/len(processed_seqs[1]) == 1/2


def test_one_hot_encode_seqs():
    onehot_test_in = ["cagtgtcatcgactgacgactgagcagcccgcgccgacgttta",
                      "cagtgtcatcgactggaagggagaaaattgagcagcccgcgccgacgttta",
                      "ttgagcagcccgcccccccgccgacg",
                      "atcgactgacgactgagcaggaagggagaaaattgagcagcccgcgcc"]

    onehot_test_ref = [[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1,
       0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
       0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
       0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
       0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
       0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
       0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]]

    equalities = [list(preprocess.one_hot_encode_seqs(onehot_test_in)[i]) == onehot_test_ref[i] for i in range(len(onehot_test_in))]
    assert equalities, "malfunctioning one-hot encoding"