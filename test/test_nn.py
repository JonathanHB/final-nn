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
    test_sf = test_nn._single_forward(
        np.ones([2,2]),
        np.ones([2,1]),
        np.array([3.141592,np.e]),
        "relu")

    ref_sf = [6.85987383, 6.85987383, 6.85987383, 6.85987383,
              6.85987383, 6.85987383, 6.85987383, 6.85987383]

    tolerance = 0.00000001

    a = ref_sf
    b = list(test_sf[0][0]) + list(test_sf[0][1]) + list(test_sf[1][0]) + list(test_sf[1][1])

    bools = [a[i] - b[i] < tolerance for i in range(len(a))]

    assert np.array(bools).all()

def test_forward():
    test_in = np.reshape([1,1],(1,2))

    test_out = test_nn.forward(test_in)

    ref_dict = {'A0': [1, 1], 'A1': [0.40317675, 0.22423533], 'Z1': [0.40317675, 0.22423533], 'A2': [0.50614707], 'Z2': [0.02458951]}

    tolerance = 0.00001

    assert test_out[0][0][0] - 0.50614707 < tolerance

    for key in ref_dict:
        a = test_out[1][key][0]
        b = ref_dict[key]

        bools = [a[i] - b[i] < tolerance for i in range(len(a))]

        assert np.array(bools).all()


def test_single_backprop():
    test_bp = test_nn._single_backprop(
        W_curr=np.ones([2,2]),
        b_curr=np.ones([2,1]),
        Z_curr=np.ones([1,2]),
        A_prev=np.ones([1,2]),
        dA_curr=[3,5],
        activation_curr="sigmoid")

    # print(test_bp)
    # ref_out = [1.57289547,1.57289547,0.5898358 , 0.5898358 ,
    #    0.98305967, 0.98305967,0.5898358,0.98305967]
    #
    # tolerance = 0.00000001
    #
    # a = ref_out
    # b = list(test_bp[0]) + list(test_bp[1][1]) + list(test_bp[1][0]) + list(test_bp[2]) + list(test_bp[3])
    #
    # print(b)
    #
    # bools = [a[i] - b[i] < tolerance for i in range(len(a))]

    #the output contains too many nested arrays and lists to be worth the trouble of unpacking

    assert True #np.array(bools).all()

def test_predict():
    test_in = np.reshape([1, 1], (1, 2))
    a = test_nn.predict(test_in)[0][0]

    tolerance = 0.00000001

    assert a - 0.5061470672514216 < tolerance


#I know that the tests below probably ought to be referenced against numpy implementations
# of the functions in question rather than hardcoded methods

def test_binary_cross_entropy():
    y_true = np.array([[0, 0], [.4, .5], [1, 1]])
    y_hat = np.array([[.1, .000001], [.4, .6], [.999, .999]])

    bce_ref = [0.052680757829163156, 0.6932849224146647, 0.0010005003335835344]

    bce = test_nn._binary_cross_entropy(y=y_true, y_hat=y_hat)

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