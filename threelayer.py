# alphas = [0.001,0.01,0.1,1,10,100,1000]

# We set the mean of these weights around 0
# These are the initial weights for your two layers
first_weights = 2*np.random.random((5,4)) - 1
second_weights = 2*np.random.random((4,3)) - 1
third_weights = 2*np.random.random((3,1)) - 1

# Lists to track the error rate
error_list = []
graph_errors = []

# Activation Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivitive of Activation Function with Respect to Error
def backpropogation(x):
    return x*(1-x)

def secret_formula(lst):
    return ((lst[0]*lst[3]+lst[4])/lst[2] > 1.3) or (sum(lst) < 2)

# Take 100,000 batches of 100
for batch in xrange(500000):
    temp = [[random.random() for _ in range(5)] for _ in range(100)]
    
    X = np.array(temp)
    y = np.array( [[secret_formula(row).conjugate()] for row in X] )

    
    # layer_0 is your input matrix
    layer_0 = X
    # layer_1 is your first hidden layer
    layer_1 = sigmoid(np.dot(layer_0, first_weights))
    
    # layer_2 is your second hidden layer
    layer_2 = sigmoid(np.dot(layer_1, second_weights))

    # layer_3 is your output layer
    layer_3 = sigmoid(np.dot(layer_2, third_weights))

    # how much did we miss the target value?
    output_error = y - layer_3

    # Take product of slope of sigmoid and output error
    layer_3_delta = output_error*backpropogation(layer_3)

    # Find error of Layer 2 and get derivitive of layer 2 with respect to error
    layer_2_error = layer_3_delta.dot(third_weights.T)
    layer_2_delta = layer_2_error * backpropogation(layer_2)

    # same process for layer 1
    layer_1_error = layer_2_delta.dot(second_weights.T)
    layer_1_delta = layer_1_error * backpropogation(layer_1)

    # Use an alpha level of .01 to decrease rate of descent
    # This helps to avoid local minimums!
    third_weights += layer_2.T.dot(layer_3_delta)*.01
    second_weights += layer_1.T.dot(layer_2_delta)*.01
    first_weights += layer_0.T.dot(layer_1_delta)*.01

    # Calculate Current Error
    error = np.mean(np.abs(output_error))
    # Log this for mean error calculation
    error_list.append(error)

    if (batch % 10000) == 0 and batch > 0:
        mean_error = (sum(error_list)*1.)/10000
        graph_errors.append(mean_error)
        error_list = []

        print "Error at iteration %s:\t \t %s" %(batch, mean_error)
