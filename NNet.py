# N.Z. - May 2017
# Let's try to write a Feedworward Neural Network with Backpropagation to identify MNIST numbers from scratch
# I am rewriting in Python an old University task written in Octave years ago
# https://en.wikipedia.org/wiki/MNIST_database
# http://yann.lecun.com/exdb/mnist/

# Surprise n.1: there are no datastructures, NNets are just matrices (or Tensors, if you want)
# (in fact, see Tensorflow implementation) 
# Surprise n.2: Such a schematic and minimalistic implementation can still have a word for a simple task
# such as the MNIST challenge

import numpy as np   
from scipy.io import loadmat  


data = loadmat('data/ex3data1.mat')  
data 

X = data['X'] #5000 x 400 
y = data['y'] #5000 x 1

#Now we need a one-hot encoder: a vector of length k where index n is set to 1 (and rest = 0)
# Let's use scikit-learn

from sklearn.preprocessing import OneHotEncoder  
encoder = OneHotEncoder(sparse=False)  
y_onehot = encoder.fit_transform(y)  #5000 x 10 (0...9 hot encoder)


# Functions to evaluate the loss
def sigmoid(z):  
    return 1 / (1 + np.exp(-z))


# Function to update the hypothesis for each state of the model:
# i.e. given a set of inputs, it calculates the output at each layer in the network
def forward_propagate(X, theta1, theta2):  
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    # h is the hypothesis vector
    # containing the prediction probabilities for each class
    # Its shape matches our one-hot encoding for y

    return a1, z2, a2, z3, h

# The Cost function runs the Forward-Propagation and calculates the error of the
# hypothesis (h, i.e. the predictions) against the actual values for the instance
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):  
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # Intialization
    J = 0
    # The deltas will be useful later
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    # The final result is an average sum of Logistic Regression
    J = J / m

    return J, delta1, delta2

######-----######

# Ok, let's test the simplest model
# Only with Forward Propagation

# Initial setup
input_size = 400  
hidden_size = 25  
num_labels = 10  
learning_rate = 1 #it is probably way too much

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

m = X.shape[0]  
X = np.matrix(X)  
y = np.matrix(y)

# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))  
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

# Print the  values
theta1.shape, theta2.shape

#((25L, 401L), (10L, 26L))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)  

# Print the  values
a1.shape, z2.shape, a2.shape, z3.shape, h.shape

#((5000L, 401L), (5000L, 25L), (5000L, 26L), (5000L, 10L), (5000L, 10L))

# Let's use the Cost function

# Cost Function computes the hypothesis matrix (h) to find the total error between h and y 
cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

# 6.8228086634127862 # Mmh... Not so good actually...


######-----######

# Let's move forward and add the Regularization part

# Let's re-write the Cost Function adding the Regularization

def cost_reg(params, input_size, hidden_size, num_labels, X, y, learning_rate):  
    
    J_r, delta1, delta2 = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)
    
    J_r += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))  

    return J_r, delta1, delta2


######-----######

# Let's add the Back Propagation

# First of all, we need to implement a Gradient Descent for the previous Sigmoid Function
def sigmoid_gradient(z):  
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate): 
    
    # First of all let's Forward-Propagate the data + current params and compare
    # the output with actual labels
    
    # Old Cost Function with Regularization
    J_r, delta1, delta2 = cost_reg(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
    
    # J_r is the total error across the whole data set
    
    # Backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J_r, grad




J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)  

J, grad.shape

# 6.8281541822949299, (10285L,)


######-------#######

# Let's train our network and predict the letters of the MNIST 

from scipy.optimize import minimize

# Minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),  
                method='TNC', jac=True, options={'maxiter': 250})
# options={'maxiter': 250} 
# Let's put a limit to the number of iterations: objective function is very unlikely
# going to converge...

# Observed output:
#    nfev: 250
#     fun: 0.33900736818312283 # Total cost is less than 0.5, so much better !

######-------#######

# Now let's use the found parameters and forward-propagate them through the network to get predictions 

X = np.matrix(X) 

# Numpy 'reshape' function helps us to reshape the output from the optimizer to match the parameter matrix 
# that our network is expecting
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))  
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))


# Let's generate an hypothesis for the MNIST data
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)  
y_pred = np.array(np.argmax(h, axis=1) + 1)  
y_pred 

# And in the end, let's calculate its accuracy
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]  
accuracy = (sum(map(int, correct)) / float(len(correct)))  
accuracy

# 0.9922  # 99.22% accuracy !!


