#taskd by Rafaela

import numpy as np

np.random.seed(42)

initial_weights = np.random.rand(6)

import matplotlib.pyplot as plt



X = np.array([[0.04], 
             [0.20]])

Y = np.array([[0.50]]) 

alpha = 0.4 

epochs = 10000
tolerance = 1e-5


def sigmoid(x):
    return 1/(1+ np.exp(- x))

def sigmoid_derivative(x):
    return x * (1 - x)

def sse(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)

W_1 = np.array([[initial_weights[0], initial_weights[1]], 
                [initial_weights[2], initial_weights[3]]])
W_2 = np.array([[initial_weights[4],initial_weights[5]]])




def training(W_1, W_2, X, Y, epochs = 10000, tolerance = 1e-5, 
plot_loss=True):
     
    loss_history = []
    prev_loss = float('inf')
    
    
    for epoch in range(epochs):
    #forwards propagation
        z1 = np.dot(W_1, X)
        x1 = sigmoid(z1)
        z2 = np.dot(W_2, x1)
        y = sigmoid(z2)
        
        loss = sse(Y, y)
        loss_history.append(loss)
        
        if abs(prev_loss - loss) < tolerance:
            print(f"Stopped at epoch {epoch}, error change < {tolerance}")
            break
        prev_loss = loss

    
    
    #deltas
        delta2 = (y - Y) * sigmoid_derivative(y)
        delta1 = np.multiply(np.dot(np.transpose(W_2), delta2) , sigmoid_derivative(x1))
    
    #back propagation
        W_1 = W_1 - alpha * delta1 * np.transpose(X)
        W_2 = W_2 - alpha * delta2 * np.transpose(x1)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
            
    if plot_loss:
        plt.plot(loss_history, label="Training Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss (SSE)')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    
    return W_1, W_2, loss_history, y


W_1, W_2, losses, final_prediction = training(W_1, W_2, X, Y, plot_loss=True)
print("Final prediction:", final_prediction[0, 0])
    
    
    
   
  
    




