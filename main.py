import numpy as np
import pandas as pd

def save_model(network, filename="model_weights.npz"):
    model_data = {}
    for i, layer in enumerate(network.layers):
        model_data[f"weights_{i}"] = layer.weights
        model_data[f"biases_{i}"] = layer.biases
    np.savez(filename, **model_data)
    print(f"Model saved to {filename}")

def load_model(network, filename="model_weights.npz"):
    loaded_data = np.load(filename)
    for i, layer in enumerate(network.layers):
        layer.weights = loaded_data[f"weights_{i}"]
        layer.biases = loaded_data[f"biases_{i}"]
    print(f"Model loaded from {filename}")

def ReLU(x):
    return np.maximum(0, x)

def ReLUDeriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class Layer:
    def __init__(self, input_size, output_size, activation="relu"):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None
        self.activation = activation

    def forward(self, X):
        self.input = X
        z = np.dot(X, self.weights) + self.biases
        if self.activation == "relu":
            self.output = ReLU(z)
        elif self.activation == "softmax":
            self.output = softmax(z)
        return self.output
    
    def backward(self, d_output, learning_rate):
        if self.activation == "softmax":  
            dZ = d_output  
        else:
            dZ = d_output * ReLUDeriv(self.output)

        dW = np.dot(self.input.T, dZ) / self.input.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / self.input.shape[0]
        d_input = np.dot(dZ, self.weights.T)

        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db

        return d_input
    
class Network:
    def __init__(self, layer_size, learning_rate):
        self.layers = [Layer(layer_size[i], layer_size[i+1], activation="relu") for i in range(len(layer_size) - 2)]
        self.layers.append(Layer(layer_size[-2], layer_size[-1], activation="softmax"))  
        self.learning_rate = learning_rate

    def forward_propagation(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward_propagation(self, X, Y):
        output = self.forward_propagation(X)

        lossDeriv = output - Y

        for i in reversed(range(len(self.layers))):
            lossDeriv = self.layers[i].backward(lossDeriv, self.learning_rate)

def accuracy(predictions, labels):
    predicted_classes = np.argmax(predictions, axis=1)  
    true_classes = np.argmax(labels, axis=1)  
    return np.mean(predicted_classes == true_classes) * 100  


layer_sizes = [784, 128, 10]


data = pd.read_csv("./Neural Network/mnist.csv")
data = np.array(data)

np.random.shuffle(data)  

split_index = 1000  
Y_dev = data[:split_index, 0]  
X_dev = data[:split_index, 1:] / 255.  

Y_train = data[split_index:, 0]  
X_train = data[split_index:, 1:] / 255.  

Y_train_one_hot = np.zeros((Y_train.size, 10))
Y_train_one_hot[np.arange(Y_train.size), Y_train.astype(int)] = 1

network = Network(layer_sizes, learning_rate=0.01)
load_model(network)

epochs = 1000

X_sample = X_train
Y_sample = Y_train_one_hot  

Y_dev_one_hot = np.zeros((Y_dev.size, 10))
Y_dev_one_hot[np.arange(Y_dev.size), Y_dev.astype(int)] = 1

for epoch in range(epochs):
    output = network.forward_propagation(X_sample)
    network.backward_propagation(X_sample, Y_sample)

    if (epoch + 1) % 100 == 0:
        acc = accuracy(output, Y_sample)
        print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {acc:.2f}%")

final_output = network.forward_propagation(X_dev)
final_accuracy = accuracy(final_output, Y_dev_one_hot)

print("\nFinal Accuracy: {:.2f}%".format(final_accuracy))

save_model(network)
