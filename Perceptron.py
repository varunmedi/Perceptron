# class Perceptron(object):
#
# 	# Create a new Perceptron
# 	#
# 	# Params:	bias -	arbitrarily chosen value that affects the overall output
# 	#					regardless of the inputs
# 	#
# 	#			synaptic_weights -	list of initial synaptic weights for this Perceptron
# 	def __init__(self, bias, synaptic_weights):
#
# 		self.bias = bias
# 		self.synaptic_weights = synaptic_weights
#
#
# 	# Activation function
# 	#	Quantizes the induced local field
# 	#
# 	# Params:	z - the value of the indiced local field
# 	#
# 	# Returns:	an integer that corresponds to one of the two possible output values (usually 0 or 1)
# 	def activation_function(self, z):
#
#
#
# 	# Compute and return the weighted sum of all inputs (not including bias)
# 	#
# 	# Params:	inputs - a single input vector (which may contain multiple individual inputs)
# 	#
# 	# Returns:	a float value equal to the sum of each input multiplied by its
# 	#			corresponding synaptic weight
# 	def weighted_sum_inputs(self, inputs):
#
#
# 	# Compute the induced local field (the weighted sum of the inputs + the bias)
# 	#
# 	# Params:	inputs - a single input vector (which may contain multiple individual inputs)
# 	#
# 	# Returns:	the sum of the weighted inputs adjusted by the bias
# 	def induced_local_field(self, inputs):
#
#
# 	# Predict the output for the specified input vector
# 	#
# 	# Params:	input_vector - a vector or row containing a collection of individual inputs
# 	#
# 	# Returns:	an integer value representing the final output, which must be one of the two
# 	#			possible output values (usually 0 or 1)
# 	def predict(self, input_vector):
#
#
# 	# Train this Perceptron
# 	#
# 	# Params:	training_set - a collection of input vectors that represents a subset of the entire dataset
# 	#			learning_rate_parameter - 	the amount by which to adjust the synaptic weights following an
# 	#										incorrect prediction
# 	#			number_of_epochs -	the number of times the entire training set is processed by the perceptron
# 	#
# 	# Returns:	no return value
# 	def train(self, training_set, learning_rate_parameter, number_of_epochs):
#
#
# 	# Test this Perceptron
# 	# Params:	test_set - the set of input vectors to be used to test the perceptron after it has been trained
# 	#
# 	# Returns:	a collection or list containing the actual output (i.e., prediction) for each input vector
# 	def test(self, test_set):


import numpy as np


class Perceptron:
    def __init__(self, bias, num_inputs, alpha=0.9, lambda_reg=0.001):
        self.bias = bias
        self.synaptic_weights = np.random.uniform(-0.5, 0.5, num_inputs)  # Random weight initialization
        self.momentum = np.zeros(num_inputs)
        self.alpha = alpha  # Momentum coefficient
        self.lambda_reg = lambda_reg  # Reduced regularization

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def activation_function(self, z):
        return 1 if self.sigmoid(z) >= 0.5 else 0

    def weighted_sum_inputs(self, inputs):
        return np.dot(inputs, self.synaptic_weights) + self.bias

    def induced_local_field(self, inputs):
        return self.weighted_sum_inputs(inputs)

    def predict(self, input_vector):
        return self.activation_function(self.induced_local_field(input_vector))

    def train(self, training_set, learning_rate, epochs, batch_size=10):
        for epoch in range(epochs):
            np.random.shuffle(training_set)
            for i in range(0, len(training_set), batch_size):
                batch = training_set[i:i + batch_size]
                weight_update = np.zeros_like(self.synaptic_weights)
                bias_update = 0
                for inputs, target in batch:
                    prediction = self.predict(inputs)
                    error = target - prediction
                    weight_update += learning_rate * error * np.array(inputs)
                    bias_update += learning_rate * error

                # Apply momentum and reduced regularization
                self.momentum = self.alpha * self.momentum + weight_update / batch_size - self.lambda_reg * self.synaptic_weights
                self.synaptic_weights += self.momentum
                self.bias += bias_update / batch_size

            learning_rate *= 0.995  # Smoother learning rate decay

    def test(self, test_set):
        predictions = [self.predict(inputs) for inputs, _ in test_set]
        return predictions


def load_csv(filename):
    dataset = np.loadtxt(filename, delimiter=',', dtype=str)
    return dataset.tolist()


def preprocess_data(dataset):
    features = np.array([list(map(float, row[:-1])) for row in dataset])
    labels = np.array([0 if row[-1] == 'R' else 1 for row in dataset])

    # Standardization (Z-score normalization per column)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8  # Avoid division by zero
    features = (features - mean) / std

    return [(features[i], labels[i]) for i in range(len(features))]


def split_data(dataset, train_ratio=0.8):
    np.random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]
    return train_set, test_set


def evaluate(predictions, test_set):
    correct = sum(1 for pred, (_, actual) in zip(predictions, test_set) if pred == actual)
    accuracy = correct / len(test_set) * 100
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    dataset = load_csv(
        r'C:\Users\varun\Downloads\neural networks\Programming Project #1 - Implementing a Perceptron\sonar_all-data.csv')
    dataset = preprocess_data(dataset)
    train_set, test_set = split_data(dataset, train_ratio=0.8)

    perceptron = Perceptron(bias=0.01, num_inputs=60)
    perceptron.train(train_set, learning_rate=0.01, epochs=5000, batch_size=10)
    predictions = perceptron.test(test_set)
    evaluate(predictions, test_set)

