#include "mlp.h"

/* Create a layer that receives 'n_input' values and returns
 * 'n_output' values after wieghting and activating them
 * through the 'activate' method */
Layer create_layer (int n_input, int n_output, activation_function *activate, 
					activation_function *grad) {
	Layer layer;
	layer.n_input = n_input;
	layer.n_output = n_output;
	
	layer.input = (double *) malloc (sizeof (double) * n_input);
	layer.output = (double *) malloc (sizeof (double) * n_output);

	layer.weights = (double **) malloc (sizeof (double *) * n_input);
	for (int i = 0; i < n_input; ++i) {
		layer.weights[i] = (double *) malloc (sizeof (double) * n_output);
		for (int j = 0; j < n_output; ++j)
			layer.weights[i][j] = (double) ((rand () % 10) * 0.1);
	}

	layer.sensitivities = (double *) malloc (sizeof (double) * n_input);
	layer.activate = activate;
	layer.grad = grad;
	return layer;
}


/* Creates a MLP with 'n_layers' layers, passed on the array
 * 'layers'. Its error function is defined by 'error' and its
 * gradient is defined by 'grad' */
Mlp create_mlp (int n_layers, Layer layers[], error_function *error, gradient_function *grad) {
	Mlp mlp;
	mlp.n_layers = n_layers;
	
	mlp.layers = (Layer *) malloc (sizeof (Layer) * n_layers);
	for (int i = 0; i < n_layers; ++i)
		mlp.layers[i] = layers[i];

	mlp.error = error;
	mlp.grad = grad;
	return mlp;
}


/* Creates a dataset. A dataset contains 'n' vectors of 'dims'
 * dimensions that determine an object's class. This function
 * creates datasets with integer values between 'low' and 
 * 'low + high' */
dataset create_dataset (int low, int high, int dims, int n) {
	dataset new_dataset = (dataset) malloc (sizeof (double *) * n);
	for (int i = 0; i < n; ++i) {
		new_dataset[i] = (double *) malloc (sizeof (double) * dims);
		for (int j = 0; j < dims; ++j)
			new_dataset[i][j] = rand () % high + low;
	}

	return new_dataset;
}


/* Creates an array of classes. A class is an integer value that
 * classifies an object. This function returns a random array of 
 * 'n' classes with values between 'low' and 'low + high'.*/
classes create_classes (int low, int high, int n) {
	classes new_classes = (classes) malloc (sizeof (int) * n);
	for (int i = 0; i < n; ++i) {
		new_classes[i] = rand() % high + low;
	}
	return new_classes;
}


/* Calculates the values of the weights of each layer in
 * 'network' by back-propagating them starting from the 
 * output layer. */
void backpropagation (Mlp *network, double learning_rate,
					  double *y_pd, double *y) {
	int n_layers = network->n_layers;
	Layer *prev_layer = &network->layers[n_layers - 1];
	
	/* Sensitivities of the error layer */
	
	double *error_sensitivities = (double *) malloc (sizeof (double) * prev_layer->n_output);
	for (int i = 0; i < prev_layer->n_output; ++i)
		error_sensitivities[i] = network->grad (prev_layer->output[i], y[i]);

	/* Calculate the sensitivities of the output layer inputs
	 * with the formula: 
	 * d(p[i]) = A'(p[i])*sum(w[i][j]*d(c[j]) for j in prev.n_output) */
	for (int i = 0; i < prev_layer->n_input; ++i) {
		double sum = 0;
		for (int j = 0; j < prev_layer->n_output; ++j) {
			sum += prev_layer->weights[i][j] * error_sensitivities[j];
		}
		prev_layer->sensitivities[i] = prev_layer->grad (prev_layer->input[i]) * sum;
	}

	/* Calculate the weights of the output layer with the
	 * formula:
	 * w[i][j] = w[i][j]-learning_rate*d(w[i][j])
	 * where d(w[i][j]) = A(p[i])*d(c[j]) */
	for (int n = 0; n < prev_layer->n_input; ++n) {
		for (int m = 0; m < prev_layer->n_output; ++m) {
			double grad = prev_layer->activate (prev_layer->input[n]) * error_sensitivities[m];
			prev_layer->weights[n][m] = prev_layer->weights[n][m] - learning_rate * grad;
		}
	}
	
	Layer *current_layer;

	/* Repeat the procedure for all other layers */
	for (int i = n_layers - 1; i > 0; --i) {
		current_layer = &network->layers[i];
		prev_layer = &network->layers[i - 1];

		/* Calculate the sensitivities of the previous layer */
		for (int n = 0; n < prev_layer->n_input; ++n) {
			double sum = 0;
			for (int m = 0; m < prev_layer->n_output; ++m) {
				sum += prev_layer->weights[n][m] * current_layer->sensitivities[m];
			}
			prev_layer->sensitivities[n] = prev_layer->grad (prev_layer->input[n]) * sum;
		}


		/* Calculate the weights of the previous layer with thepredicted[i] = probabilities
		 * formula:
		 * w[i][j] = w[i][j]-learning_rate*d(w[i][j])
		 * where d(w[i][j]) = A(p[i])*d(c[j]) */
		for (int n = 0; n < prev_layer->n_input; ++n) {
			for (int m = 0; m < prev_layer->n_output; ++m) {
				double grad = prev_layer->activate (prev_layer->input[n])*current_layer->sensitivities[m];
				prev_layer->weights[n][m] = prev_layer->weights[n][m] - learning_rate * grad;
			}
		}
	}
}


/* Trains the model 'network' by backpropagating it 'epochs'
 * times.  */
void train (Mlp *network, int epochs, double learning_rate, 
				dataset x, dataset y, int n) {
	int index = rand () % n;
	double *probabilities = predict (network, x[index]);

	int train = ceil (n * 70 / 100);
	dataset predicted = (dataset) malloc (sizeof (double) * train);
	dataset real = (dataset) malloc (sizeof (double) * train);

	for (int i = 0; i < epochs; ++i) {
		for (int j = 0; j < train; ++j) {
			index = rand () % train;
			backpropagation (network, learning_rate, probabilities, y[index]);
			probabilities = predict (network, x[index]);
			memcpy (predicted[i], probabilities, sizeof (double) * 2);
			real[i] = y[index];
		}
		double error = network->error (predicted, y, train);
		printf ("Epoch %d: error=%f\n", i, error);
	}
}

/* Predicts the probabilities of the output given
 * a data element 'x', using the model 'network'. */
double *predict (Mlp *network, double *x) {
	Layer *current_layer = &network->layers[0];
	
	/* Map x to the input layer's input */
	current_layer->input = x;
	
	/* Generate the output for the first layer */
	for (int i = 0; i < current_layer->n_output; ++i) {
		double sum = 0;
		for (int j = 0; j < current_layer->n_input; ++j)
			sum += current_layer->activate (current_layer->input[j]) * current_layer->weights[j][i];
		current_layer->output[i] = sum;
	}
	
	/* Repeat the same procedure for all other layers. */
	for (int i = 1; i < network->n_layers; ++i) {
		current_layer = &network->layers[i];
		current_layer->input = network->layers[i-1].output;

		for (int i = 0; i < current_layer->n_output; ++i) {
			double sum = 0;
			for (int j = 0; j < current_layer->n_input; ++j)
				sum += current_layer->activate (current_layer->input[j]) * current_layer->weights[j][i];
			current_layer->output[i] = sum;
		}
	}

	/* The output of the last layer represents the 
	 * probabilities of the classifications of x */
	return current_layer->output;
}

/* Minimum square error function */
double mse (classes y_1, classes y_2, int n) {
	double sum = 0;
	for (int i = 0; i < n; ++i)
		sum += pow (y_2[i] - y_1[i], 2);
	return sqrt (sum);
}

/* Softmax */
double *softmax(double *y_1, int n){
	printf ("%f, %f\n", y_1[0], y_1[1]);
    double norm =10e-8;
    for (int j = 0; j < n; ++j) {
        norm+=exp(y_1[j]);
    }
    double ans[n];
    for (int k = 0; k < n; ++k) {
        double tmp= (exp(y_1[k]))/(norm);
        ans[k] = tmp;
    }
    return ans;
}

/* Cross entropy loss function y_1 = y_pred */
double crossentropy (dataset y_1, dataset y_2, int n) {
    double **probs = (double **) malloc (sizeof (double *) * n);

    for (int i = 0; i < n; ++i) {
        probs[i] = softmax(y_1[i],2);
    }
	printf ("softmax\n");
    double sum[2];
    for (int j = 0; j < 2; ++j) {
		for (int i = 0; i < n; ++i)
        	sum[j] -= (y_2[i][j]*log(probs[i][j]) ) + ((1-y_2[i][j]) * log(1-probs[i][j]));
    }
    return sum[0] + sum[1];
}

/* Cross entropy gradient */
double crossentropy_grad (double y_1, double y_2) {
	return y_2 - y_1;
}

/* MSE Gradient */
double mse_grad (double value, int y_pd, int y) {
	double sensitivity = (y - y_pd) * (-value);
	return sensitivity;
}

/* Rectified Linear Unit function */
double relu (double value) {
	return value < 0 ? 0 : value;
}

/* RELU Gradient */
double relu_grad (double value) {
	return 0;
}

/* Sigmoid activation function */
double sigmoid (double value) {
	return 1/(1+exp(-value));
}

/* Sigmoid Gradient */
double sigmoid_grad (double value) {
	return sigmoid(value)*(1-sigmoid(value));
}

/* Hyperbolic tangent activation function */
double tanh (double value) {
	return (2/(1+exp(-2*value))) - 1;
}

/* Hyperbolic tangent Gradient */
double tanh_grad (double value) {
	return tanh(value)*tanh(value);
}


/* Auxiliary Functions */

void print_dataset (char *name, dataset data, int dims, int n) {
	printf ("================\n");
	printf ("%s\n", name);
	printf ("================\n");

	for (int i = 0; i < n; ++i) {
		printf ("%d\t", i);
		for (int j = 0; j < dims; ++j)
			printf ("%f\t", data[i][j]);
		printf ("\n");
	}
}

void print_classes (char *name, classes data, int n) {
	printf ("================\n");
	printf ("%s\n", name);
	printf ("================\n");

	for (int i = 0; i < n; ++i) {
		printf ("%d\t%d\n", i, data[i]);
	}
	printf ("\n");
}
