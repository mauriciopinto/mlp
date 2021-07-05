#include "mlp.h"
#include "parser.h"
#include <stdio.h>

int main () {
	/* Create datasets */
	int train_length = 417;
	int x_dimensions = 31;
	int categories = 2;
	int test_length = 154;

	dataset x_train = parse_x ("X_train.csv");
	dataset y_train = parse_y ("Y_train.csv");
	
	dataset x_test = parse_x ("X_test.csv");
	dataset y_test = parse_y ("Y_test.csv");


	print_dataset ("X_train", x_train, x_dimensions, train_length);
	print_dataset ("Y_train", y_train, categories, test_length);
	
	int n_layers = 4;

	Layer layer_1 = create_layer (31, 4, sigmoid, sigmoid_grad);
	Layer layer_2 = create_layer (4, 6, sigmoid, sigmoid_grad);
	Layer layer_3 = create_layer (6, 8, sigmoid, sigmoid_grad);
	Layer layer_4 = create_layer (8, 2, sigmoid, sigmoid_grad);
	
	Layer layers [] = {layer_1, layer_2, layer_3, layer_4};
	
	Mlp mlp_1 = create_mlp (4, layers, crossentropy, crossentropy_grad);
	
	double learning_rate = 0.001;
	int epochs = 100;
	
	train (&mlp_1, epochs, learning_rate, x_train, y_train, train_length);
	double *probs = predict (&mlp_1, x_test[0]);
	return 0;
}
