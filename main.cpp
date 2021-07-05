#include "mlp.h"
#include "parser.h"
#include <stdio.h>

int main () {
	/* Create datasets */
	int train_length = 417;
	int x_dimensions = 32;
	int categories = 2;
	int test_length = 154;

	dataset x_train = parse_x ("X_train.csv");
	dataset y_train = parse_y ("Y_train.csv");
	
	dataset x_test = parse_x ("X_test.csv");
	dataset y_test = parse_y ("Y_test.csv");


	//print_dataset ("X_train", x_train, x_dimensions, train_length);
	//print_dataset ("Y_train", y_train, categories, test_length);
	
	int n_layers = 4;

	Layer layer_1 = create_layer (x_dimensions, 16, sigmoid, sigmoid_grad);
	Layer layer_2 = create_layer (16, 8, sigmoid, sigmoid_grad);
	Layer layer_3 = create_layer (8, 6, sigmoid, sigmoid_grad);
	Layer layer_4 = create_layer (6, 4, sigmoid, sigmoid_grad);
	Layer layer_5 = create_layer (4, 2, sigmoid, sigmoid_grad);
	
	Layer layers [] = {layer_1, layer_2, layer_3, layer_4, layer_5};
	
	Mlp mlp_1 = create_mlp (5, layers, crossentropy, crossentropy_grad);
	
	double learning_rate = 0.00001;
	int epochs = 1000;
	
	train (&mlp_1, epochs, learning_rate, x_train, y_train, train_length);
	//test (&mlp_1, x_test, y_test, test_length);
	return 0;
}
