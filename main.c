#include "mlp.h"
#include <stdio.h>

int main () {
	/* Create datasets */
	int dataset_length = 100;
	dataset x_ds = create_dataset (0, 10, 3, dataset_length);
	classes y_ds = create_classes (0, 4, dataset_length);

	print_dataset ("X_DS", x_ds, 3, dataset_length);
	print_classes ("Y_DS", y_ds, dataset_length);

	int n_layers = 4;

	Layer layer_1 = create_layer (10, 4, sigmoid, sigmoid_grad);
	Layer layer_2 = create_layer (4, 6, sigmoid, sigmoid_grad);
	Layer layer_3 = create_layer (6, 8, sigmoid, sigmoid_grad);
	Layer layer_4 = create_layer (8, 4, sigmoid, sigmoid_grad);

	double learning_rate = 0.001;
	int epochs = 20;

	Layer layers [] = {layer_1, layer_2, layer_3, layer_4};
	Mlp mlp_1 = create_mlp (4, layers, mse, mse_grad);
	train (&mlp_1, epochs, learning_rate, x_ds, y_ds, dataset_length);
	double *probs = predict (&mlp_1, x_ds[0]);
	
	return 0;
}
