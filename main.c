#include "mlp.h"
#include <stdio.h>

int main () {
	/* Create datasets */
	dataset x_ds = create_dataset (0, 10, 3, 100);
	classes y_ds = create_classes (0, 4, 100);

	//print_dataset ("X_DS", x_ds, 3, 100);
	//print_classes ("Y_DS", y_ds, 100);

	int n_layers = 4;

	Layer layer_1 = create_layer (10, 4, 4, relu, relu_grad);
	Layer layer_2 = create_layer (4, 6, 6, relu, relu_grad);
	Layer layer_3 = create_layer (6, 8, 8, relu, relu_grad);
	Layer layer_4 = create_layer (8, 4, 4, relu, relu_grad);

	double learning_rate = 0.001;
	int epochs = 20;

	Layer layers [] = {layer_1, layer_2, layer_3, layer_4};
	Mlp mlp_1 = create_mlp (4, layers, mse, mse_grad);

	return 0;
}
