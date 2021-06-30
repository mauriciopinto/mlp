#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double **dataset;
typedef int *classes;
typedef double activation_function(double);
typedef double error_function (classes, classes, int);
typedef double *gradient_function (double *, int);

typedef struct layer {
	int n_input;
	int n_output;
	int n_neurons;
	double *input;
	double *output;
	double **weights;
	double *sensitivities;
	activation_function *activate;
	activation_function *grad;
} Layer;


typedef struct mlp {
	int n_layers;
	Layer *layers;
	error_function *error;
	gradient_function *grad;
} Mlp;

/*Main functions*/
Layer create_layer (int, int, int, activation_function *, activation_function *);

Mlp create_mlp (int, Layer[], error_function *, gradient_function *);

dataset create_dataset (int, int, int, int);

classes create_classes (int, int, int);

void backpropagation (Mlp *, double);

void train (Mlp *, int, double, dataset, classes, int);

classes predict (Mlp *, dataset);

double mse (classes, classes, int);

double *mse_grad (double *, int);

double relu (double);

double relu_grad (double);

double **calculate_weights (double **);

/* Auxiliary Functions */
void print_dataset (char *, dataset, int, int);

void print_classes (char *, classes, int);
