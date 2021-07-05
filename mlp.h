#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>

typedef double **dataset;
typedef int *classes;
typedef double activation_function(double);
typedef double error_function (dataset, dataset, int);
typedef double gradient_function (double, double);

typedef struct layer {
	int n_input;
	int n_output;
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
Layer create_layer (int, int, activation_function *, activation_function *);

Mlp create_mlp (int, Layer[], error_function *, gradient_function *);

dataset create_dataset (int, int, int, int);

classes create_classes (int, int, int);

void backpropagation (Mlp *, double, double *, int);

void train (Mlp *, int, double, dataset, dataset, int);

double *predict (Mlp *, double *);

double mse (classes, classes, int);

double crossentropy (dataset, dataset, int);

double crossentropy_grad (double, double);

double mse_grad (double, int, int);

double relu (double);

double relu_grad (double);

double sigmoid (double);

double sigmoid_grad (double);

double tanh (double);

double tanh_grad (double);

double **calculate_weights (double **);

/* Auxiliary Functions */
void print_dataset (char *, dataset, int, int);

void print_classes (char *, classes, int);
