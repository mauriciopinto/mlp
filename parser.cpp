#include "parser.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>
using namespace std;
/*
 * 1 = M
 * 0 = B
 */

double **parse_x (char *path){
    ifstream data(path);
    vector<vector<double>> datafame;

    if (data) {
        string line;

        while (getline(data, line)) {
            stringstream sep(line);
            string cell;
            datafame.emplace_back();
            while (getline(sep, cell, ',')) {
                datafame.back().push_back(stod(cell));
            }
        }
    }

    double **dataset = (double **) malloc (sizeof (double *) * datafame.size());
    for (int i = 0; i < datafame.size (); ++i)
        dataset[i] = (double *) malloc (sizeof (double *) * datafame[i].size());

    for (int i = 0; i < datafame.size (); ++i) {
        for (int j = 0; j < datafame[i].size (); ++j) {
            dataset[i][j] = datafame[i][j];
        }
        cout << endl;
    }

    return dataset;
}

double **parse_y (char *path) {
    ifstream data(path);
    vector<int> datafame;

    if (data) {
        string line;

        while (getline(data, line)) {
            datafame.push_back(stoi(line));
        }
    }

    double **dataset = (double **) malloc (sizeof (double *) * datafame.size());
    for (int i = 0; i < datafame.size (); ++i)
        dataset[i] = (double *) malloc (sizeof (double) * 2);

    for (int i = 0; i < datafame.size (); ++i) {
        if (datafame[i] == 0) {
            dataset[i][0] = 1;
            dataset[i][1] = 0;
        }
        else if (datafame[i] == 1) {
            dataset[i][0] = 0;
            dataset[i][1] = 1;
        }
    }

    return dataset;
}