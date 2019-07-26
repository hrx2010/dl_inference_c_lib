#ifndef DL_LAYERS_H_INCLUDED
#define DL_LAYERS_H_INCLUDED

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


struct cls_tensor_weights_1D{
    int num_filters , num_rows , num_cols;
    char ***filters;
};

struct cls_tensor_activations_1D{
    int num_rows , num_cols;
    char **feature_map;
};


// load data from file

int compute_output_cols_convolution_1D(int num_input_col , int num_weight_col);

struct cls_tensor_activations_1D convolution_1D_no_padding(struct cls_tensor_activations_1D input , struct cls_tensor_weights_1D weights);

struct cls_tensor_activations_1D read_activations_from_source_code(char *data , int num_rows , int num_cols);

struct cls_tensor_weights_1D read_weights_1D_from_source_code(char *data , int num_filters , int num_rows , int num_cols);

// compute mean square error
float compute_mean_square_error(struct cls_tensor_activations output , struct cls_tensor_activations ground_truth);

#endif // DL_LAYERS_H_INCLUDED
