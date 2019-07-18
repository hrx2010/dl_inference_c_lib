#ifndef DL_LAYERS_H_INCLUDED
#define DL_LAYERS_H_INCLUDED

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// define structure of weights
struct cls_tensor_weights{
    int kernal_height, kernal_width, kernal_depth;
    int num_filters;
    char *filters;
};

// define structure of biases
struct cls_tensor_biases{
    int kernal_depth;
    char *biases;
};

// define strcucture of activations
struct cls_tensor_activations{
    int height, width, depth;
    char *feature_map;
};

struct cls_tensor_weights_1D{
    int num_filters , num_rows , num_cols;
    float ***filters;
};

struct cls_tensor_activations_1D{
    int num_rows , num_cols;
    float **feature_map;
};

// compute tensor size after convolutional layer
int calc_dim_conv_layer(int original , int zero_padding , int size_filter , int stride);

int compute_output_cols_convolution_1D(int num_input_col , int num_weight_col);

// load data from file
struct cls_tensor_activations convolution_2D(struct cls_tensor_activations input , struct cls_tensor_weights weights , struct cls_tensor_biases biases, int zero_padding , int stride);

struct cls_tensor_weights read_weights_from_file(char *filename , int kernal_height , int kernal_width , int kernal_depth , int num_filters);

struct cls_tensor_biases read_biases_from_file(char *filename , int kernal_depth);

struct cls_tensor_weights_1D read_weights_1D_from_file(char *filename , int num_filters , int num_rows , int num_cols);

struct cls_tensor_activations read_activations_from_file(char *filename , int height , int width , int depth);

struct cls_tensor_activations_1D read_activations_1D_from_file(char *filename , int num_rows , int num_cols);

struct cls_tensor_activations_1D convolution_1D_no_padding(struct cls_tensor_activations_1D input , struct cls_tensor_weights_1D weights);


// compute mean square error
float compute_mean_square_error(struct cls_tensor_activations output , struct cls_tensor_activations ground_truth);

#endif // DL_LAYERS_H_INCLUDED
