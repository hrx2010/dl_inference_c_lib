#ifndef DL_LAYERS_H_INCLUDED
#define DL_LAYERS_H_INCLUDED

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// define structure of weights
struct cls_tensor_weights{
    int kernal_height, kernal_width, kernal_depth;
    int num_filters;
    float *filters;
};

// define structure of biases
struct cls_tensor_biases{
    int kernal_depth;
    float *biases;
};

// define strcucture of activations
struct cls_tensor_activations{
    int height, width, depth;
    float *feature_map;
};

// compute tensor size after convolutional layer
int calc_dim_conv_layer(int original , int zero_padding , int size_filter , int stride);

// load data from file
struct cls_tensor_activations convolution_2D(struct cls_tensor_activations input , struct cls_tensor_weights weights , struct cls_tensor_biases biases, int zero_padding , int stride);

struct cls_tensor_weights read_weights_from_file(char *filename , int kernal_height , int kernal_width , int kernal_depth , int num_filters);

struct cls_tensor_biases read_biases_from_file(char *filename , int kernal_depth);

struct cls_tensor_activations read_activations_from_file(char *filename , int height , int width , int depth);

// compute mean square error
float compute_mean_square_error(struct cls_tensor_activations output , struct cls_tensor_activations ground_truth);

#endif // DL_LAYERS_H_INCLUDED
