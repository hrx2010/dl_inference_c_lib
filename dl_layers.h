#ifndef DL_LAYERS_H_INCLUDED
#define DL_LAYERS_H_INCLUDED

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

typedef char VALUE_TYPE;

struct cls_tensor_weights_1D{
    int layer_index;
    int num_filters , num_rows , num_cols;
    VALUE_TYPE ***filters;
    int *bias;
    int shift;
    bool wReLU;
};

struct cls_tensor_activations_1D{
    int num_rows , num_cols;
    VALUE_TYPE **feature_map;
};

int compute_output_cols_convolution_1D(int num_input_col , int num_weight_col);

struct cls_tensor_activations_1D convolution_1D_no_padding(struct cls_tensor_activations_1D input , struct cls_tensor_weights_1D weights);

struct cls_tensor_activations_1D read_activations_from_source_code(VALUE_TYPE *data , int num_rows , int num_cols);

struct cls_tensor_activations_1D flatten_activations(struct cls_tensor_activations_1D input);

struct cls_tensor_weights_1D read_weights_1D_from_source_code(int layer_idx, VALUE_TYPE *data , int *bias, int num_filters , int num_rows , int num_cols, int shift, bool wReLU);

#endif // DL_LAYERS_H_INCLUDED
