#include "dl_layers.h"

struct cls_tensor_activations_1D convolution_1D_no_padding(struct cls_tensor_activations_1D input , struct cls_tensor_weights_1D weights){
    struct cls_tensor_activations_1D outputs;

    outputs.num_cols = compute_output_cols_convolution_1D(input.num_cols , weights.num_cols);
    outputs.num_rows = weights.num_filters;

    outputs.feature_map = (VALUE_TYPE **)malloc(outputs.num_rows * sizeof(VALUE_TYPE *));
    for(int i = 0 ; i < outputs.num_rows ; i ++){
        outputs.feature_map[i] = (VALUE_TYPE *)malloc(outputs.num_cols * sizeof(VALUE_TYPE));
    }

    int accumulated_data = 0;
    for(int i = 0 ; i < weights.num_filters ; i ++){
        for(int j = 0 ; j + weights.num_cols <= input.num_cols ; j ++){

            for(int k = 0 ; k < weights.num_rows ; k ++){
                for(int t = 0 ; t < weights.num_cols ; t ++){
                    accumulated_data += input.feature_map[k][j + t] * weights.filters[i][k][t];
                }
            }
            accumulated_data += weights.bias[i];
            accumulated_data = accumulated_data >> weights.shift;
//            accumulated_data = round(accumulated_data * pow(2, weights.shift));
//            accumulated_data = accumulated_data * pow(2, weights.shift);

            if(accumulated_data > 127){
                accumulated_data = 127;
            }

            if (weights.wReLU) {
                if(accumulated_data <= 0){
                    accumulated_data = 0;
                }
            }

            outputs.feature_map[i][j] = accumulated_data;
            accumulated_data = 0;
        }
    }

    //debug
    char fo[1000];
    sprintf(fo, "layer_%d_output.dat", weights.layer_index);
    FILE *of = fopen(fo, "w");
    for(int j = 0 ; j < outputs.num_cols ; j ++)
        for(int i = 0 ; i < outputs.num_rows ; i ++)
            fprintf(of, "%d,\n", outputs.feature_map[i][j]);
    fclose(of);

    return outputs;
}


int compute_output_cols_convolution_1D(int num_input_col , int num_weight_col){
    return num_input_col - num_weight_col + 1;
}

struct cls_tensor_activations_1D read_activations_from_source_code(VALUE_TYPE *data , int num_rows , int num_cols){
    struct cls_tensor_activations_1D output;

    output.num_rows = num_rows;
    output.num_cols = num_cols;

    output.feature_map = (VALUE_TYPE **)malloc(num_rows * sizeof(VALUE_TYPE *));

    for(int i = 0 ; i < num_rows ; i ++) {
        output.feature_map[i] = (VALUE_TYPE *)malloc(num_cols * sizeof(VALUE_TYPE));
        for(int j = 0 ; j < num_cols ; j ++){
            output.feature_map[i][j] = data[j * num_rows + i];
        }
    }

    return output;
}

void release_tensor_activations_1D(struct cls_tensor_activations_1D input) {
    if (input.feature_map != NULL) {
        for(int i = 0 ; i < input.num_rows ; i ++)
            free(input.feature_map[i]);
        free(input.feature_map);
    }
}


struct cls_tensor_activations_1D flatten_activations(struct cls_tensor_activations_1D input){
    struct cls_tensor_activations_1D output;

    output.num_rows = input.num_cols * input.num_rows;
    output.num_cols = 1;

    output.feature_map = (VALUE_TYPE **)malloc(output.num_rows * sizeof(VALUE_TYPE *));

    for(int i = 0 ; i < output.num_rows ; i ++) {
        output.feature_map[i] = (VALUE_TYPE *)malloc(output.num_cols * sizeof(VALUE_TYPE));
        for(int j = 0 ; j < output.num_cols ; j ++){
            //printf("%d, %d\n", i%input.num_rows, i/input.num_rows);
            output.feature_map[i][j] = input.feature_map[i%input.num_rows][i/input.num_rows];
        }
    }

    return output;
}

struct cls_tensor_weights_1D read_weights_1D_from_source_code(int layer_idx, VALUE_TYPE *data , int *bias , int num_filters , int num_rows , int num_cols, int shift, bool wReLU){
    struct cls_tensor_weights_1D weights;

    weights.layer_index = layer_idx;
    weights.num_filters = num_filters;
    weights.num_rows = num_rows;
    weights.num_cols = num_cols;
    weights.shift = shift;
    weights.wReLU = wReLU;

    weights.filters = (VALUE_TYPE ***)malloc(num_filters * sizeof(VALUE_TYPE **));
    weights.bias = (int *)malloc(num_filters * sizeof(int));

    for(int i = 0 ; i < num_filters ; i ++){
        weights.bias[i] = bias[i];

        weights.filters[i] = (VALUE_TYPE **)malloc(num_rows * sizeof(VALUE_TYPE *));
        for(int j = 0 ; j < num_rows ; j ++) {
            weights.filters[i][j] = (VALUE_TYPE *)malloc(num_cols * sizeof(VALUE_TYPE));
            for(int k = 0 ; k < num_cols ; k ++){
                weights.filters[i][j][k] = data[i * num_cols * num_rows  + k * num_rows + j];
            }
        }

    }

    return weights;
}

void release_tensor_weights_1D(struct cls_tensor_weights_1D weights) {
    if (weights.bias != NULL)
        free(weights.bias);

    if (weights.filters != NULL) {
        for(int i = 0 ; i < weights.num_filters ; i ++){
            for(int j = 0 ; j < weights.num_rows ; j ++)
                free(weights.filters[i][j]);
            free(weights.filters[i]);
        }
        free(weights.filters);
    }
}

