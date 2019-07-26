#include "dl_layers.h"

struct cls_tensor_activations_1D convolution_1D_no_padding(struct cls_tensor_activations_1D input , struct cls_tensor_weights_1D weights){
    struct cls_tensor_activations_1D outputs;

    outputs.num_cols = compute_output_cols_convolution_1D(input.num_cols , weights.num_cols);
    outputs.num_rows = weights.num_filters;

    outputs.feature_map = (char **)malloc(outputs.num_rows * sizeof(char *));
    int **accumulated_data = (int **)malloc(outputs.num_rows * sizeof(int *));

    for(int i = 0 ; i < outputs.num_rows ; i ++){
        outputs.feature_map[i] = (char *)malloc(outputs.num_cols * sizeof(char));
        accumulated_data[i] = (int *)malloc(outputs.num_cols * sizeof(int));
    }

    for(int i = 0 ; i < weights.num_filters ; i ++){
        for(int j = 0 ; j + weights.num_cols <= input.num_cols ; j ++){
            //outputs.feature_map[i][j] = 0;
            accumulated_data[i][j] = 0;

            for(int k = 0 ; k < weights.num_rows ; k ++){
                for(int t = 0 ; t < weights.num_cols ; t ++){
                    accumulated_data[i][j] += input.feature_map[k][j + t] * weights.filters[i][k][t];
                }
            }

            if(accumulated_data[i][j] >= -127 && accumulated_data[i][j] <= 127){
                outputs.feature_map[i][j] = accumulated_data[i][j];
            }else if(accumulated_data[i][j] < -127){
                outputs.feature_map[i][j] = -127;
            }else{
                outputs.feature_map[i][j] = 127;
            }
        }
    }

    for(int i = 0 ; i < outputs.num_rows ; i ++)
        free(accumulated_data[i]);

    free(accumulated_data);

    return outputs;
}


int compute_output_cols_convolution_1D(int num_input_col , int num_weight_col){
    return num_input_col - num_weight_col + 1;
}

struct cls_tensor_activations_1D read_activations_from_source_code(char *data , int num_rows , int num_cols){
    struct cls_tensor_activations_1D output;

    output.num_rows = num_rows;
    output.num_cols = num_cols;

    output.feature_map = (char **)malloc(num_rows * sizeof(char *));

    for(int i = 0 ; i < num_rows ; i ++)
        output.feature_map[i] = (char *)(data + num_cols);

    return output;
}

struct cls_tensor_weights_1D read_weights_1D_from_source_code(char *data , int num_filters , int num_rows , int num_cols){
    struct cls_tensor_weights_1D weights;

    weights.num_filters = num_filters;
    weights.num_rows = num_rows;
    weights.num_cols = num_cols;

    weights.filters = (char ***)malloc(num_filters * sizeof(char **));

    for(int i = 0 ; i < num_filters ; i ++){
        weights.filters[i] = (char **)malloc(num_rows * sizeof(char *));

        for(int j = 0 ; j < num_rows ; j ++)
            weights.filters[i][j] = (data + i * num_rows * num_cols + j * num_cols);
    }

    return weights;
}
