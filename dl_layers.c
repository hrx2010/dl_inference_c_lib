#include "dl_layers.h"

int calc_dim_conv_layer(int original , int zero_padding , int size_filter , int stride){
	return (original + 2*zero_padding - size_filter) / stride + 1;
}

struct cls_tensor_activations convolution_2D(struct cls_tensor_activations input , struct cls_tensor_weights weights , struct cls_tensor_biases biases, int zero_padding , int stride){

    int output_height = calc_dim_conv_layer(input.height , zero_padding , weights.kernal_height , stride);
	int output_width = calc_dim_conv_layer(input.width , zero_padding , weights.kernal_width , stride);
    int output_depth = weights.num_filters;
    int size_output = output_height * output_width * output_depth;

	char *output = (char *)malloc(size_output * sizeof(char));
    memset(output , 0 , sizeof(char) * size_output);


    for(int k = 0 ; k < weights.num_filters; k ++){
		for(int st_row = -(zero_padding) ; st_row + weights.kernal_height <= input.height + zero_padding ; st_row += stride){
			for(int st_col = -(zero_padding) ; st_col + weights.kernal_width <= input.width + zero_padding ; st_col += stride){

                int rst_row = (st_row + zero_padding) / stride;
                int rst_col = (st_col + zero_padding) / stride;
                int id_rst = (rst_row * output_width + rst_col) * output_depth + k;

				for(int i = 0 ; i < weights.kernal_height ; i ++){
					for(int j = 0 ; j < weights.kernal_width ; j ++){
						int id_row = st_row + i;
						int id_col = st_col + j;
						if(id_row < 0 || id_row >= input.height)
							continue;
						if(id_col < 0 || id_col >= input.width)
							continue;

						int id_weights = k * weights.kernal_height * weights.kernal_width * weights.kernal_depth + (i * weights.kernal_width + j) * weights.kernal_depth;
						int id_input = (id_row * input.width + id_col) * input.depth;
						for(int t = 0 ; t < input.depth ; t ++)
							output[id_rst] += weights.filters[id_weights + t] * input.feature_map[id_input + t];
					}
				}
			}
		}
	}

	for(int i = 0 ; i < output_height ; i ++){
		for(int j = 0 ; j < output_width ; j ++){
			int id = (i * output_width + j) * output_depth;
			for(int k = 0 ; k < output_depth ; k ++)
				output[id + k] += biases.biases[k];
		}
	}

    struct cls_tensor_activations activations;
    activations.height = output_height;
    activations.width = output_width;
    activations.depth = output_depth;
    activations.feature_map = output;

    return activations;
}

struct cls_tensor_weights read_weights_from_file(char *filename , int kernal_height , int kernal_width , int kernal_depth , int num_filters){
    struct cls_tensor_weights weights;

    weights.kernal_height = kernal_height;
    weights.kernal_width = kernal_width;
    weights.kernal_depth = kernal_depth;
    weights.num_filters = num_filters;
    weights.filters = (char *)malloc(weights.kernal_height * weights.kernal_width * weights.kernal_depth * weights.num_filters * sizeof(char));

    FILE *file = fopen(filename , "rb");
    fread(weights.filters , sizeof(char) , weights.kernal_height * weights.kernal_width * weights.kernal_depth * weights.num_filters , file);
    fclose(file);

    return weights;
}

struct cls_tensor_biases read_biases_from_file(char *filename , int kernal_depth){
    struct cls_tensor_biases biases;

    biases.kernal_depth = kernal_depth;
    biases.biases = (char *)malloc(biases.kernal_depth * sizeof(char));

    FILE *file = fopen(filename , "rb");
    fread(biases.biases , sizeof(char) , biases.kernal_depth , file);
    fclose(file);

    return biases;
}

struct cls_tensor_activations read_activations_from_file(char *filename , int height , int width , int depth){
    struct cls_tensor_activations activations;

    activations.height = height;
    activations.width = width;
    activations.depth = depth;
    activations.feature_map = (char *)malloc(height * width * depth * sizeof(char));

    FILE *file = fopen(filename , "rb");
    fread(activations.feature_map , sizeof(char) , height * width * depth , file);
    fclose(file);

    return activations;
}

float compute_mean_square_error(struct cls_tensor_activations output , struct cls_tensor_activations ground_truth){
    float error = 0.0;

    int num_values = output.height * output.width * output.depth;

    for(int i = 0 ; i < num_values ; i ++){
        float diff = output.feature_map[i] - ground_truth.feature_map[i];
        error += diff * diff;
    }

    error /= num_values;

    return error;
}
