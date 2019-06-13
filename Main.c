#include "dl_layers.h"
#include <stdio.h>

int main(){

    // load input from file
    struct cls_tensor_activations input = read_activations_from_file("./testing_data/input.bin" , 224 , 224 , 3);
    // load weights from file
    struct cls_tensor_weights weights = read_weights_from_file("./testing_data/weights.bin" , 3 , 3 , 3 , 8);
    // load bias from file
    struct cls_tensor_biases biases = read_biases_from_file("./testing_data/biases.bin" , 8);
    // compute 2D convolution results
    struct cls_tensor_activations output = convolution_2D(input , weights , biases , 1 , 1);
    // load ground truth from file
    struct cls_tensor_activations output_ground_truth = read_activations_from_file("./testing_data/output.bin" , output.height , output.width , output.depth);
    // test convolution 2D results
    printf("mean square error is %f\n" , compute_mean_square_error(output , output_ground_truth));

    return 0;
}

