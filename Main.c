#include "dl_layers.h"
#include <stdio.h>

int main(){

    struct cls_tensor_activations_1D input = read_activations_1D_from_file("../testing_data/input_1d.dat" , 9 , 128);

    printf("read input data successfully!\n");

    struct cls_tensor_weights_1D weights = read_weights_1D_from_file("../testing_data/weight_1d.dat" , 51 , 9 , 50);

    printf("read weights data successfully!\n");

    struct cls_tensor_activations_1D output =convolution_1D_no_padding(input , weights);

    printf("perform 1D conv layer successfully!\n");

    return 0;
}

