#include "dl_layers.h"
#include "dl_data.h"

int main(){

    printf("prepare input data!\n");
    struct cls_tensor_activations_1D input = read_activations_from_source_code(inputs_data , 9 , 128);

    printf("CONV 1 (1/4) - prepare weights data!\n");
    struct cls_tensor_weights_1D layer1_weights = read_weights_1D_from_source_code(1 , weights_1st_layer , weights_bias_1st_layer , 32 , 9 , 64 , 10 , true);
    printf("CONV 1 (1/4) - compute 1D conv!\n");
    struct cls_tensor_activations_1D layer1_output = convolution_1D_no_padding(input , layer1_weights);
    release_tensor_weights_1D(layer1_weights);
    release_tensor_activations_1D(input);

    printf("CONV 2 (2/4) - prepare weights data!\n");
    struct cls_tensor_weights_1D layer2_weights = read_weights_1D_from_source_code(2 , weights_2nd_layer , weights_bias_2nd_layer , 64 , 32 , 32 , 10 , true);
    printf("CONV 2 (2/4) - compute 1D conv!\n");
    struct cls_tensor_activations_1D layer2_output = convolution_1D_no_padding(layer1_output , layer2_weights);
    release_tensor_weights_1D(layer2_weights);
    release_tensor_activations_1D(layer1_output);

    printf("CONV 3 (3/4) - prepare weights data!\n");
    struct cls_tensor_weights_1D layer3_weights = read_weights_1D_from_source_code(3 , weights_3rd_layer , weights_bias_3rd_layer , 128 , 64 , 16 , 10 , true);
    printf("CONV 3 (3/4) - compute 1D conv!\n");
    struct cls_tensor_activations_1D layer3_output = convolution_1D_no_padding(layer2_output , layer3_weights);
    release_tensor_weights_1D(layer3_weights);
    release_tensor_activations_1D(layer2_output);

    printf("FC 1 (4/4) - flatten activations data!\n");
    struct cls_tensor_activations_1D layer3_output_flatten = flatten_activations(layer3_output);
    printf("FC 1 (4/4) - prepare weights data!\n");
    struct cls_tensor_weights_1D layer4_weights = read_weights_1D_from_source_code(4 , weights_4th_layer , weights_bias_4th_layer , 6 , 2432 , 1 , 13 , false);
    printf("FC 1 (4/4) - compute FC!\n");
    struct cls_tensor_activations_1D layer4_output = convolution_1D_no_padding(layer3_output_flatten , layer4_weights);
    release_tensor_weights_1D(layer4_weights);
    release_tensor_activations_1D(layer3_output);
    release_tensor_activations_1D(layer3_output_flatten);
    release_tensor_activations_1D(layer4_output);

    printf("Completed!\n");

    return 0;
}

