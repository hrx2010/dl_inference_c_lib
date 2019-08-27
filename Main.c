#include "dl_layers.h"
#include "dl_data.h"
#include "sample_data.h"

int main(){

//    FILE *of = fopen("sample_data.h", "w");
//    fprintf(of, "#ifndef SAMPLE_DATA_H_INCLUDED\n");
//    fprintf(of, "#define SAMPLE_DATA_H_INCLUDED\n\n");
//    fprintf(of, "VALUE_TYPE inputs_data[2947][1152] = {\n");
//    for (int sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) {
//        fprintf(of, "{");
//
//        char filename[1000];
//        sprintf(filename, "../data_int/%06d.bin", sample_idx + 1);
//        printf("Read sample: %s\n", filename);
//        VALUE_TYPE* inputs =read_values_from_file(filename, NUM_FEATURES);
//
//        for(int j = 0 ; j < NUM_FEATURES - 1 ; j ++)
//            fprintf(of, "%d, ", inputs[j]);
//        fprintf(of, "%d", inputs[NUM_FEATURES - 1]);
//
//        if (sample_idx == NUM_SAMPLES-1)
//            fprintf(of, "}\n");
//        else
//            fprintf(of, "},\n");
//
//    }
//    fprintf(of, "};\n");
//    fprintf(of, "#endif\n");
//    fclose(of);

    for (int sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) {
    //    printf("prepare input data!\n");
//        char filename[1000];
//        sprintf(filename, "../data_int/%06d.bin", sample_idx + 1);
        printf("Read sample - %d\n", sample_idx);

//        VALUE_TYPE* inputs =read_values_from_file(filename, NUM_FEATURES);
        struct cls_tensor_activations_1D input = read_activations_from_source_code(inputs_data[sample_idx] , 9 , 128);

    //    printf("CONV 1 (1/4) - prepare weights data!\n");
        struct cls_tensor_weights_1D layer1_weights = read_weights_1D_from_source_code(1 , weights_1st_layer , weights_bias_1st_layer , 32 , 9 , 64 , 10 , true);
    //    printf("CONV 1 (1/4) - compute 1D conv!\n");
        struct cls_tensor_activations_1D layer1_output = convolution_1D_no_padding(input , layer1_weights);
        release_tensor_weights_1D(layer1_weights);
        release_tensor_activations_1D(input);
//        free(inputs);

    //    printf("CONV 2 (2/4) - prepare weights data!\n");
        struct cls_tensor_weights_1D layer2_weights = read_weights_1D_from_source_code(2 , weights_2nd_layer , weights_bias_2nd_layer , 64 , 32 , 32 , 10 , true);
    //    printf("CONV 2 (2/4) - compute 1D conv!\n");
        struct cls_tensor_activations_1D layer2_output = convolution_1D_no_padding(layer1_output , layer2_weights);
        release_tensor_weights_1D(layer2_weights);
        release_tensor_activations_1D(layer1_output);

    //    printf("CONV 3 (3/4) - prepare weights data!\n");
        struct cls_tensor_weights_1D layer3_weights = read_weights_1D_from_source_code(3 , weights_3rd_layer , weights_bias_3rd_layer , 128 , 64 , 16 , 10 , true);
    //    printf("CONV 3 (3/4) - compute 1D conv!\n");
        struct cls_tensor_activations_1D layer3_output = convolution_1D_no_padding(layer2_output , layer3_weights);
        release_tensor_weights_1D(layer3_weights);
        release_tensor_activations_1D(layer2_output);

    //    printf("FC 1 (4/4) - flatten activations data!\n");
        struct cls_tensor_activations_1D layer3_output_flatten = flatten_activations(layer3_output);
    //    printf("FC 1 (4/4) - prepare weights data!\n");
        struct cls_tensor_weights_1D layer4_weights = read_weights_1D_from_source_code(4 , weights_4th_layer , weights_bias_4th_layer , 6 , 2432 , 1 , 13 , false);
    //    printf("FC 1 (4/4) - compute FC!\n");
        struct cls_tensor_activations_1D layer4_output = convolution_1D_no_padding(layer3_output_flatten , layer4_weights);

        int act_idx = arg_max(layer4_output, NUM_CLASSES);
        printf("Predict activity: %s\n", ACT_STR[act_idx]);

        release_tensor_weights_1D(layer4_weights);
        release_tensor_activations_1D(layer3_output);
        release_tensor_activations_1D(layer3_output_flatten);
        release_tensor_activations_1D(layer4_output);

    }

    return 0;
}

