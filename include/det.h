# pragma once
# include "postprocess_op.h"
# include "utility.h"

#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"
#include "rknn_api.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

using namespace cv;


class TextDetect{
public:
    TextDetect(){};
    int Model_Init(const char* rk_model_name);
    int Model_Infer(cv::Mat& Input_Image, vector<vector<vector<int>>> &boxes, double &times);
    ~TextDetect();

private:
    //config
    rknn_context ctx;
    int ret;
    rknn_tensor_attr input_attrs[1];
    rknn_tensor_attr output_attrs[1];
    rknn_input inputs[1]; //det网络只有一个输入
    rknn_output outputs[1]; //det网络只有一个输出

    //task
    double det_db_thresh_ = 0.3;
    double det_db_box_thresh_ = 0.5;
    double det_db_unclip_ratio_ = 2.0;
    bool use_polygon_score_ = false;

    // input image
    int HEIGHT_ = 480;
    int WIDTH_ = 640;
    int CHANNEL_ = 3;

    vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};

    // output result
    PostProcessor post_processor_;
  
};

