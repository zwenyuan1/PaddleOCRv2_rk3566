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

using namespace cv;
using namespace std;

class TextRec{
public:
    TextRec(){
        this->label_list_ = Utility::ReadDict(this->label_path);
        this->label_list_.insert(this->label_list_.begin(), "#"); // blank char for ctc
        this->label_list_.push_back(" ");
    };
    int Model_Init(const char* rk_model_name);
    int Model_Infer(cv::Mat Input_Image, pair< vector<string>, double> &rec_res, double &times);
    ~TextRec();

private:
    vector<string> label_list_;
    string label_path = "./model/ppocr_keys_v1.txt";

    //config
    int ret;
    rknn_context ctx;
    rknn_tensor_attr input_attrs[1];
    rknn_tensor_attr output_attrs[1];
    rknn_input inputs[1]; //det网络只有一个输入
    

    int HEIGHT_ = 32;
    int WIDTH_ = 96;
    int CHANNEL_ = 3;

    vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};



};