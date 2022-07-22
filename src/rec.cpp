# include "rec.h"
static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
    unsigned char* data;
    int            ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
    FILE* fp;
    unsigned char* data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int TextRec::Model_Init(const char* rk_model_name){
    //printf("Loading model....\n");
    int model_data_size = 0;
    unsigned char* model_data = load_model(rk_model_name, &model_data_size);

    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    
    memset(input_attrs, 0, sizeof(input_attrs));
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[0]), sizeof(rknn_tensor_attr));
    //dump_tensor_attr(&(input_attrs[0]));

    memset(output_attrs, 0, sizeof(output_attrs));
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[0]), sizeof(rknn_tensor_attr));
    //dump_tensor_attr(&(output_attrs[0]));

    CHANNEL_ = 3;
    HEIGHT_  = input_attrs[0].dims[1];
    WIDTH_  = input_attrs[0].dims[2];

    //printf("model input height=%d, width=%d, channel=%d\n", HEIGHT_, WIDTH_, CHANNEL_);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = WIDTH_ * HEIGHT_ * CHANNEL_;
    inputs[0].fmt          = RKNN_TENSOR_NHWC; 
    inputs[0].pass_through = 0;

    return 1;

}

int TextRec::Model_Infer(cv::Mat Input_Image, pair< vector<string>, double> &rec_res, double &times){
    cv::Mat rgb_img = Input_Image.clone();
    cv::cvtColor(Input_Image, rgb_img, cv::COLOR_BGR2RGB);
    cv::Mat resize_img = rgb_img.clone();

    struct timeval start_time, stop_time;
    // 1. 图像前处理
    //resize image and get image data to input[0].buf
    float ratio_h=1.0;
    float ratio_w=1.0;

    if(Input_Image.rows!=HEIGHT_ || Input_Image.cols!=WIDTH_){
        cv::Mat padding_rgb_img = rgb_img.clone();
        int target_col = rgb_img.rows*(WIDTH_/HEIGHT_);
        if(target_col > rgb_img.cols)
            cv::copyMakeBorder(rgb_img, padding_rgb_img, 0, 0, (target_col-rgb_img.cols)/2, (target_col-rgb_img.cols)/2, BORDER_CONSTANT, cv::Scalar(143, 143, 145) );
        cv::resize(padding_rgb_img, resize_img, cv::Size(WIDTH_, HEIGHT_), (0, 0), (0, 0), cv::INTER_LINEAR);
        ratio_h = float(HEIGHT_)/Input_Image.rows;
        ratio_w = float(WIDTH_)/Input_Image.cols;
    }

    inputs[0].buf = resize_img.data;

    //2. 网络推理
    gettimeofday(&start_time, NULL);
    // 设置模型的输入数据
    rknn_inputs_set(ctx, 1, inputs);

    // 设置输出数据
    rknn_output outputs[1]; //det网络只有一个输出
    memset(outputs, 0, sizeof(outputs));
    outputs[0].is_prealloc = 0;
    outputs[0].want_float = 1;

    // 进行推理
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    times =  (__get_us(stop_time) - __get_us(start_time)) / 1000;
    //printf("once run use %f ms\n", times);

    //3.后处理
    float* outBlob = (float*)outputs[0].buf;

    vector<int> predict_shape;
    for(int j=0; j<output_attrs[0].n_dims; j++) 
        predict_shape.push_back(output_attrs[0].dims[j]);

    std::vector<std::string> str_res;
    int argmax_idx;
    int last_index = 0;
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    for (int n = 0; n < predict_shape[1]; n++) { // n = 2*l + 1
        argmax_idx =
            int(Utility::argmax(&outBlob[(n) * predict_shape[2]],
                                &outBlob[(n + 1) * predict_shape[2]]));
        max_value =
            float(*std::max_element(&outBlob[(n) * predict_shape[2]],
                                    &outBlob[(n + 1) * predict_shape[2]]));

        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
            score += max_value;
            count += 1;
            str_res.push_back(this->label_list_[argmax_idx]);
        }
        last_index = argmax_idx;
    }
    score /= count;
    if (isnan(score)){
        
        cout<<"there is nothing"<<endl;
    }   
    rec_res.first=str_res;
    rec_res.second=score;

    ret = rknn_outputs_release(ctx, 1, outputs);
    return 1;

}

TextRec::~TextRec(){
    ret = rknn_destroy(ctx);
}

