# include "det.h"


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
    FILE*          fp;
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

static int saveFloat(const char* file_name, float* output, int element_size)
{
    FILE* fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++) {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

int TextDetect::Model_Init(const char* rk_model_name){
    //printf("Loading model...\n");
    int model_data_size = 0;
    unsigned char* model_data = load_model(rk_model_name, &model_data_size);

    // 1. 创建rknn_context对象
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 2. 确定输入/输出的各种属性，创建rknn_input和rknn_output
    // 通过context获取网络输入/输出的个数分别是多少
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    //printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 通过context获取输入/输出tensor的属性 
    rknn_tensor_attr input_attrs[1];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        //dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[1];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        //dump_tensor_attr(&(output_attrs[i]));
    }

    //
    int channel = 3;
    int width   = 0;
    int height  = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) { 
        //printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height   = input_attrs[0].dims[2]; 
        width  = input_attrs[0].dims[3];
    } else {
        //printf("model is NHWC input fmt\n");
        height   = input_attrs[0].dims[1];
        width  = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    //printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = width * height * channel;
    inputs[0].fmt          = RKNN_TENSOR_NHWC; 
    inputs[0].pass_through = 0;

    return 1;
}

int TextDetect::Model_Infer(cv::Mat& Input_Image, vector<vector<vector<int>>> &boxes, double &times){  
    cv::Mat rgb_img = Input_Image.clone();
    cv::cvtColor(Input_Image, rgb_img, cv::COLOR_BGR2RGB);
    cv::Mat resize_img = rgb_img.clone();

    struct timeval start_time, stop_time;
    // 1. 图像前处理
    //resize image and get image data to input[0].buf
    float ratio_h=1.0;
    float ratio_w=1.0;

    if(Input_Image.rows!=HEIGHT_ || Input_Image.cols!=WIDTH_){
        cv::resize(rgb_img, resize_img, cv::Size(WIDTH_, HEIGHT_), (0, 0), (0, 0), cv::INTER_LINEAR);
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
    outputs[0].want_float = 1;

    // 进行推理
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    times =  (__get_us(stop_time) - __get_us(start_time)) / 1000;
    //printf("once run use %f ms\n", times);

    //3. 输出后处理

    std::vector<float> pred(HEIGHT_ * WIDTH_, 0.0);
    std::vector<unsigned char> cbuf(HEIGHT_ * WIDTH_, ' ');

    float* outBlob = (float*)outputs[0].buf;

    for (int i = 0; i < HEIGHT_ * WIDTH_; i++) {
        pred[i] = float(outBlob[i]);
        cbuf[i] = (unsigned char)((outBlob[i]) * 255);
    }

    cv::Mat cbuf_map(HEIGHT_, WIDTH_, CV_8UC1, (unsigned char *)cbuf.data());
    cv::Mat pred_map(HEIGHT_, WIDTH_, CV_32F, (float *)pred.data());

    const double threshold = this->det_db_thresh_ * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    cv::Mat dilation_map;
    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, dilation_map, dila_ele);

    boxes = post_processor_.BoxesFromBitmap(
        pred_map, dilation_map, this->det_db_box_thresh_,
        this->det_db_unclip_ratio_, this->use_polygon_score_);

    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, resize_img); // 将resize_img中得到的bbox 映射回srcing中的bbox
    cout<<"successful get "<<boxes.size()<<" boxes"<<endl;
    ret = rknn_outputs_release(ctx, 1, outputs);
    return 0;
}


TextDetect::~TextDetect(){
    ret = rknn_destroy(ctx);
}

