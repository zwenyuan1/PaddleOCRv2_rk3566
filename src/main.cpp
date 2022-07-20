# include "det.h"
# include "rec.h"

/*
int camera_ocr(){
    // det and rec init
    TextDetect* td = new TextDetect();
    td->Model_Init("./model/qua_det.rknn");

    TextRec*tr_3 = new TextRec();
    tr_3->Model_Init("./model/ch_rec_3.rknn");

    TextRec*tr_5 = new TextRec();
    tr_5->Model_Init("./model/ch_rec_5.rknn");

    VideoCapture cap(-1);
    cv::Mat test_img;
    while(1){

        if(!cap.read(test_img)){
            std::cout <<"open camera is error!"<<std::endl;
            return -1;
        }
        // det infer
        vector<vector<vector<int>>> boxes;
        double det_times;
        td->Model_Infer(test_img, boxes, det_times);

        double rec_all_times=0;
        for (int j = 0; j < boxes.size(); j++) {
            cv::Mat crop_img;
            crop_img = Utility::GetRotateCropImage(test_img, boxes[j]);

            // find best adapt model
            float wh_ratio = float(crop_img.cols)/float(crop_img.rows);
            cout<<crop_img.cols<<","<<crop_img.rows<<":"<<wh_ratio<<endl;

            pair< vector<string>, double> rec_res;
            double rec_times;
            if(wh_ratio <= 3)
                tr_3->Model_Infer(crop_img, rec_res, rec_times);
            if(wh_ratio <=5 && wh_ratio > 3)
                tr_5->Model_Infer(crop_img, rec_res, rec_times);

            rec_all_times += rec_times;
                
            cout << "rec: ";
            for (int i = 0; i < rec_res.first.size(); i++) {
                std::cout<< rec_res.first[i];
            }
            cout << "\tscore: " << rec_res.second << std::endl;

        }
        cout<<"ocr time is "<< det_times + rec_all_times <<" ms"<<endl;
    }
    cout<<"finish ocr"<<endl;
    delete td;
    delete tr_3;
    delete tr_5;
    return 0; 
}
*/

int imgs_ocr(char* img_dir){
    // det and rec init
    TextDetect* td = new TextDetect();
    td->Model_Init("./model/qua_det.rknn");

    TextRec*tr_3 = new TextRec();
    tr_3->Model_Init("./model/ch_rec_3.rknn");

    TextRec*tr_5 = new TextRec();
    tr_5->Model_Init("./model/ch_rec_5.rknn");

    TextRec*tr_10 = new TextRec();
    tr_10->Model_Init("./model/ch_rec_10.rknn");

    vector<String> all_img_names;
    cv::glob(img_dir, all_img_names);

   
    //ofstream file("./rec_res.csv");
    vector<string> best_res;
    vector<double> best_res_score;
    double best_score = 0;
    for(int i=0; i<all_img_names.size(); i++){
        cout << all_img_names[i] << endl;
        //file << all_img_names[i] << "\n";
        cv::Mat test_img = cv::imread(all_img_names[i], -1);
         // det infer
        vector<vector<vector<int>>> boxes;
        double det_times;
        td->Model_Infer(test_img, boxes, det_times);

        double rec_all_times=0;
        vector<string> a_frame_res;
        vector<double> a_frame_score;
        double mean_score=0;
        for (int j = 0; j < boxes.size(); j++) {
            cv::Mat crop_img;
            crop_img = Utility::GetRotateCropImage(test_img, boxes[j]);

            // find best adapt model
            float wh_ratio = float(crop_img.cols)/float(crop_img.rows);
            //cout<<crop_img.cols<<","<<crop_img.rows<<":"<<wh_ratio<<endl;

            pair< vector<string>, double> rec_res;
            double rec_times;
            if(wh_ratio <= 3)
                tr_3->Model_Infer(crop_img, rec_res, rec_times);
            if(wh_ratio <=5 && wh_ratio > 3)
                tr_5->Model_Infer(crop_img, rec_res, rec_times);
            if(wh_ratio > 5)
                tr_10->Model_Infer(crop_img, rec_res, rec_times);

            rec_all_times += rec_times;

            string res="";
            for (int i = 0; i < rec_res.first.size(); i++) {
                //std::cout<< rec_res.first[i];
                res += rec_res.first[i];
            }
            
            if(res=="" || isnan(rec_res.second)|| rec_res.second < 0.3)
                continue;
            //file << res << ","<< rec_res.second << "\n";
            if(mean_score == 0)
                mean_score = rec_res.second; // the first 
            mean_score = (mean_score + rec_res.second)/2;
            a_frame_res.push_back(res);
            a_frame_score.push_back(rec_res.second);

            //cout << "rec: "<< res <<" , score: " << rec_res.second <<" , wh_ratio: "<<wh_ratio<< endl;
        }
        if(mean_score > best_score){
            best_score = mean_score;
            best_res = a_frame_res;
            best_res_score = a_frame_score;
        }
        cout<<"a image ocr time is "<< det_times + rec_all_times/boxes.size() <<" ms"<<endl;
    }
    for(int i=0; i<best_res.size(); i++){
        cout << "rec: "<< best_res[i] <<" , score: " << best_res_score[i] << endl;
    }
    //file.close();
    return 1;
}

int img_ocr(char* img_path){
    // det and rec init
    TextDetect* td = new TextDetect();
    td->Model_Init("./model/qua_det.rknn");

    TextRec*tr_3 = new TextRec();
    tr_3->Model_Init("./model/ch_rec_3.rknn");

    TextRec*tr_5 = new TextRec();
    tr_5->Model_Init("./model/ch_rec_5.rknn");

    TextRec*tr_10 = new TextRec();
    tr_10->Model_Init("./model/ch_rec_10.rknn");
    
    cv::Mat test_img = cv::imread(img_path, -1);

    // det infer
    vector<vector<vector<int>>> boxes;
    double det_times;
    td->Model_Infer(test_img, boxes, det_times);

    double rec_all_times=0;
    for (int j = 0; j < boxes.size(); j++) {
        cv::Mat crop_img;
        crop_img = Utility::GetRotateCropImage(test_img, boxes[j]);

        // find best adapt model
        float wh_ratio = float(crop_img.cols)/float(crop_img.rows);
        //cout<<crop_img.cols<<","<<crop_img.rows<<":"<<wh_ratio<<endl;

        pair< vector<string>, double> rec_res;
        double rec_times;
        if(wh_ratio <= 3)
            tr_3->Model_Infer(crop_img, rec_res, rec_times);
        if(wh_ratio <=5 && wh_ratio > 3)
            tr_5->Model_Infer(crop_img, rec_res, rec_times);
        if(wh_ratio > 5)
            tr_10->Model_Infer(crop_img, rec_res, rec_times);

        rec_all_times += rec_times;

        string res="";
        for (int i = 0; i < rec_res.first.size(); i++) {
            //std::cout<< rec_res.first[i];
            res += rec_res.first[i];
        }
        
        if(res=="" || isnan(rec_res.second)|| rec_res.second < 0.3)
            continue;

        cout << "rec: "<< res <<" , score: " << rec_res.second <<" , wh_ratio: "<<wh_ratio<< endl;

    }
    cout<<"ocr time is "<< det_times + rec_all_times/boxes.size() <<" ms"<<endl;

    cout<<"finish ocr"<<endl;
    delete td;
    delete tr_3;
    delete tr_5;
    delete tr_10;

    return 0;
}

int main(int argc, char** argv){

    //return camera_ocr();
    if (argc != 2) {
        printf("Usage: %s <jpg> \n", argv[0]);
        return -1;
    }
    return imgs_ocr(argv[1]);
    //return img_ocr(argv[1]);

}