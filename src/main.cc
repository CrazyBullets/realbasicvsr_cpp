#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include "utils.hpp"
// #include "NvInfer.h"

int main() {
    cv::Mat img_input = cv::imread("/root/zst/Realbasicvsr/realbasicvsr_cpp/data/000.png",cv::IMREAD_COLOR);
    std::string model_path = "/root/zst/Realbasicvsr/realbasicvsr_cpp/engine/RealBasicvsr.trt";
    std::cerr<<"img_input row,cols,channel:"<<img_input.rows<<","<<img_input.cols<<","<<img_input.channels()<<std::endl;
    
    utils utils(img_input);
    int ret = utils.pre_process();
    ret = utils.inference(model_path);
    ret = utils.post_process();

    if(ret== 0){
        std::cout<<"process success"<<std::endl;
    }
    return 0;
}