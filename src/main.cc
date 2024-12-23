#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include "utils.hpp"
// #include "NvInfer.h"

int main() {
    cv::Mat img_input = cv::imread("C:/Codes/CPP/real_basicvsr/data/00003.png",cv::IMREAD_COLOR);
    
    utils utils(img_input);
    int ret = utils.pre_process();

    if(ret== 0){
        std::cout<<"process success"<<std::endl;
    }
    return 0;
}