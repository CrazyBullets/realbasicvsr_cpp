#ifndef UTILS_HPP
#define UTILS_HPP
#include <opencv2/opencv.hpp>

class utils
{
private:
    /* data */
public:
    utils(/* args */cv::Mat& img_input);
    ~utils();

    int pre_process();
    int inference();
    int post_process();

    // float *input_data;
    cv::Mat img;
    std::vector<float> input_data;
    int input_size;

    float*output_data;
};





#endif // UTILS_HPP