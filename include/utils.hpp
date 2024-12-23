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
    int inference(std::string  model_path);
    int post_process();

    // float *input_data;
    cv::Mat img;
    int ROWS,COLS,CHANNELS;
    std::vector<float>input_data;
    int input_size;
    std::vector<float>output_data;
    int output_size;
    cv::Mat output_img;

};





#endif // UTILS_HPP