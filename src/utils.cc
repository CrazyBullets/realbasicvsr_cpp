#include "utils.hpp"


utils::utils(/* args */cv::Mat& img_input)
{
    img = img_input;
    input_size = img.rows*img.cols*img.channels()*2;
    input_data = std::vector<float>(input_size);
}

utils::~utils()
{
}

//添加两这张图（复制）；归一化到0-1之间；permute(2,0,1)；拓展维度到5维
int utils::pre_process(){
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    double max_val;
    cv::minMaxLoc(img, nullptr, &max_val, nullptr, nullptr);
    if(max_val < 256){
        std::cout << "image is in 0-255 range." << std::endl;
        img.convertTo(img, CV_32F, 1.0/255.0);
    }
    else{
        std::cout << "image is in 0-65535 range." << std::endl;
        img.convertTo(img, CV_32F, 1.0/65535.0);
    }


    int ROWS = img.rows, COLS = img.cols, CHANNELS = img.channels();
    
    // float* input_data_ptr = input_data.data();
    for(int row = 0; row < ROWS; row++){
        for(int col = 0; col < COLS; col++){
            for(int channel = 0; channel < CHANNELS; channel++){
                //img(row, col, channel) -> input_data(channel, row, col)
                int new_idx = channel*ROWS*COLS + row*COLS + col;
                // std::cout << "input_data[" <<  static_cast<float>(img.ptr<uchar>(row, col)[channel]) << std::endl;
                input_data[new_idx] = img.ptr<uchar>(row, col)[channel];
                input_data[new_idx + input_size/2] = input_data[new_idx];
            }
        }
    }
   
    
    

    
    return 0;
}
//推理
int utils::inference(){
    
    return 0;
}

// output = np.transpose(outputs, (0, 1, 3, 4, 2))[0, i, :, :, :]*255；保存图片
int utils::post_process(){

    return 0;
}



