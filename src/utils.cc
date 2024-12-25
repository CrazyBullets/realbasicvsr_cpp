#include "utils.hpp"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <fstream>
#include <iostream>
#include "logging.h"
#include "cuda_runtime_api.h"


utils::utils(/* args */cv::Mat& img_input)
{
    img = img_input;
    ROWS = img.rows;
    COLS = img.cols;
    CHANNELS = img.channels();
    
    input_size = ROWS*COLS*CHANNELS*2;
    output_size = input_size + ROWS*4*COLS*4*CHANNELS*2;

    input_data = std::vector<float>(input_size);
    output_data = std::vector<float>(output_size);
    output_img = cv::Mat(ROWS*4,COLS*4,CV_8UC3);
}

utils::~utils()
{
    // delete[]input_data;
    // delete[]output_data;
}

//添加两这张图（复制）；归一化到0-1之间；permute(2,0,1)；拓展维度到5维
int utils::pre_process(){
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    double max_val;
    cv::minMaxLoc(img, nullptr, &max_val, nullptr, nullptr);
    if(max_val < 256){
        std::cout << "image is in 0-255 range." << std::endl;
        // img.convertTo(img, CV_32F, 1.0/255.0);
    }
    else{
        std::cout << "image is in 0-65535 range." << std::endl;
        img.convertTo(img, CV_32F, 1.0/65535.0);
    }


    
    
    //(h,w,c) -> (c,h,w) b (0,bathc,)
    for(int row = 0; row < ROWS; row++){
        for(int col = 0; col < COLS; col++){
            for(int channel = 0; channel < CHANNELS; channel++){
                //img(row, col, channel) -> input_data(channel, row, col)
                // int new_idx = channel*ROWS*COLS + row*COLS + col;
                int new_idx = (row*COLS+col)*CHANNELS+channel;
                // std::cout << "input_data[" <<  static_cast<float>(img.ptr<uchar>(row, col)[channel]) << std::endl;
                input_data[new_idx] = img.ptr<uchar>(row, col)[channel]/255.;
                input_data[new_idx + input_size/2] = input_data[new_idx];
            }
        }
    }
   
    
    

    
    return 0;
}
//推理
int utils::inference(std::string  model_path){
    //读取二进制engine文件
    std::string engine_path = model_path;
    std::ifstream file(engine_path, std::ios::binary);
    char*trt_model_stream = NULL;
    int size = 0;
    if(file.good()){
        file.seekg(0,file.end);
        size = file.tellg();
        file.seekg(0,file.beg);
        trt_model_stream = new char[size];
        assert(trt_model_stream);

        file.read(trt_model_stream,size);
        file.close();
    }


    Logger glogger;
    nvinfer1::IRuntime*runtime = nvinfer1::createInferRuntime(glogger);
    assert(runtime != nullptr);

    nvinfer1::ICudaEngine * engine = runtime->deserializeCudaEngine(trt_model_stream,size);
    assert(engine != nullptr);


    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trt_model_stream;

    void**data_buffer = new void*[2];
    int input_node_index = engine->getBindingIndex("input");
    cudaMalloc(&(data_buffer[input_node_index]), input_size * sizeof(float));

    int output_node_index = engine->getBindingIndex("output");
    cudaMalloc(&(data_buffer[output_node_index]), input_size*4 * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(data_buffer[0],input_data.data(),input_size*sizeof(float),cudaMemcpyHostToDevice,stream);


    context->enqueueV2(data_buffer,stream,nullptr);
    cudaMemcpyAsync(output_data.data(),data_buffer[output_node_index],input_size*4*sizeof(float),cudaMemcpyDeviceToHost,stream);
    return 0;
}

// output = np.transpose(outputs, (0, 1, 3, 4, 2))[0, i, :, :, :]*255；保存图片
int utils::post_process(){
    //(c,h,w)  -> (h,w,c)

    // std::cerr<<"img_output row,cols,channel:"<<output_img.rows<<","<<output_img.cols<<","<<output_img.channels()<<std::endl;
    // for(int channel = 0;channel < CHANNELS;channel++){
    //     for(int row = 0; row < ROWS*4; row++){
    //         for(int col = 0; col < COLS*4; col++){
                
    //             output_img.ptr<uchar>(row,col)[channel] = static_cast<uchar>(output_data[channel*ROWS*4*COLS*4 + row*COLS*4 + col + input_size])*255.;
    //         }
    //     }
    // }

    for(int row = 0; row < ROWS*4; row++){
        for(int col = 0; col < COLS*4; col++){
            for(int channel = 0; channel < CHANNELS; channel++){
                //img(row, col, channel) -> input_data(channel, row, col)
                // int new_idx = channel*ROWS*COLS + row*COLS + col;
                int new_idx = (row*COLS*4+col)*CHANNELS+channel + input_size;
                // std::cout << "input_data[" <<  static_cast<float>(img.ptr<uchar>(row, col)[channel]) << std::endl;
            
                output_img.ptr<uchar>(row,col)[channel] = static_cast<uchar>(output_data[new_idx])*255.;
            }
        }
    }
    // cv::cvtColor(output_img, output_img, cv::COLOR_RGB2BGR);
    if(cv::imwrite("/root/zst/Realbasicvsr/realbasicvsr_cpp/results/test.png",output_img)){
        std::cout<<"color png is  saved successfully in /root/zst/Realbasicvsr/realbasicvsr_cpp/results/test.png"<<std::endl;
    }

    return 0;
}



