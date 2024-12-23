# # 指定 CMake 的最小版本
# cmake_minimum_required(VERSION 3.10)

# # 设置项目名称和版本
# project(MyProject)

# 查找 OpenCV 版本
# set(OpenCV_DIR "C:/MyLibrary/OpenCV/opencv4_5_4_bin")
find_package(OpenCV REQUIRED)
# link_directories(${OpenCV_INCLUDE_DIRS})

# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)


find_package(CUDA REQUIRED)

# include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)


# # 查找TensorRT头文件路径
# find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h PATHS /usr/local/TensorRT-8.6.2/include)
# # 查找TensorRT核心库（nvinfer）路径
# find_library(TensorRT_LIBRARY NAMES nvinfer PATHS /usr/local/TensorRT-8.6.2/lib)
# # 查找TensorRT插件库（nvinfer_plugin）路径
# find_library(TensorRT_PLUGIN_LIBRARY NAMES nvinfer_plugin PATHS /usr/local/TensorRT-8.6.2/lib)

# if(NOT TensorRT_INCLUDE_DIR)
#     message(FATAL_ERROR "TensorRT include directory not found!")
# endif()

# if(NOT TensorRT_LIBRARY)
#     message(FATAL_ERROR "TensorRT library (nvinfer) not found!")
# endif()

# if(NOT TensorRT_PLUGIN_LIBRARY)
#     message(FATAL_ERROR "TensorRT plugin library (nvinfer_plugin) not found!")
# endif()
# # set(TensorRT_DIR "C:/MyLibrary/TensorRT-8.5.3.1")
# # find_package(TensorRT REQUIRED)
# # if(TensorRT_FOUND)
# #     include_directories(${TensorRT_INCLUDE_DIRS})
# #     target_link_libraries(your_target_name ${TensorRT_LIBRARIES})
# # endif(TensorRT_FOUND)

# ##CUDA配置
# find_library(CUDA_LIBRARY cuda PATHS /usr/local/cuda-12.2/lib64) 




#添加头文件目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${TensorRT_INCLUDE_DIR}
    )
link_directories(${TensorRT_LIBRARY} ${TensorRT_PLUGIN_LIBRARY})
aux_source_directory(. SRC)#添加所有源文件




# 添加可执行文件
add_executable(${PROJECT_NAME}_exec ${SRC})



# 链接库
target_link_libraries(${PROJECT_NAME}_exec ${OpenCV_LIBS} nvinfer cudart)




