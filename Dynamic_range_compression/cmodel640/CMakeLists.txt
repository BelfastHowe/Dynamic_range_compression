# 将新文件夹下的代码编译成一个独立的静态库，名为 model640
add_library(cmodel640 STATIC
 "cmodel640_main.cpp"
 "cmodel640_main.h"
 "lsc.cpp"
 "lsc.h"
 "dpdc.h"
 "dpdc.cpp"
 "gamma.h"
 "gamma_correction.cpp" "colormap.h" "applycolormap_black.cpp")

# 暴露头文件路径给主程序
target_include_directories(cmodel640 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 如果新代码中也用到了 OpenCV，给它链接 OpenCV
#find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(cmodel640 PRIVATE ${OpenCV_LIBS})
