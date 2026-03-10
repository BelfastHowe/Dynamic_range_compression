# CMakeList.txt: Dynamic_range_compression 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 将源代码添加到此项目的可执行文件。
add_executable (Dynamic_range_compression "Dynamic_range_compression.cpp" "Dynamic_range_compression.h" "quantitative_assessment.cpp" "quantitative_assessment.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Dynamic_range_compression PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
