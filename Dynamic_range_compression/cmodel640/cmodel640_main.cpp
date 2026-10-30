#define NOMINMAX

#include <filesystem>
#include <iostream>
#include <ShlObj.h>

#include "cmodel640_main.h"
#include "lsc.h"
#include "dpdc.h"
#include "gamma.h"
#include "colormap.h"


namespace fs = std::filesystem;
using Ms = std::chrono::duration<double, std::milli>;

inline
int imwrite_private(cv::InputArray input, const std::string file_name)
{
    cv::Mat src = input.getMat();

    PWSTR desktop_path_w = nullptr;
    HRESULT hr = SHGetKnownFolderPath(FOLDERID_Desktop, 0, NULL, &desktop_path_w);

    if (FAILED(hr))
    {
        std::cerr << "获取桌面路径失败！" << std::endl;
        return -1;
    }

    std::filesystem::path result_dir = std::filesystem::path(desktop_path_w) / "result";

    CoTaskMemFree(desktop_path_w);
    desktop_path_w = nullptr;

    std::filesystem::create_directories(result_dir);

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);

    std::tm now_tm;
    localtime_s(&now_tm, &now_c);

    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y%m%d_%H%M%S_") << now.time_since_epoch().count() << std::string("_");

    std::filesystem::path output_path = result_dir / (oss.str() + file_name + ".png");
    std::string output_file_name = output_path.string();

    //std::cout << "已保存至: " << output_file_name << std::endl;
    cv::imwrite(output_file_name, src);
    cv::waitKey(1);

    return 0;
}

int cmodel_640_main()
{
    int height = 512;
    int width = 640;

    std::cout << "读取输入图像" << std::endl;

    std::string self_png = "C:/Users/Belfast/Desktop/640";

    fs::path inputDir(self_png);
    //fs::path inputDir(NOISE_IMAGE_DIR);
    //fs::path inputDir(IMAGE_DIR);

    if (!fs::exists(inputDir))
    {
        std::cerr << "Input directory not found: " << self_png << std::endl;
        return -1;
    }

    std::vector<cv::Mat> images;
    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        // 过滤 png 文件
        if (!entry.is_regular_file() || entry.path().extension() != ".png")
            continue;

        cv::Mat src = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        if (src.empty())
        {
            std::cerr << "Failed to read: " << entry.path() << std::endl;
            continue;
        }

        // 处理图像
        CV_CheckTypeEQ(src.type(), CV_16UC1, "");
        cv::subtract(16383, src, src);

        images.push_back(src);
    }

    // 4. 处理所有找到的图像
    for (int i = 0; i < images.size(); ++i)
    {
        cv::Mat img_in = images[i]; //CV_16UC1
        double minVal, maxVal;
        cv::Size targetSize(width, height);
        cv::Mat img_i = cv::Mat::zeros(height, width, CV_16UC1);
        cv::resize(img_in, img_i, targetSize, 0, 0, cv::INTER_LINEAR);

        cv::minMaxLoc(img_i, &minVal, &maxVal);
        std::cout << "img_in minval=" << minVal << " maxVal=" << maxVal << '\n';

        //--------------------模拟静态坏点------------------------------------
        //img_in.at<ushort>(1, 10) = 0;img_in.at<ushort>(1, 20) = 0;img_in.at<ushort>(1, 30) = 0;
        int (*DEAD)[2] = nullptr;
        int dead_count = 0;
        std::cout << "DEAD POINT " << DEAD << std::endl;
        imwrite_private(img_in,"cmodel640_src");

        //------------------使能信号-------------------------------
        bool unc_en = 0;
        bool b_change = 0;
        bool k_change = 0;
        bool lsc_en = 1;
        bool dpc_en = 1;
        bool dn_en = 1;
        bool clip_en = 1;
        bool adc_en = 1;
        bool sp_en = 1;

        int Y_value = 12000;//ooc标定值
        int dpc_thr = 3000;

        double dp_slope = 1;    // 斜率参数    控制检测灵敏度，差值每增加1，检测值增加0.001，举例：差值为1000时，检测值 = 1000×0.001 - 0.3 = 0.7
        double dp_thresh = 300;   // 阈值参数    设置检测阈值的基础偏移，即使差值为0，检测值也是-0.3（被限制为0）;
        int dp_diff_th = 1000;           // DPC差值阈值   中心像素与邻域像素差值超过该阈值才考虑为坏点
        int sigma_space = 10;
        int sigma_color = 30;
        float low_percent = 0.5;
        int high_percent = 2;
        int Thr = 600;
        int delta = 0;
        uint8_t A = (1 << 7);
        int temp_kernel[3][3] = {
        { 0,  -16,  0 },
        { -16, 64, -16 },
        { 0,  -16,  0 }
        };
        int temp_akt[3] = { 64,32,4 };
        std::cout << "temp_akt=" << temp_akt[0] << " " << temp_akt[1] << " " << temp_akt[2] << '\n';
        int gamma_type = 0;
        int colormap_sel = 0;

        //获取图像镜头阴影校正校正标定信息！！！
        int16_t** K_lsc = new int16_t * [height];
        for (int i = 0; i < height; i++) {
            K_lsc[i] = new int16_t[width]{};
        }
        int16_t** B_lsc = new int16_t * [height];
        for (int i = 0; i < height; i++) {
            B_lsc[i] = new int16_t[width]{};
        }

        //读 10+10 张图做标定
        std::cout << "求两个温度下ooc平均图像" << std::endl;

        std::string lsc1_path = "./ooc_lsc/26-10-12/cold";
        cv::Mat lsc_avg1 = avg(lsc1_path, 50);
        cv::Mat lsc_avg11 = cv::Mat::zeros(height, width, CV_16UC1);
        cv::resize(lsc_avg1, lsc_avg11, targetSize, 0, 0, cv::INTER_LINEAR);

        std::string lsc2_path = "./ooc_lsc/26-10-12/hot";
        cv::Mat lsc_avg2 = avg(lsc2_path, 50);
        cv::Mat lsc_avg21 = cv::Mat::zeros(height, width, CV_16UC1);
        cv::resize(lsc_avg2, lsc_avg21, targetSize, 0, 0, cv::INTER_LINEAR);

        calibrate(lsc_avg11, lsc_avg21, K_lsc, B_lsc);
        write_kb_coe("gain_kb_16bit.coe", K_lsc, B_lsc, height, width);
        cv::Mat img_lsc = cv::Mat::zeros(height, width, CV_16UC1);
        lsc(img_i, K_lsc, B_lsc, img_lsc, lsc_en);

		imwrite_private(img_lsc, "cmodel640_lsc");

        cv::Mat img_dpc = cv::Mat::zeros(height, width, CV_16UC1);

        ours_dpc(img_lsc, dpc_en, img_dpc, dp_slope, dp_thresh, dp_diff_th, DEAD, dead_count);//基于像素差值的动态阈值，斜率参数和阈值参数可调，适应不同场景3*5

    }

    return 0;
}



