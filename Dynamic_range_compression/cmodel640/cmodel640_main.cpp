#define NOMINMAX

#include <filesystem>
#include <iostream>
#include <ShlObj.h>

#include "cmodel640_main.h"
#include "lsc.h"
#include "dpdc.h"
#include "gamma.h"
#include "colormap.h"
#include "clip.h"
#include "ours_bilateral.h"
#include "adc.h"
#include "isp_sharpen_function.h"


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

int imwrite_private_normalization_8u(cv::InputArray input, const std::string file_name)
{
    cv::Mat src = input.getMat();

    cv::Mat normalized;
    cv::normalize(src, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    return imwrite_private(normalized, file_name);
}

int cmodel_640_process()
{
    int height = 512;
    int width = 640;

    int dead_coords[13][2] = {
        { 420, 5 },{ 437, 5 },{ 393, 114 },{ 410, 114 },
        { 338, 162 },{ 355, 162 },{ 627, 170 },{ 4, 171 },
        { 503, 192 },{ 520, 192 },{ 35, 447 },/*{ 11, 456 },*/
        { 18, 447 }
    };

	int dead_sum = sizeof(dead_coords) / sizeof(dead_coords[0]);

    std::cout << "读取输入图像" << std::endl;

    std::string self_png = "C:/Users/Belfast/Desktop/640clip/drc640";

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
		std::cout << "img_in minval=" << minVal << " maxVal=" << maxVal << std::endl;

        //--------------------模拟静态坏点------------------------------------
        //img_in.at<ushort>(1, 10) = 0;img_in.at<ushort>(1, 20) = 0;img_in.at<ushort>(1, 30) = 0;
        int (*DEAD)[2] = nullptr;
        int dead_count = 0;

        DEAD = dead_coords;
        dead_count = dead_sum;

        std::cout << "DEAD POINT " << DEAD << std::endl;
        imwrite_private_normalization_8u(img_in,"cmodel640_src");
		//imwrite_private_normalization_8u(img_i, "src_resize");

        //------------------使能信号-------------------------------
        bool unc_en = 0;
        bool b_change = 0;
        bool k_change = 0;
        bool lsc_en = 0;
        bool dpc_en = 1;
        bool dn_en = 1;
        bool clip_en = 1;
        bool adc_en = 1;
        bool sp_en = 0;

        int Y_value = 12000;//ooc标定值
        int dpc_thr = 3000;

        double dp_slope = 6;    // 斜率参数    控制检测灵敏度，差值每增加1，检测值增加0.001，举例：差值为1000时，检测值 = 1000×0.001 - 0.3 = 0.7
        double dp_thresh = 250;   // 阈值参数    设置检测阈值的基础偏移，即使差值为0，检测值也是-0.3（被限制为0）;
        int dp_diff_th = 100;           // DPC差值阈值   中心像素与邻域像素差值超过该阈值才考虑为坏点
        int sigma_space = 10;
        int sigma_color = 10;
        float low_percent = 2.0f;
        int high_percent = 1.0f;
        int Thr = 300;
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

#if 0
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
#endif
        cv::Mat img_lsc = img_i.clone();
        imwrite_private_normalization_8u(img_lsc, "lsc");

        // dpc
        cv::Mat img_dpc = cv::Mat::zeros(height, width, CV_16UC1);
        ours_dpc(img_lsc, dpc_en, img_dpc, dp_slope, dp_thresh, dp_diff_th, DEAD, dead_count);//基于像素差值的动态阈值，斜率参数和阈值参数可调，适应不同场景3*5

        imwrite_private_normalization_8u(img_dpc, "dpc");   
        cv::minMaxLoc(img_dpc, &minVal, &maxVal);
        std::cout << "img_dpc minVal=" << minVal << " maxVal=" << maxVal << std::endl;

		// dn
        cv::Mat img_dn = cv::Mat::zeros(height, width, CV_16UC1);
        ours_bilateral(img_dpc, img_dn, dn_en, sigma_color, sigma_space);

		imwrite_private_normalization_8u(img_dn, "dn");
        cv::minMaxLoc(img_dn, &minVal, &maxVal);
        std::cout << "img_dn minVal=" << minVal << " maxVal=" << maxVal << std::endl;


        const int total_pixels = height * width;
        int low_number = static_cast<int>(total_pixels * low_percent / 100.0f);
        int high_number = total_pixels -
            static_cast<int>(total_pixels * high_percent / 100.0f);

        // 处理图像并获取最小像素信息
        std::vector<cv::Point> min_positions;
        std::vector<uint16_t> min_values;
        int autothreshold_low = 0;
        int autothreshold_high = 16383;
        std::cout << "clip low_number=" << low_number << " high_number=" << high_number << std::endl;

        // clip
        cv::Mat img_clip = histogram_equalization_clipped_fixed2(
            img_dn, min_positions, min_values, clip_en, low_number, high_number,
            autothreshold_low, autothreshold_high);

        std::cout << "autothreshold_low=" << autothreshold_low
            << " autothreshold_high=" << autothreshold_high << '\n';

        // 输出14bit图像的最小像素信息
        std::cout << "14bit图像中像素值最低的16个像素：" << std::endl;
        std::cout << "值范围: 0-16383 (14bit)" << std::endl;
        for (size_t i = 0; i < min_positions.size(); ++i) {
            std::cout << "值: " << min_values[i]
                << "\t位置: (" << min_positions[i].x
                << ", " << min_positions[i].y << ")" << std::endl;
        }

		imwrite_private_normalization_8u(img_clip, "clip");
        cv::minMaxLoc(img_clip, &minVal, &maxVal);
        std::cout << "img_clip minVal=" << minVal << " maxVal=" << maxVal << std::endl;

        // adc
        cv::Mat img2_adc = cv::Mat::zeros(height, width, CV_8UC1);
        adc(img_clip, img2_adc, adc_en, Thr, delta);

        cv::Mat adc_inversion;
		cv::subtract(255, img2_adc, adc_inversion);
		imwrite_private(adc_inversion, "adc");

		// sharpen
        cv::Mat img2_sp = cv::Mat::zeros(height, width, CV_8UC1);
        sharpenImage_int8(img2_adc, img2_sp, sp_en, temp_kernel, temp_akt);

        imwrite_private(img2_sp, "sharpen");

		// gamma
        cv::Mat img_gm = cv::Mat::zeros(height, width, CV_8UC1);
        gamma(img2_sp, img_gm, gamma_type);

        imwrite_private(img_gm, "gamma");

		// colormap
        cv::Mat img_color = cv::Mat::zeros(height, width, CV_8UC3);
        ColorMap(img_gm, img_color, colormap_sel, gamma_type);

		imwrite_private(img_color, "colormap");

    }

    return 0;
}

int fast_gamma()
{
    std::string self_png = "C:/Users/Belfast/Desktop/before_mapping/2026-09-18_17-37-04";

    fs::path inputDir(self_png);

    if (!fs::exists(inputDir))
    {
        std::cerr << "Input directory not found: " << self_png << std::endl;
        return -1;
    }

    std::array<uchar, 256> jungle_lut{};
    std::array<uchar, 256> bird_lut{};

    for (int i = 0; i < 256; i++)
    {
        jungle_lut[i] = cv::saturate_cast<uchar>(std::floor(std::pow(static_cast<double>(i) / 255.0, 0.4) * 255.0 + 0.5));
        bird_lut[i] = cv::saturate_cast<uchar>(std::floor(std::pow(static_cast<double>(i) / 255.0, 0.25) * 255.0 + 0.5));
    }

    std::vector<cv::Mat> images;
    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        // 过滤 bmp 文件
        if (!entry.is_regular_file() || entry.path().extension() != ".bmp")
            continue;

        cv::Mat src = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (src.empty())
        {
            std::cerr << "Failed to read: " << entry.path() << std::endl;
            continue;
        }

        cv::Mat grayImage;
        cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);

        // 处理图像
        CV_CheckTypeEQ(grayImage.type(), CV_8UC1, "");
        cv::subtract(255, grayImage, grayImage);

        images.push_back(grayImage);
    }


    for (int i = 0; i < images.size(); ++i)
    {
		cv::Mat dst = images[i].clone();

        cv::Mat jungle_lut_mat(1, 256, CV_8UC1, jungle_lut.data());
        cv::Mat bird_lut_mat(1, 256, CV_8UC1, bird_lut.data());

        cv::Mat jungle_result, bird_result;

        cv::LUT(dst, jungle_lut_mat, jungle_result);
        cv::LUT(dst, bird_lut_mat, bird_result);

        imwrite_private(dst, "01_dst");
		imwrite_private(jungle_result, "02_jungle_result");
		imwrite_private(bird_result, "03_bird_result");
    }

    return 0;
}


int cmodel_640_main()
{
    //fast_gamma();

    cmodel_640_process();

    return 0;
}


