#include "slider_display.h"


namespace fs = std::filesystem;

std::vector<cv::Mat> load_images(const std::string& image_dir)
{
    //std::string self_png = "C:/Users/Belfast/Desktop/before_mapping/30度黑体2";
    std::string self_png = "C:/Users/Belfast/Desktop/DDE";
    fs::path inputDir(self_png);
    //fs::path inputDir(NOISE_IMAGE_DIR);
    //fs::path inputDir(IMAGE_DIR);

    std::vector<cv::Mat> images;

    if (!fs::exists(inputDir))
    {
        std::cerr << "Input directory not found: " << NOISE_IMAGE_DIR << std::endl;
        return images;
    }

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

    return images;
}

cv::Mat build_grid(
    const std::vector<cv::Mat>& images,
    int cell_w,
    int cell_h)
{
    std::vector<cv::Mat> grid_imgs;

    for (const auto& img : images)
    {
        cv::Mat resize;
        cv::resize(
            img,
            resize,
            cv::Size(cell_w, cell_h)
        );

        // 对应 Python：
        // cv2.normalize(..., 0, 255, NORM_MINMAX, CV_8U)
        cv::Mat normalized;

        cv::normalize(
            resize,
            normalized,
            0,
            255,
            cv::NORM_MINMAX,
            CV_8U
        );

        // 单通道 -> BGR
        cv::Mat bgr;

        cv::cvtColor(
            normalized,
            bgr,
            cv::COLOR_GRAY2BGR
        );

        grid_imgs.push_back(bgr);
    }

    // 黑图
    cv::Mat blank(
        cell_h,
        cell_w,
        CV_8UC3,
        cv::Scalar(0, 0, 0)
    );

    // 补到16张
    while (grid_imgs.size() < 16)
    {
        grid_imgs.push_back(blank);
    }

    // 4x4
    std::vector<cv::Mat> rows;

    for (int i = 0; i < 4; ++i)
    {
        cv::Mat row;

        std::vector<cv::Mat> row_imgs(
            grid_imgs.begin() + i * 4,
            grid_imgs.begin() + (i + 1) * 4
        );

        cv::hconcat(row_imgs, row);

        rows.push_back(row);
    }

    cv::Mat grid;

    cv::vconcat(rows, grid);

    return grid;
}


int slider_display()
{
    // --------------------------------------------------------
    // 1. 加载图像
    // --------------------------------------------------------
    std::vector<cv::Mat> images =
        load_images(IMAGE_DIR);

    if (images.empty())
    {
        std::cout
            << "未找到图像："
            << IMAGE_DIR
            << std::endl;

        return 0;
    }

    std::cout
        << "加载了 "
        << images.size()
        << " 张图像"
        << std::endl;


    // --------------------------------------------------------
    // 2. 创建窗口
    // --------------------------------------------------------
    const std::string WIN = "CLAHE Tuner";

    cv::namedWindow(
        WIN,
        cv::WINDOW_NORMAL
    );


    // --------------------------------------------------------
    // 3. 窗口尺寸
    // --------------------------------------------------------
    const int SCREEN_W = 1536;
    const int SCREEN_H = 1152;

    const int cell_w = SCREEN_W / 4;
    const int cell_h = SCREEN_H / 4;


    // --------------------------------------------------------
    // 4. Trackbar
    // --------------------------------------------------------

    int plaRatio = 12;
    int minDR = 100;
    int lastGamma = 100;

    cv::createTrackbar(
        "plaRatio",
        WIN,
        &plaRatio,
        200
    );

    cv::createTrackbar(
        "minDR",
        WIN,
        &minDR,
        1000
    );

    cv::createTrackbar(
        "lastGamma",
        WIN,
        &lastGamma,
        200
    );


    // --------------------------------------------------------
    // 5. 上一次参数
    // --------------------------------------------------------
    int last_d = -1;
    int last_minDR = -1;
    double last_gamma = -1;


    // --------------------------------------------------------
    // 6. 主循环
    // --------------------------------------------------------
    while (true)
    {
        // ----------------------------------------------------
        // 获取Trackbar
        // ----------------------------------------------------
        int d = cv::getTrackbarPos(
            "plaRatio",
            WIN
        );

        int sigmaColor = cv::getTrackbarPos(
            "minDR",
            WIN
        );

        int sigmaSpaceInt = cv::getTrackbarPos(
            "lastGamma",
            WIN
        );


        // ----------------------------------------------------
        // 对应 Python 参数处理
        // ----------------------------------------------------
        d = std::max(d, 1);

        sigmaColor =
            std::max(sigmaColor, 10);

        sigmaSpaceInt =
            std::max(sigmaSpaceInt, 10);

        double sigmaSpace =
            sigmaSpaceInt / 100.0;


        // ----------------------------------------------------
        // 参数没有变化，不重新计算
        // ----------------------------------------------------
        if (
            d != last_d ||
            sigmaColor != last_minDR ||
            sigmaSpace != last_gamma
            )
        {
            // ------------------------------------------------
            // 处理16张图
            // ------------------------------------------------
            std::vector<cv::Mat> processed;

            for (const auto& img : images)
            {
                cv::Mat dst;

                // 这里调用你已有的 dde_guided
                //
                // Python：
                // dde_guided(
                //     img,
                //     d,
                //     sigmaColor,
                //     sigmaSpace
                // )
                //
                // C++：
               /* dde_guided(
                    img,
                    dst,
                    d,
                    sigmaColor,
                    sigmaSpace
                );*/

                processed.push_back(dst);
            }


            // ------------------------------------------------
            // 构建4x4 Grid
            // ------------------------------------------------
            cv::Mat grid =
                build_grid(
                    processed,
                    cell_w,
                    cell_h
                );


            // ------------------------------------------------
            // 左上角显示参数
            // ------------------------------------------------
            char label[128];

            std::snprintf(
                label,
                sizeof(label),
                "d: %.1f   sigma: %d x %.2f",
                static_cast<double>(d),
                sigmaColor,
                sigmaSpace
            );

            cv::putText(
                grid,
                label,
                cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 255),
                2
            );


            // ------------------------------------------------
            // 显示
            // ------------------------------------------------
            cv::imshow(
                WIN,
                grid
            );


            // ------------------------------------------------
            // 保存参数
            // ------------------------------------------------
            last_d = d;
            last_minDR = sigmaColor;
            last_gamma = sigmaSpace;


            std::cout
                << "d = "
                << d
                << ", sigma = "
                << sigmaColor
                << " x "
                << sigmaSpace
                << std::endl;
        }


        // ----------------------------------------------------
        // 按键
        // q / ESC退出
        // ----------------------------------------------------
        int key = cv::waitKey(1);

        if (
            key == 'q' ||
            key == 27
            )
        {
            break;
        }
    }


    cv::destroyAllWindows();

    return 0;
}



