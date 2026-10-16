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

// 选择适合坐标轴的刻度间隔，例如 1、2、5、10、20、50...
static int64_t chooseNiceStep(double raw)
{
    if (raw <= 1.0)
        return 1;

    const double power =
        std::pow(10.0, std::floor(std::log10(raw)));

    const double normalized = raw / power;

    double scale;
    if (normalized <= 1.0)
        scale = 1.0;
    else if (normalized <= 2.0)
        scale = 2.0;
    else if (normalized <= 5.0)
        scale = 5.0;
    else
        scale = 10.0;

    return std::max<int64_t>(
        1, static_cast<int64_t>(std::ceil(scale * power)));
}

// 将半整数索引转换为坐标标签：
// 0 -> "0", 1 -> "0.5", 2 -> "1", 3 -> "1.5" ...
static std::string formatHalfStep(int64_t index)
{
    std::ostringstream oss;

    if (index % 2 == 0)
    {
        oss << index / 2;
    }
    else
    {
        oss << std::fixed << std::setprecision(1)
            << static_cast<double>(index) * 0.5;
    }

    return oss.str();
}

// 显示以 0.5 为精确步长的直方图。
//
// xMin / xMax：显示的横轴数值范围。
//              传入负数表示自动使用数据范围。
// plotWidth / plotHeight：绘图区的默认宽度和高度。
//
// 注意：本函数要求输入为非负、有限且以 0.5 为步长的数值。
int showHalfStepHistogram(const std::vector<float>& values, const std::string& windowName, double xMin, double xMax, int plotWidth, int plotHeight)
{
    CV_Assert(!values.empty());
    CV_Assert(plotWidth > 0 && plotHeight > 0);

    // ========================================================
    // 1. 将所有数值精确映射到整数索引
    //
    // index = round(value * 2)
    //
    // 例如：
    // value = 0.0  -> index = 0
    // value = 0.5  -> index = 1
    // value = 1.0  -> index = 2
    // value = 1.5  -> index = 3
    // ========================================================

    std::vector<int> indices;
    indices.reserve(values.size());

    int minIndex = INT_MAX;
    int maxIndex = 0;

    for (float v : values)
    {
        CV_Assert(std::isfinite(v));
        CV_Assert(v >= 0.0f);

        const double scaled = static_cast<double>(v) * 2.0;
        const int64_t index = std::llround(scaled);

        // 检查输入是否确实以 0.5 为步长。
        CV_Assert(std::abs(scaled - static_cast<double>(index))
            < 1e-3);

        CV_Assert(index >= 0 && index < INT_MAX);

        indices.push_back(static_cast<int>(index));

        minIndex = std::min(minIndex, static_cast<int>(index));
        maxIndex = std::max(maxIndex, static_cast<int>(index));
    }

    std::cout << "hh_coefficients range: " << formatHalfStep(minIndex) << " ~ " << formatHalfStep(maxIndex) << std::endl;

    // ========================================================
    // 2. 统计每个精确取值出现的次数
    //
    // hist[k] 表示数值 k * 0.5 出现的次数。
    // 不合并相邻区间。
    // ========================================================

    std::vector<uint64_t> hist(
        static_cast<size_t>(maxIndex) + 1, 0);

    for (int index : indices)
    {
        ++hist[index];
    }

    // ========================================================
    // 3. 求中位数，用于图中标记
    // ========================================================

    std::vector<int> medianIndices = indices;

    const size_t mid = medianIndices.size() / 2;

    std::nth_element(
        medianIndices.begin(),
        medianIndices.begin() + mid,
        medianIndices.end());

    const int medianIndex = medianIndices[mid];

    // ========================================================
    // 4. 确定横轴显示范围
    //
    // 将横轴范围转换成 0.5 步长的整数索引。
    // ========================================================

    int64_t xStart;
    int64_t xEnd;

    if (xMin < 0.0)
        xStart = minIndex;
    else
        xStart = static_cast<int64_t>(
            std::ceil(xMin * 2.0 - 1e-9));

    if (xMax < 0.0)
        xEnd = maxIndex;
    else
        xEnd = static_cast<int64_t>(
            std::floor(xMax * 2.0 + 1e-9));

    CV_Assert(xStart >= 0);
    CV_Assert(xEnd >= xStart);

    // 只有一个柱时，稍微扩展坐标范围，避免图像退化。
    if (xStart == xEnd)
    {
        if (xStart > 0)
            --xStart;

        ++xEnd;
    }

    const int64_t numBins = xEnd - xStart + 1;

    // 防止误传极大的绘图范围造成内存或计算问题。
    CV_Assert(numBins <= 1000000);

    // 每个半步长取值至少分配一个横向像素，
    // 14-bit 数据的 HH 范围通常不超过 32767 个半步长索引。
    if (numBins <= 32767)
    {
        plotWidth = std::max(
            plotWidth, static_cast<int>(numBins));
    }
    else if (numBins > plotWidth)
    {
        std::cerr
            << "Warning: X range contains more bins than pixels. "
            << "Use xMin/xMax to zoom in for detailed inspection."
            << std::endl;
    }

    auto countAt = [&](int64_t index) -> uint64_t
        {
            if (index < 0 ||
                index >= static_cast<int64_t>(hist.size()))
            {
                return 0;
            }

            return hist[static_cast<size_t>(index)];
        };

    uint64_t maxCount = 0;

    for (int64_t i = xStart; i <= xEnd; ++i)
    {
        maxCount = std::max(maxCount, countAt(i));
    }

    // 避免空显示区间造成 Y 轴为零。
    if (maxCount == 0)
        maxCount = 1;

    const int64_t yStep =
        chooseNiceStep(std::ceil(maxCount / 5.0));

    const uint64_t yAxisMax =
        ((maxCount + yStep - 1) / yStep) * yStep;  //向上取整到 B 的倍数

    // ========================================================
   // 5. 创建绘图画布
   // ========================================================

    const int leftMargin = 110;
    const int rightMargin = 35;
    const int topMargin = 70;
    const int bottomMargin = 105;

    const int plotX0 = leftMargin;
    const int plotY0 = topMargin;
    const int plotX1 = plotX0 + plotWidth;
    const int plotY1 = plotY0 + plotHeight;

    cv::Mat canvas(
        topMargin + plotHeight + bottomMargin,
        leftMargin + plotWidth + rightMargin,
        CV_8UC3,
        cv::Scalar(255, 255, 255));

    const cv::Scalar axisColor(30, 30, 30);
    const cv::Scalar gridColor(225, 225, 225);
    const cv::Scalar barColor(210, 140, 70);  // BGR
    const cv::Scalar medianColor(40, 50, 220);

    const int font = cv::FONT_HERSHEY_SIMPLEX;

    // 标题
    std::ostringstream title;

    title << "HH |abs| Histogram"
        << "   N = " << values.size()
        << "   Bin step = 0.5"
        << "   Median = "
        << formatHalfStep(medianIndex);

    cv::putText(
        canvas,
        title.str(),
        cv::Point(plotX0, 35),
        font,
        0.65,
        axisColor,
        1,
        cv::LINE_AA);

    // ========================================================
    // 6. 绘制 Y 轴网格、刻度和标签
    // ========================================================

    for (uint64_t count = 0;
        count <= yAxisMax;)
    {
        const int y = plotY1 -
            cvRound(
                static_cast<double>(count) /
                static_cast<double>(yAxisMax) *
                plotHeight);

        cv::line(
            canvas,
            cv::Point(plotX0, y),
            cv::Point(plotX1, y),
            gridColor,
            1,
            cv::LINE_AA);

        const std::string label = std::to_string(count);

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            label, font, 0.45, 1, &baseline);

        cv::putText(
            canvas,
            label,
            cv::Point(
                plotX0 - 12 - textSize.width,
                y + textSize.height / 2),
            font,
            0.45,
            axisColor,
            1,
            cv::LINE_AA);

        if (yAxisMax - count < static_cast<uint64_t>(yStep))
            break;

        count += static_cast<uint64_t>(yStep);
    }

    // ========================================================
    // 7. 绘制 X 轴网格和刻度
    // ========================================================

    const int64_t xStep =
        chooseNiceStep(static_cast<double>(numBins) / 8.0);

    int64_t tick =
        ((xStart + xStep - 1) / xStep) * xStep;

    for (; tick <= xEnd;)
    {
        const int x = plotX0 + cvRound(
            (static_cast<double>(tick - xStart) + 0.5) /
            static_cast<double>(numBins) *
            plotWidth);

        cv::line(
            canvas,
            cv::Point(x, plotY0),
            cv::Point(x, plotY1),
            gridColor,
            1,
            cv::LINE_AA);

        const std::string label = formatHalfStep(tick);

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            label, font, 0.45, 1, &baseline);

        cv::putText(
            canvas,
            label,
            cv::Point(
                x - textSize.width / 2,
                plotY1 + 27),
            font,
            0.45,
            axisColor,
            1,
            cv::LINE_AA);

        if (xEnd - tick < xStep)
            break;

        tick += xStep;
    }

    // ========================================================
   // 8. 绘制直方图柱
   // ========================================================

    for (int64_t i = xStart; i <= xEnd; ++i)
    {
        const uint64_t count = countAt(i);

        if (count == 0)
            continue;

        const double binLeft =
            static_cast<double>(i - xStart) / numBins;

        const double binRight =
            static_cast<double>(i - xStart + 1) / numBins;

        const int x0 =
            plotX0 + static_cast<int>(
                std::floor(binLeft * plotWidth));

        int x1 =
            plotX0 + static_cast<int>(
                std::ceil(binRight * plotWidth)) - 1;

        x1 = std::max(x0, x1);

        const int yTop = plotY1 - cvRound(
            static_cast<double>(count) /
            static_cast<double>(yAxisMax) *
            plotHeight);

        cv::rectangle(
            canvas,
            cv::Point(x0, plotY1 - 1),
            cv::Point(x1, std::max(plotY0, yTop)),
            barColor,
            cv::FILLED);
    }

    // ========================================================
    // 9. 标记中位数位置
    // ========================================================

    if (medianIndex >= xStart && medianIndex <= xEnd)
    {
        const int medianX = plotX0 + cvRound(
            (static_cast<double>(medianIndex - xStart) + 0.5) /
            static_cast<double>(numBins) *
            plotWidth);

        // 红色虚线
        for (int y = plotY0; y < plotY1; y += 14)
        {
            cv::line(
                canvas,
                cv::Point(medianX, y),
                cv::Point(medianX, std::min(y + 7, plotY1)),
                medianColor,
                2,
                cv::LINE_AA);
        }

        cv::putText(
            canvas,
            "Median",
            cv::Point(medianX + 5, plotY0 + 20),
            font,
            0.45,
            medianColor,
            1,
            cv::LINE_AA);
    }

    // ========================================================
    // 10. 绘制坐标轴和轴名称
    // ========================================================

    cv::line(
        canvas,
        cv::Point(plotX0, plotY0),
        cv::Point(plotX0, plotY1),
        axisColor, 2);

    cv::line(
        canvas,
        cv::Point(plotX0, plotY1),
        cv::Point(plotX1, plotY1),
        axisColor, 2);

    // Y 轴标题使用竖排文字
    cv::Mat yLabelH(
        32, 100, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::putText(
        yLabelH,
        "Count",
        cv::Point(3, 23),
        font,
        0.55,
        axisColor,
        1,
        cv::LINE_AA);

    cv::Mat yLabelV;
    cv::rotate(
        yLabelH,
        yLabelV,
        cv::ROTATE_90_COUNTERCLOCKWISE);

    const int yLabelX = 5;
    const int yLabelY =
        plotY0 + (plotHeight - yLabelV.rows) / 2;

    yLabelV.copyTo(
        canvas(cv::Rect(
            yLabelX, yLabelY,
            yLabelV.cols, yLabelV.rows)));

    // X 轴标题
    const std::string xLabel =
        "abs(HH) coefficient (step = 0.5)";

    int baseline = 0;
    const cv::Size xTextSize = cv::getTextSize(
        xLabel, font, 0.6, 1, &baseline);

    cv::putText(
        canvas,
        xLabel,
        cv::Point(
            plotX0 + (plotWidth - xTextSize.width) / 2,
            plotY1 + 75),
        font,
        0.6,
        axisColor,
        1,
        cv::LINE_AA);

    // 显示
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, canvas);

	imwrite_mdy_private(canvas, "HH_histogram.png");
    return 0;
}



