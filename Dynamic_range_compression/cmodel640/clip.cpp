#include "clip.h"



// 像素值比较函数
bool comparePixel14b(const Pixel14b& a, const Pixel14b& b) {
    return a.value < b.value;
}

// 查找14bit图像中最小像素
void findMinPixels14b(const cv::Mat& img_14bit, int n, std::vector<cv::Point>& positions, std::vector<uint16_t>& values)
{
    std::vector<Pixel14b> pixels;
    pixels.reserve(img_14bit.rows * img_14bit.cols);

    // 遍历14bit图像收集数据
    for (int y = 0; y < img_14bit.rows; ++y) {
        const uint16_t* row = img_14bit.ptr<uint16_t>(y);
        for (int x = 0; x < img_14bit.cols; ++x) {
            pixels.emplace_back(row[x], x, y);
        }
    }

    // 按像素值排序
    sort(pixels.begin(), pixels.end(), comparePixel14b);

    // 提取前n个
    int count = std::min(n, (int)pixels.size());
    positions.clear();
    values.clear();
    for (int i = 0; i < count; ++i) {
        positions.push_back(cv::Point(pixels[i].x, pixels[i].y));
        values.push_back(pixels[i].value);
    }
}

cv::Mat histogram_equalization_clipped_fixed2(const cv::Mat & image_14bit,
    std::vector<cv::Point>& min_positions,
    std::vector<uint16_t>& min_values, bool clip_en, int low_number, int high_number,
    int& autothreshold_low, int& autothreshold_high)
{
    const int width = image_14bit.cols;
    const int height = image_14bit.rows;
    constexpr int bin_width = 8;
    constexpr int bin_count = 16384 / bin_width;

    findMinPixels14b(image_14bit, 16, min_positions, min_values);

    int hist[bin_count] = { 0 };
    for (int y = 0; y < height; ++y) {
        const uint16_t* row = image_14bit.ptr<uint16_t>(y);
        for (int x = 0; x < width; ++x) {
            uint16_t pixel = row[x];
            if (pixel < 16384) {
                hist[pixel / bin_width]++;
            }
        }
    }

    int cdf[bin_count] = { 0 };
    cdf[0] = hist[0];
    for (int i = 1; i < bin_count; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    autothreshold_low = 0;
    autothreshold_high = 16383;

    for (int i = 0; i < bin_count; ++i) {
        if (i < bin_count - 1 && cdf[i] <= low_number && cdf[i + 1] > low_number) {
            autothreshold_low = i * bin_width + bin_width / 2;
        }
        else if (i > 0 && cdf[i] >= high_number && cdf[i - 1] < high_number) {
            autothreshold_high = i * bin_width + bin_width / 2 - 1;
        }
    }

    cv::Mat thresholded_14bit = image_14bit.clone();
    if (clip_en) {
        for (int y = 0; y < height; ++y) {
            uint16_t* row = thresholded_14bit.ptr<uint16_t>(y);
            for (int x = 0; x < width; ++x) {
                if (row[x] < autothreshold_low) {
                    row[x] = static_cast<uint16_t>(autothreshold_low);
                }
                else if (row[x] > autothreshold_high) {
                    row[x] = static_cast<uint16_t>(autothreshold_high);
                }
            }
        }
    }

    return thresholded_14bit;
}

