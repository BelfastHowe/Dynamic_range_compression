#include "dpdc.h"

ushort find_med_3(ushort x1, ushort x2, ushort x3) {
    ushort max = 0;
    ushort min = 0;
    ushort med = 0;

    if (x1 >= x2 && x2 >= x3) {
        max = x1;
        min = x3;
        med = x2;
    }
    else if (x3 >= x2 && x2 >= x1) {
        max = x3;
        min = x1;
        med = x2;
    }

    if (x3 >= x1 && x1 >= x2) {
        max = x3;
        min = x2;
        med = x1;
    }
    else if (x2 >= x1 && x1 >= x3) {
        max = x2;
        min = x3;
        med = x1;
    }

    if (x1 >= x3 && x3 >= x2) {
        max = x1;
        min = x2;
        med = x3;
    }
    else if (x2 >= x3 && x3 >= x1) {
        max = x2;
        min = x1;
        med = x3;
    }

    return med;
}

void order(ushort* nums, int len)
{
    ushort temp = 0;
    for (int i = 0; i < len - 1; i++) {
        for (int j = 0; j < len - 1 - i; j++) {
            if (nums[j] > nums[j + 1]) {
                temp = nums[j];
                nums[j] = nums[j + 1];
                nums[j + 1] = temp;
            }
        }
    }
}

void ours_dpc(cv::Mat img, bool dpc_en, cv::Mat img_dpc, double dp_slope, double dp_thresh, int diff_th, int DEAD[][2], int dead_count)
{

    // 获取图像尺寸
    int width = img.cols;
    int height = img.rows;

    // 创建校正后的图像
    cv::Mat img_correct = cv::Mat::zeros(height, width, CV_16UC1);
    int padding_size1 = 1;
    int padding_size2 = 2;
    // 图像边缘扩展
    int extended_height = height + 2 * padding_size1;
    int extended_width = width + 2 * padding_size2;
    cv::Mat imgn = cv::Mat::zeros(extended_height, extended_width, CV_16UC1);
    cv::copyMakeBorder(img, imgn, padding_size1, padding_size1, padding_size2, padding_size2, cv::BORDER_REFLECT_101);

    if (dpc_en) {
        for (int i = padding_size1; i < height + padding_size1; i++) {
            for (int j = padding_size2; j < width + padding_size2; j++) {

                std::vector<ushort> img_r_values;
                for (int m = -padding_size1; m <= padding_size1; m++) {
                    for (int n = -padding_size2; n <= padding_size2; n++) {
                        img_r_values.push_back(imgn.at<ushort>(i + m, j + n));

                    }
                }

                // 中心像素值
                ushort data_r_center = imgn.at<ushort>(i, j);

                // 计算与中心像素的差值绝对值
                std::vector<double> data_r_diff(img_r_values.size());
                for (size_t k = 0; k < img_r_values.size(); k++) {
                    data_r_diff[k] = abs(static_cast<double>(img_r_values[k]) - data_r_center);
                }
                // 判断所有 signed differences 是否同号
                bool all_same_sign = true;
                if (!data_r_diff.empty()) {
                    bool all_positive = true;
                    bool all_negative = true;
                    for (size_t k = 0; k < img_r_values.size(); k++) {
                        if (data_r_diff[k] < 0) all_positive = false;
                        if (data_r_diff[k] > 0) all_negative = false;
                    }
                    all_same_sign = all_positive || all_negative;
                }
                ushort ori1[5];
                ori1[0] = img_r_values[0];
                ori1[1] = img_r_values[1];
                ori1[2] = img_r_values[2];
                ori1[3] = img_r_values[3];
                ori1[4] = img_r_values[4];
                ushort ori2[5];
                ori2[0] = img_r_values[5];
                ori2[1] = img_r_values[6];
                ori2[2] = img_r_values[7];
                ori2[3] = img_r_values[8];
                ori2[4] = img_r_values[9];
                ushort ori3[5];
                ori3[0] = img_r_values[10];
                ori3[1] = img_r_values[11];
                ori3[2] = img_r_values[12];
                ori3[3] = img_r_values[13];
                ori3[4] = img_r_values[14];
                order(ori1, 5);
                ushort median1 = ori1[2];

                order(ori2, 5);
                ushort median2 = ori2[2];

                order(ori3, 5);
                ushort median3 = ori3[2];


                ushort data_r_median = find_med_3(median1, median2, median3);/*avg_on_min_gradient_3x3(img_r_values[1], img_r_values[2], img_r_values[3], img_r_values[6], img_r_values[8], img_r_values[11], img_r_values[12], img_r_values[13]);*/

                // 计算检测阈值
                std::vector<double> data_r_detect(img_r_values.size());
                int data_r_judge = 0;
                double detect_sum = 0.0;

                for (size_t k = 0; k < img_r_values.size(); k++) {
                    data_r_detect[k] = data_r_diff[k] * dp_slope - dp_thresh;
                    if (data_r_detect[k] < 0) data_r_detect[k] = 0;
                    if (data_r_detect[k] > 1024) data_r_detect[k] = 1024;
                    if (data_r_detect[k] > 0) {
                        data_r_judge++;
                        detect_sum += data_r_detect[k];
                    }
                }

                int data_r_weight;
                if (data_r_judge > 0) {
                    int ratio = detect_sum / data_r_judge;
                    data_r_weight = ratio;
                }
                else {
                    data_r_weight = 0;
                    /*for (size_t k = 0; k < img_r_values.size(); k++) {
                        cout << data_r_diff[k]<<" ";
                    }
                    cout << "\n";
                    for (size_t k = 0; k < img_r_values.size(); k++) {
                        cout << data_r_detect[k] << " ";
                    }*/
                }

                int data_r_correct;

                // 判断当前像素是否在 DEAD 列表中（如果 dead_count <=0 或 DEAD==nullptr 则认为没有坏点）
                bool is_dead_pixel = false;
                if (DEAD != nullptr && dead_count > 0) {
                    for (int k = 0; k < dead_count; ++k) {
                        if ((i - padding_size1) == DEAD[k][0] && (j - padding_size2) == DEAD[k][1]) {
                            is_dead_pixel = true;
                            std::cout << DEAD[k][0] << " " << DEAD[k][1] << std::endl;
                            break;
                        }
                    }
                }

                // 根据判断条件进行校正
                // 新增：如果邻域所有 signed differences 同号（全部正或全部负），也视为坏点（用中值修正）
                if (data_r_judge >= 7 && abs(data_r_center - data_r_median) > diff_th/* && all_same_sign*/) {
                    data_r_correct = data_r_median * data_r_weight + (1024 - data_r_weight) * data_r_center;
                    data_r_correct /= 1024;
                }
                else if (is_dead_pixel) {
                    data_r_correct = data_r_median;
                }
                else {
                    data_r_correct = data_r_center;
                }

                img_dpc.at<ushort>(i - padding_size1, j - padding_size2) = static_cast<ushort>(data_r_correct);
            }
        }
    }
    else {
        img.copyTo(img_dpc);
    }
}


