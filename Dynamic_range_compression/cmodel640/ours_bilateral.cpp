#include "ours_bilateral.h"


void ours_bilateral(cv::Mat img_in, cv::Mat img_dn, bool dn_en, int sigma_color, int sigma_space)
{
    int d = 7;
    int height = img_in.rows;
    int width = img_in.cols;

    int r = d >> 1;
    cv::Mat img_padding;
    int padding_size = r;
    copyMakeBorder(img_in, img_padding, padding_size, padding_size, padding_size, padding_size, cv::BORDER_REFLECT_101);
    /*    for (int i = 285; i < 292; i++) {
            for (int j = 1; j < 8; j++) {
                printf("%hx ", img_padding.at<ushort>(i, j));
            }
            printf("\n");
        }*/

    int weight1_index = 0;

    for (int row_d = -r; row_d <= r; row_d++) {
        for (int col_d = -r; col_d <= r; col_d++) {
            double distance_Square = row_d * row_d + col_d * col_d;
            double weight1 = exp(-1 * (distance_Square / (2 * sigma_space * sigma_space)));
            long long weight1_d = weight1 * (1 << 14);//14bit
            if (weight1_d >= (1 << 14)) {
                weight1_d = (1 << 14) - 1;
            }
            std::cout << weight1_d << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int diff = 0; diff < 16384; diff++) {
        long long value_Square = pow(diff, 2);
        double weight2 = exp(-1 * double(value_Square) / (2 * sigma_color * sigma_color));

        long long weight2_d = weight2 * (1 << 16);//16bit
        if (weight2_d >= (1 << 16)) {
            weight2_d = (1 << 16) - 1;
        }
        if (weight2_d > 0) {
            std::cout << diff << ":" << weight2_d << std::endl;
        }
    }
    std::cout << std::endl;


    if (dn_en) {
        for (int i = 0; i < height; i++)    //对每一个点进行处理
        {
            for (int j = 0; j < width; j++)
            {
                double weightSum = 0;
                double filterValue = 0;
                for (int row_d = -r; row_d <= r; row_d++)   //以图像中的一点为中心，r为半径的方形区域内进行计算
                {
                    double weightline = 0;
                    double valueline = 0;
                    for (int col_d = -r; col_d <= r; col_d++)
                    {

                        double distance_Square = row_d * row_d + col_d * col_d;
                        double weight1 = exp(-1 * (distance_Square / (2 * sigma_space * sigma_space)));
                        long long weight1_d = weight1 * (1 << 14);//14bit
                        if (weight1_d >= (1 << 14)) {
                            weight1_d = (1 << 14) - 1;
                        }

                        ushort difference = abs(img_padding.at<ushort>(i + r + row_d, j + r + col_d) - img_in.at<ushort>(i, j));

                        long long value_Square = pow(difference, 2);
                        double weight2 = exp(-1 * double(value_Square) / (2 * sigma_color * sigma_color));

                        long long weight2_d = weight2 * (1 << 16);//16bit
                        if (weight2_d >= (1 << 16)) {
                            weight2_d = (1 << 16) - 1;
                        }



                        double weight = (weight1_d * weight2_d) >> 14;//16*14 >>14
                        double pixel_value = weight * img_padding.at<ushort>(i + r + row_d, j + r + col_d);//14*16
                        weightSum += weight;               //求滤波窗口内的权重和，用于归一化
                        filterValue += (weight * img_padding.at<ushort>(i + r + row_d, j + r + col_d));

                        weightline += weight;
                        valueline += pixel_value;
                        /*if (i == 0 && j == 44) {

                            cout << weight1_d <<"  " << weight2_d << "  " << weight << "  " << pixel_value << endl;

                        }*/
                    }/*if (i == 0 && j == 44) {
                        cout <<"weightline:"<< weightline << endl;
                        cout << "valueline:" << valueline << endl;
                    }*/
                }
                /*if (i == 0 && j == 44) {
                    cout<<weightSum<<endl;
                    cout << filterValue << endl;
                }*/
                img_dn.at<ushort>(i, j) = filterValue / weightSum;
                int out = long long(filterValue) / long long(weightSum);

            }
        }
    }
    else {
        img_in.copyTo(img_dn);
    }

}


