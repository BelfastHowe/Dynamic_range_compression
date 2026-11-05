#include "adc.h"


cv::Mat vector2mat(std::vector<uint8_t>output, cv::Size2i size)
{
	cv::Mat out_result(size.height, size.width, CV_8UC1);
	memcpy(out_result.data, output.data(), output.size() * sizeof(uchar));
	return out_result;
}

std::vector<uint16_t> mat2vector(cv::Mat img)
{
	//img /= 4;
	std::vector<uint16_t> v = img.reshape(1, 1);
	return v;
}

void adc(cv::Mat img_in, cv::Mat img_out, bool adc_en, int Thr, int delta2)
{
	int H = img_in.rows;
	int W = img_in.cols;

	std::vector<uint16_t> img = mat2vector(img_in);
	std::vector<uint8_t> imgdata;


	std::vector<uint16_t> imgl(H * W);
	std::vector<int64_t> imgc(H * W);
	std::vector<int16_t> imgdata5(H * W);

	std::vector<int16_t> img5(H * W);

	uint32_t up;
	uint32_t down;


	for (int k = 0; k < img.size() / (H * W); k++)
	{
		imgl.assign(img.begin() + k * H * W, img.begin() + k * H * W + H * W);
		imgc.assign(img.begin() + k * H * W, img.begin() + k * H * W + H * W);

		int n = imgl.size();

		up = 0;
		down = 16383;
		for (int i = 0; i < W; i++) {
			for (int j = 0; j < H; j++) {

				if (up < imgl[i * H + j])
					up = imgl[i * H + j];
				if (down > imgl[i * H + j])
					down = imgl[i * H + j];

			}
		}
		int max = up;
		int min = down;
		int sub = max - min;
		//cout << sub<<" " <<min<<" "<<max << endl;
		int delta1 = Thr - sub;
		if (delta1 < 0) {
			delta1 = 0;
		}
		int delta = delta1 + delta2;
		int max_new = max + delta;
		int min_new = min - delta;
		int sub_new = max_new - min_new;
		sub_new = sub_new > 0 ? sub_new : 1;
		std::cout << "adc new range:" << sub_new << " " << min_new << " " << max_new << std::endl;

		for (int i = 0; i < W; i++) {
			for (int j = 0; j < H; j++) {
				img5[i * H + j] = (((((imgc[i * H + j]) - min_new) << 24) / sub_new) * 255) >> 24;
			}
		}


		for (int i = 0; i < W; i++) {
			for (int j = 0; j < H; j++) {
				if (img5[i * H + j] > 255)
					imgdata5[i * H + j] = 255;
				//imgdata5[i * H + j] = 0;
				else if (img5[i * H + j] < 0)
					imgdata5[i * H + j] = 0;
				//imgdata5[i * H + j] = 255;
				else
					//imgdata5[i * H + j] = 255 - img5[i * H + j];
					imgdata5[i * H + j] = img5[i * H + j];
				imgdata.push_back(imgdata5[i * H + j]);
			}
		}

	}

	if (adc_en) {
		cv::Mat out_result = vector2mat(imgdata, cv::Size2i(W, H));
		out_result.copyTo(img_out);

	}
	else {
		img_in /= 256;
		img_in.convertTo(img_out, CV_8UC1);
	}
}






