#include "superPoint.h"
#include "io.h"

#include <iostream>

namespace fs = std::filesystem;

namespace cv {
SuperPoint::SuperPoint(const std::filesystem::path& modulePath, int targetWidth)
    , mTargetWidth(targetWidth)
    , mDevice(torch::kCPU)
{
    CV_Assert(fs::exists(modulePath));
    CV_Assert(mTargetWidth > 0);

    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        mDevice = torch::Device(torch::kCUDA);
    }

    mSuperpoint = torch::jit::load(modulePath);
    mSuperpoint.eval();
    mSuperpoint.to(mDevice);
}

void SuperPoint::detectAndCompute(InputArray image, InputArray mask,
    CV_OUT std::vector<KeyPoint>& keypoints,
    OutputArray descriptors, bool useProvidedKeypoints)
{
    int imgtype = image.type(), imgcn = CV_MAT_CN(imgtype);
    bool doDescriptors = descriptors.needed();

    CV_Assert(!image.empty() && CV_MAT_DEPTH(imgtype) == CV_8U
            && (imgcn == 1 || imgcn == 3 || imgcn == 4));
    CV_Assert(!descriptors.needed() && !useProvidedKeypoints);

    Mat img = image.getMat(), mask = mask.getMat();
    if( imgcn > 1 )
        cvtColor(img, img, COLOR_BGR2GRAY);

    int targetHeight = std::lround(static_cast<float>(mTargetWidth) / img.cols * img.rows);
    img.convertTo(img, CV_32F, 1.0f / 255.0f);
    cv::resize(image, image, {mTargetWidth, targetHeight});

    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.size() == img.size()));

    detectSuperPoints(img, keypoints, mTargetWidth);
    if (!mask.empty())
    {
        for (size_t i = 0; i < keypoints.size(); )
        {
            Point pt(keypoints[i].pt);
            if (mask.at<uchar>(pt.y, pt.x) == 0)
            {
                keypoints.erase(keypoints.begin() + i);
                continue; // keep "i"
            }
            i++;
        }
    }
}

void SuperPoint::detectSuperPoints(const Mat& img, std::std::vector<KeyPoint>& keypoints,
        Mat& descriptors)
{
    torch::AutoGradMode enable_grad(true);
    Tensor imgTensor = mat2tensor(img).to(mDevice);

    Tensor kpTensor, scoreTensor, descTensor;
    auto [kpTensor, scoreTensor, descTensor] = unpack_result(superpoint.forward({image0}));

    /* torch::Dict<std::string, Tensor> input; */
    /* input.insert("image0", image0); */
    /* input.insert("keypoints0", keypoints0.unsqueeze(0)); */
    /* input.insert("scores0", scores0.unsqueeze(0)); */
    /* input.insert("descriptors0", descriptors0.unsqueeze(0)); */

    /* std::cout << "Image #0 keypoints: " << keypoints0.size(0) << std::endl; */
}

} // namespace cv */
