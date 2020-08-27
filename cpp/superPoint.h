#ifndef SUPERGLUE_H
#define SUPERGLUE_H

#include <filesystem>

#include <opencv2/features2d.hpp>

#include <torch/torch.h>
#include <torch/script.h>

namespace cv
{
class SuperPoint : public Feature2D
{
public:
    SuperPoint(const std::filesystem::path& modulePath, int targetWidth);

    void detectAndCompute(InputArray image, InputArray mask,
        CV_OUT std::vector<KeyPoint>& keypoints,
        OutputArray descriptors,
        bool useProvidedKeypoints=false) override;
private:
    void detectSuperPoints(const Mat& img, std::vector<KeyPoint>& keypoints, int targetWidth);

private:
    int mTargetWidth;
    torch::Device mDevice;
    torch::jit::script::Module mSuperpoint;

};

} // namespace cv
#endif // SUPERGLUE_H
