#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <random>
#include <cmath>

#define WIN_NAME "VANISHING POINTS ESTIMATION BY RANSAC"
static std::mt19937 rng;
static std::uniform_real_distribution<float> udist(0.f, 1.f);
static std::normal_distribution<float> ndist(0.f, 5.f);

static void init_random()
{
    rng.seed(static_cast<long unsigned int>(time(0)));
}

static float random_number()
{
    return udist(rng);
}

static float random_noise()
{
    return ndist(rng);
}

class VanishingPointEstimation
{
private:
	static float thresholdDistance;
	static cv::Mat windowImage;
	static cv::Mat windowImageColored;

	static cv::Mat redrawResized(cv::Mat& img, int width, int height);
	static cv::Point2d intersection(cv::Vec4i segment1, cv::Vec4i segment2);
public:
	static std::vector<cv::Mat> getImages(std::string path);
	static bool setWindowImage(std::vector<cv::Mat> images, int selectedImageIndex);
	static cv::Mat detect(std::vector<cv::Vec4i>& lineSegments);
	static std::vector<cv::Point2d> selectCandidatePoints(std::vector<cv::Vec4i> segments, int howMany);
	static std::vector<cv::Vec4i> chooseBestModel(std::vector<cv::Point2d> candidatePoints, std::vector<cv::Vec4i> segments,
										float sigmaThreshold);
	static cv::Point2d reestimateVanishingPoint(std::vector<cv::Vec4i> segments);
	static cv::Mat visualizeResults(std::vector<cv::Vec4i> segments, std::vector<cv::Point2d> vanishingPoints);
	static cv::Point2d reestimateVanishingPointErrorBased(std::vector<cv::Vec4i> segments);
	static cv::Point2d reestimateVanishingPointErrorBasedFast(std::vector<cv::Vec4i> segments) {
	};
};