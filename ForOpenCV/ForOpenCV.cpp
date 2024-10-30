#include "VanishingPointEstimation.h"

int main() {
	std::string path = "T:\\ELTECar_images\\*.png";
	std::vector<cv::Mat> images = VanishingPointEstimation::getImages(path);
	for (int i = 0; i < images.size(); i++) {
		VanishingPointEstimation::setWindowImage(images, i);

		std::vector<cv::Vec4i> lineSegments;
		cv::Mat detected = VanishingPointEstimation::detect(lineSegments);

		cv::imwrite("detected" + std::to_string(i) + ".jpg", detected);

		std::vector<cv::Point2d> candidatePoints = VanishingPointEstimation::selectCandidatePoints(lineSegments, 5000);

		std::vector<cv::Vec4i> bestModelSegments = VanishingPointEstimation::chooseBestModel(candidatePoints, lineSegments, 300);

		cv::Point2d bestModelAverageVanishingPoint = VanishingPointEstimation::reestimateVanishingPoint(bestModelSegments);

		cv::Mat finalResult = VanishingPointEstimation::visualizeResults(bestModelSegments, std::vector<cv::Point2d> {bestModelAverageVanishingPoint});

		cv::imwrite("estimated" + std::to_string(i) + ".jpg", finalResult);
	}
	cv::waitKey(0);
	return 0;
}
