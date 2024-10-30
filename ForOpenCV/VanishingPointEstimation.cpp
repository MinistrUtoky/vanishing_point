#include "VanishingPointEstimation.h"

cv::Mat VanishingPointEstimation::windowImage;
cv::Mat VanishingPointEstimation::windowImageColored;

cv::Mat VanishingPointEstimation::redrawResized(cv::Mat& img, int width, int height) {
	cv::Mat resizedWindowImage;
	resize(img, resizedWindowImage, cv::Size(width, height)); // otherwise it doesn't fit my screen
	cv::imshow(WIN_NAME, resizedWindowImage);
	cv::waitKey(0);
	return resizedWindowImage;
}

std::vector<cv::Mat> VanishingPointEstimation::getImages(std::string path) {
	std::vector<cv::Mat> newImages = {};
	std::vector<std::string> fn;
	cv::glob(path, fn, false);
	for (std::vector<std::string>::iterator f = fn.begin(); f != fn.end(); ++f) {
		std::cout << f->c_str() << std::endl;
		newImages.push_back(cv::imread(f->c_str(), cv::IMREAD_COLOR));
	}
	return newImages;

}


cv::Mat VanishingPointEstimation::detect(std::vector<cv::Vec4i>& lineSegments) {
	// Detect line segments in the image. 
	// We suggest the application of LineSegmentDetector (Links to an external site.)
	// in OpenCV (see also this (Links to an external site)). (3 points) 
	cv::Mat windowImageRGB;
	cv::cvtColor(windowImage, windowImageRGB, cv::COLOR_GRAY2BGR);

	cv::Ptr<cv::LineSegmentDetector> lineSD = createLineSegmentDetector(cv::LSD_REFINE_ADV, 0.4);
	std::vector<cv::Vec4i> startEndVector;
	lineSD->detect(windowImage, lineSegments);
	lineSD->drawSegments(windowImageRGB, lineSegments);

	return redrawResized(windowImageRGB, 960, 600);
}


cv::Point2d VanishingPointEstimation::intersection(cv::Vec4i segment1, cv::Vec4i segment2) {
	// (y-y1)/(y2-y1) = (x-x1)/(x2-x1)
	// (y-y3)/(y4-y3) = (x-x3)/(x4-x3)
	// (y2−y1)*x + (x1−x2)*y = x1y2 − x2y1
	// (y4−y3)*x + (x3−x4)*y = x3y4 − x4y3
	// y = (x1y2 − x2y1)/(x1 − x2) - (y2−y1)/(x1 − x2)*x
	// x = ((x1y2 − x2y1)*(x3 − x4) - (x3y4 − x4y3)*(x1 − x2)) / ((y2−y1)*(x3 − x4) - (y4−y3)*(x1 − x2)) 
	// y = ((x3y4 − x4y3)*(y2 − y1) - (x1y2 − x2y1)*(y4 − y3)) / ((y2−y1)*(x3 − x4) - (y4−y3)*(x1 − x2)) 
	// ab - cd = |a d|
	//			 |c b|
	// a1 = x3y4 − x4y3; b1 = x1 − x2; c1 = x1y2 − x2y1; d1 = x3 − x4;
	// a2 = y4 − y3; b2 = x1 − x2; c2 = y2 − y1; d2 = x3 − x4;
	double x1 = segment1[0], y1 = segment1[1], x2 = segment1[2], y2 = segment1[3],
		   x3 = segment2[0], y3 = segment2[1], x4 = segment2[2], y4 = segment2[3];
	double dataX[4]{ x1 * y2 - x2 * y1, x1 - x2, x3 * y4 - x4 * y3, x3 - x4 };
	double dataY[4]{ y2 - y1, x1 * y2 - x2 * y1, y4 - y3, x3 * y4 - x4 * y3 };
	double dataQuotient[4]{ y2 - y1, x1 - x2, y4 - y3, x3 - x4 };
	cv::Mat mX(2, 2, CV_64F, dataX);
	cv::Mat mY(2, 2, CV_64F, dataY);
	cv::Mat quotient_m(2, 2, CV_64F, dataQuotient);
	double x = cv::determinant(mX) / cv::determinant(quotient_m);
	double y = cv::determinant(mY) / cv::determinant(quotient_m);
	return cv::Point2d(x, y);
}


std::vector<cv::Point2d> VanishingPointEstimation::selectCandidatePoints(std::vector<cv::Vec4i> segments, int howMany) {
	init_random();
	howMany %= segments.size();
	std::vector<cv::Point2d> candidatePoints;
	for (int i = 0; i < howMany; i++) {
		//Randomly select two line segments, determine the corresponding lines, then calculate their intersecting points.
		//That is a candidate point for the vanishing one. (4 point)
		int firstRandIdx = random_number() * segments.size(),
			secondRandIdx = random_number() * segments.size();
		while (secondRandIdx == firstRandIdx)
			secondRandIdx = random_number() * segments.size();

		cv::Vec4i segment1 = segments.at(firstRandIdx), segment2 = segments.at(secondRandIdx);
		cv::Point2d p = intersection(segment1, segment2);

		//visualizeResults(std::vector<cv::Vec4i>{ segment1, segment2 }, std::vector <cv::Point2d>{ p });
		candidatePoints.push_back(p);
	}
	return candidatePoints;
}

std::vector<cv::Vec4i> VanishingPointEstimation::chooseBestModel(std::vector<cv::Point2d> candidatePoints, std::vector<cv::Vec4i> segments, float sigmaThreshold) {
	/*Determine the distances of candidate points and the lines,
	select the inliers that are closer than a given threshold. (4 points) */
	/*Repeat the steps (random selection, intersection, inlier collection) many times.
	Select the best model with maximal number of inliers. (4 points) */
	std::vector<cv::Vec4i> bestModel;
	int bestCandidatePointIndex = 0;
	float bestInlinerEstimation = 0; // calculated by evaluating each inliner through the length of the segment

	for (int i = 0; i < candidatePoints.size(); i++) {
		cv::Point2d candidatePoint = candidatePoints.at(i);
		std::vector<cv::Vec4i> inlierSegments;
		float inlinerEstimation = 0;
		for (int j = 0; j < segments.size(); j++) {
			cv::Vec4i segment = segments.at(j);
			float a = segment[1] - segment[3], b = segment[2] - segment[0], c = segment[0] * segment[3] - segment[2] * segment[1];
			float distanceIndicator = abs(a * candidatePoint.x + b * candidatePoint.y + c);
			if (distanceIndicator < sigmaThreshold) {
				inlinerEstimation += (segment[2] - segment[0]) * (segment[2] - segment[0]) + (segment[3] - segment[1]) * (segment[3] - segment[1]);
				inlierSegments.push_back(segment);
			}
		}
		if (inlierSegments.size() > bestModel.size()) {
			bestCandidatePointIndex = i;
			bestInlinerEstimation += inlinerEstimation;
			bestModel = inlierSegments;
		}
	}

	//visualizeResults(bestModel, std::vector<cv::Point2d> { candidatePoints.at(bestCandidatePointIndex) });
	return bestModel;
}

cv::Point2d VanishingPointEstimation::reestimateVanishingPoint(std::vector<cv::Vec4i> segments) {
	//Finally, re-estimate the vanishing point based on all the inlier line segments. (8 points) 
	std::vector<cv::Point2d> allVanishingPoints;
	float a1, b1;
	float a2, b2;
	for (int i = 0; i < segments.size(); i++) {
		for (int j = 0; j < segments.size(); j++) {
			if (i != j) {
				a1 = segments.at(i)[1] - segments.at(i)[3]; b1 = segments.at(i)[2] - segments.at(i)[0];
				a2 = segments.at(j)[1] - segments.at(j)[3]; b2 = segments.at(j)[2] - segments.at(j)[0];
				if (abs(a1 * b2 - b1 * a2) > 1) {
					cv::Point2d p = intersection(segments.at(i), segments.at(j));
					allVanishingPoints.push_back(p);
				}
			}
		}
	}
	cv::Point2d averageVanishingPoint(0,0);
	std::cout << averageVanishingPoint << std::endl;
	for (int i = 0; i < allVanishingPoints.size(); i++) {
		averageVanishingPoint += allVanishingPoints.at(i) / (int)allVanishingPoints.size();
	}
	//visualizeResults(std::vector<cv::Vec4i>{}, allVanishingPoints);
	return averageVanishingPoint;
}


cv::Mat VanishingPointEstimation::visualizeResults(std::vector<cv::Vec4i> segments, std::vector<cv::Point2d> vanishingPoints) {
	//Visualize the results: the inliers (green), corresponding lines (red) and the vanishing points (red) should be pictures. 
	// Save the picture (7 points)
	cv::Mat windowImageRGB;
	cv::cvtColor(windowImage, windowImageRGB, cv::COLOR_GRAY2BGR);
	windowImageColored.copyTo(windowImageRGB);

	double minX = 0, minY = 0, maxX = windowImageRGB.cols, maxY = windowImageRGB.rows;
	for (cv::Point2d p : vanishingPoints)
	{
		if (p.y > maxY) maxY = p.y;
		if (p.x > maxX) maxX = p.x;
		if (p.y < minY) minY = p.y;
		if (p.x < minX) minX = p.x;
		std::cout << p << std::endl;
	}
	if (minX < 0)
		minX -= 100;
	if (minY < 0)
		minY -= 100;
	int width = maxX - minX, height = maxY - minY;
	cv::Mat windowImageExtended = cv::Mat::zeros(cv::Size(width, height), windowImageRGB.type());
	for (int i = 0; i < windowImageRGB.size().height; i++) {
		for (int j = 0; j < windowImageRGB.size().width; j++) {
			windowImageExtended.at<cv::Vec3b>(i - (int)minY, j - (int)minX) = windowImageRGB.at<cv::Vec3b>(i, j);
		}
	}
	
	cv::Point2d translation(-minX, -minY);
	int maxDistanceOnScreenSquare = windowImageExtended.rows * windowImageExtended.rows + windowImageExtended.cols * windowImageExtended.cols;
	for (cv::Vec4i segment : segments)
	{
		cv::Point2d beyondScreen = maxDistanceOnScreenSquare * (cv::Point2d(segment[2], segment[3]) - cv::Point2d(segment[0], segment[1]));
		cv::line(windowImageExtended, cv::Point2d(segment[0], segment[1]) + translation, 
										cv::Point2d(segment[2], segment[3]) + translation, cv::Scalar(0, 255, 0), 5);
		cv::line(windowImageExtended, cv::Point2d(segment[0], segment[1]) + beyondScreen + translation, 
										cv::Point2d(segment[0], segment[1]) - beyondScreen + translation, cv::Scalar(0, 0, 255), 1);
	}

	for (cv::Point2d p : vanishingPoints)
	{
		cv::circle(windowImageExtended, p + translation, 5, cv::Scalar(0, 255, 0), 10, cv::FILLED);
	}

	redrawResized(windowImageExtended, windowImageExtended.size().width/2, windowImageExtended.size().height/2);
	return windowImageExtended;
}


 bool VanishingPointEstimation::setWindowImage(std::vector<cv::Mat> images, int selectedImageIndex) {
	windowImageColored = images.at(selectedImageIndex);
	cv::cvtColor(windowImageColored, windowImage, cv::COLOR_BGR2GRAY);
	redrawResized(windowImageColored, 960, 600);
	cv::namedWindow(WIN_NAME, cv::WINDOW_AUTOSIZE);

	if (windowImage.empty())
	{
		std::cout << "Couldn't read image!" << std::endl;
		return false;
	}
	return true;
}