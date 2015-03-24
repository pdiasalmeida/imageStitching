#include "Stitching.hpp"

#include <opencv2/highgui/highgui.hpp>

Stitching::Stitching()
{
	_runSymmetryTest = false;
	_runRatioTest = false;
	_drawMatches = false;

	_fh = NULL;
}

std::vector< cv::DMatch > Stitching::matchImages( cv::Mat image1, cv::Mat image2 )
{
	std::vector< cv::KeyPoint > keypoints1 = _fh->detectKeypoints(image1);
	std::vector< cv::KeyPoint > keypoints2 = _fh->detectKeypoints(image2);

	cv::Mat descriptor1 = _fh->computeDescriptors( image1, keypoints1 );
	cv::Mat descriptor2 = _fh->computeDescriptors( image2, keypoints2 );

	std::vector< cv::DMatch > matches;
	if( _runRatioTest )
	{
		std::vector< std::vector< cv::DMatch > > kMatches = _fh->getImageKnnMatches( descriptor1, descriptor2, 2 );
		matches = _fh->ratioTest(kMatches);
	}
	if( _runSymmetryTest )
	{
		if(_runRatioTest)
		{
			std::vector< cv::DMatch > matches2To1 = _fh->getImageMatches( descriptor2, descriptor1 );
			matches = _fh->symmetryTest( matches, matches2To1 );
		} else{
			std::vector< cv::DMatch > matches1To2 = _fh->getImageMatches( descriptor1, descriptor2 );
			std::vector< cv::DMatch > matches2To1 = _fh->getImageMatches( descriptor2, descriptor1 );
			matches = _fh->symmetryTest( matches1To2, matches2To1 );
		}
	}
	if( !_runRatioTest && !_runRatioTest) matches = _fh->getImageMatches( descriptor1, descriptor2 );

	if(_drawMatches)
	{
		cv::Mat imOut;
		cv::drawMatches( image1, keypoints1, image2, keypoints2, matches, imOut );

		cv::namedWindow("matches", CV_WINDOW_NORMAL);
		cv::imshow("matches", imOut);

		cv::waitKey(0);
	}

	return matches;
}

Stitching::~Stitching(){}
