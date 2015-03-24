#include "Stitching.hpp"

#include "libs/easylogging++.h"

#include <opencv2/highgui/highgui.hpp>

Stitching::Stitching()
{
	_runSymmetryTest = false;
	_runRatioTest = false;
	_drawMatches = false;

	_fh = NULL;
}

void Stitching::runImageMatcher()
{
	std::vector< image_data >::iterator it = _images.begin();
	for( ; it != _images.end(); it++ )
	{
		std::vector< image_data >::iterator itAux = it+1;
		for( ; itAux != _images.end(); itAux++ )
		{
			if( !(it->imagePath.compare(itAux->imagePath) == 0) )
			{
				std::vector< cv::DMatch > matches;
				matches = matchImages( *it, *itAux );

				image_match m = image_match();
				m.image1 = *it;
				m.image2 = *itAux;
				m.matches = matches;

				m.confidence = (float) matches.size() /
						(float)std::min(it->imageKeypoints.size(), itAux->imageKeypoints.size());

				_imageMatches.insert(m);
			}
		}
	}
}

void Stitching::drawImageMatches()
{
	std::set< image_match>::iterator it = _imageMatches.begin();
	for( ; it != _imageMatches.end(); it++ )
	{
		cv::Mat imOut;
		cv::drawMatches( it->image1.image, it->image1.imageKeypoints,
				it->image2.image, it->image2.imageKeypoints, it->matches, imOut );

		std::stringstream ss;
		ss << "Matches between images '" << it->image1.imagePath << "' and '" << it->image2.imagePath
				<< "'. Confidence: " << it->confidence;

		cv::namedWindow(ss.str(), CV_WINDOW_NORMAL);
		cv::imshow(ss.str(), imOut);

		cv::waitKey(0);
	}
}

std::vector< cv::DMatch > Stitching::matchImages( image_data image1, image_data image2 )
{
	std::vector< cv::DMatch > matches;
	if( _runRatioTest )
	{
		std::vector< std::vector< cv::DMatch > > kMatches =
				_fh->getImageKnnMatches( image1.imageDescriptors, image2.imageDescriptors, 2 );
		matches = _fh->ratioTest(kMatches);
	}
	if( _runSymmetryTest )
	{
		if(_runRatioTest)
		{
			std::vector< cv::DMatch > matches2To1 =
					_fh->getImageMatches( image2.imageDescriptors, image1.imageDescriptors );
			matches = _fh->symmetryTest( matches, matches2To1 );
		} else{
			std::vector< cv::DMatch > matches1To2 =
					_fh->getImageMatches( image1.imageDescriptors, image2.imageDescriptors );
			std::vector< cv::DMatch > matches2To1 =
					_fh->getImageMatches( image2.imageDescriptors, image1.imageDescriptors );
			matches = _fh->symmetryTest( matches1To2, matches2To1 );
		}
	}
	if( !_runRatioTest && !_runRatioTest)
		matches = _fh->getImageMatches( image1.imageDescriptors, image2.imageDescriptors );

	return matches;
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

void Stitching::addImage( std::string imagePath )
{
	image_data imData = processImage( imagePath );
	_images.push_back( imData );
}

image_data Stitching::processImage( std::string imagePath )
{
	cv::Mat image = cv::imread( imagePath, cv::IMREAD_GRAYSCALE );
	std::vector< cv::KeyPoint > keypoints = _fh->detectKeypoints(image);
	cv::Mat descriptors = _fh->computeDescriptors(image, keypoints);

	image_data data = image_data();
	data.image = image;
	data.imagePath = imagePath;
	data.imageKeypoints = keypoints;
	data.imageDescriptors = descriptors;

	return data;
}

Stitching::~Stitching(){}
