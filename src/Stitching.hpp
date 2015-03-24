#ifndef STITCHING_HPP_
#define STITCHING_HPP_

#include "FeatureHandler.hpp"

#include <set>

struct image_data
{
	cv::Mat image;
	std::string imagePath;
	std::vector<cv::KeyPoint> imageKeypoints;
	cv::Mat imageDescriptors;
};

struct image_match
{
	image_data image1;
	image_data image2;
	std::vector< cv::DMatch > matches;
	float confidence;
};

struct match_compare
{
	bool operator() (const image_match& lhs, const image_match& rhs) const
	{
		return lhs.confidence > rhs.confidence;
	}
};

class Stitching
{
public:
	Stitching();

	void runImageMatcher();
	std::vector< cv::DMatch > matchImages( image_data image1, image_data image2 );
	std::vector< cv::DMatch > matchImages( cv::Mat image1, cv::Mat image2 );
	void drawImageMatches();

	void addImage( std::string imagePath );

	void setFeatureHandler( FeatureHandler* fh ){_fh=fh;};
	void setRunRatioTest( bool value ){_runRatioTest=value;};
	void setRunSymmetryTest( bool value ){_runSymmetryTest=value;};
	void setDrawMatches( bool value ){_drawMatches=value;};

	~Stitching();

protected:
	bool _runSymmetryTest;
	bool _runRatioTest;
	bool _drawMatches;

	FeatureHandler* _fh;
	std::vector< image_data > _images;
	std::set< image_match, match_compare > _imageMatches;

private:
	image_data processImage( std::string imagePath );
};

#endif
