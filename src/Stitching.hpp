#ifndef STITCHING_HPP_
#define STITCHING_HPP_

#include "FeatureHandler.hpp"

class Stitching
{
public:
	Stitching();

	std::vector< cv::DMatch > matchImages( cv::Mat image1, cv::Mat image2 );

	void setFeatureHandler(FeatureHandler* fh){_fh=fh;};
	void setRunRatioTest(bool value){_runRatioTest=value;};
	void setRunSymmetryTest(bool value){_runSymmetryTest=value;};
	void setDrawMatches(bool value){_drawMatches=value;};

	~Stitching();

protected:
	bool _runSymmetryTest;
	bool _runRatioTest;
	bool _drawMatches;

	FeatureHandler* _fh;
};

#endif
