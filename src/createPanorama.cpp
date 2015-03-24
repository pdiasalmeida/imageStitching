#include <cstdlib>
#include <libconfig.h++>

#include "libs/easylogging++.h"
#include "FeatureHandler.hpp"
#include "Stitching.hpp"

void parseConfig( std::string path );
int parseDetector( libconfig::Setting& set );
int parseDescriptor( libconfig::Setting& set );
int parseMatcher( libconfig::Setting& set );
void initLog( std::string path );

FeatureHandler* _fh;
Stitching* _stitcher;

INITIALIZE_EASYLOGGINGPP

int main( int argc, char** argv )
{
	initLog("config/log.conf");
	LOG(INFO) << "'createPanorama' initialized.";
	LOG(INFO) << "Parsing configuration.";
	parseConfig("config/app.conf");

	_stitcher->addImage("./data/samples/stLuzia/resize/IMG_5182m.JPG");
	_stitcher->addImage("./data/samples/stLuzia/resize/IMG_5183m.JPG");
	_stitcher->addImage("./data/samples/stLuzia/resize/IMG_5184m.JPG");
	//_stitcher->addImage("./data/samples/stLuzia/resize/IMG_5185m.JPG");

	_stitcher->runImageMatcher();
	_stitcher->drawImageMatches();

	return EXIT_SUCCESS;
}

void parseConfig( std::string path )
{
	libconfig::Config cfg;
	cfg.readFile(path.c_str());

	//----------------------------------------------- build feature handler -------------------------------------//
	_fh = new FeatureHandler();
	float ratioThresh = 0.85;

	parseDetector(cfg.lookup("application.stitching.featureHandler.detector"));
	parseDescriptor(cfg.lookup("application.stitching.featureHandler.descriptor"));
	parseMatcher(cfg.lookup("application.stitching.featureHandler.matcher"));

	cfg.lookupValue("application.stitching.featureHandler.simmetryTestRatio", ratioThresh);
	_fh->setRatioTestThreshold(ratioThresh);

	//----------------------------------------------- build stitching class -------------------------------------//
	_stitcher = new Stitching();
	bool runRatioTest = false;
	bool runSymmetryTest = false;
	bool drawMatches = false;

	cfg.lookupValue("application.stitching.matching.symmetryTest", runSymmetryTest);
	cfg.lookupValue("application.stitching.matching.ratioTest", runRatioTest);
	cfg.lookupValue("application.stitching.matching.drawMatches", drawMatches);

	_stitcher->setRunRatioTest(runRatioTest);
	_stitcher->setRunSymmetryTest(runSymmetryTest);
	_stitcher->setDrawMatches(drawMatches);

	_stitcher-> setFeatureHandler(_fh);
}

int parseDetector( libconfig::Setting& set )
{
	std::string detector;

	int detID = -1;

	set.lookupValue("name", detector);
	if( detector.compare("sift") == 0 )
	{
		detID = DET_SIFT;
		if( set.exists("params") )
		{
			int nFeatures, nOctaveLayers;
			double threshold, edgeThreshold, sigma;

			set["params"].lookupValue("nFeatures",nFeatures);
			set["params"].lookupValue("nOctaveLayers",nOctaveLayers);
			set["params"].lookupValue("threshold",threshold);
			set["params"].lookupValue("edgeThreshold",edgeThreshold);
			set["params"].lookupValue("sigma",sigma);
			_fh->setFeatureDetectorSIFT(nFeatures, nOctaveLayers, threshold, edgeThreshold, sigma);
		}
	}
	else if( detector.compare("surf") == 0 )
	{
		detID = DET_SURF;
		_fh->setFeatureDetectorSURF();
	}
	else if( detector.compare("fast") == 0 )
	{
		detID = DET_FAST;
		_fh->setFeatureDetectorFAST();
	}
	else if( detector.compare("gftt") == 0 )
	{
		detID = DET_GFTT;
		_fh->setFeatureDetectorGFTT();
	}

	return detID;
}

int parseDescriptor( libconfig::Setting& set )
{
	std::string descriptor;
	int descID = -1;

	set.lookupValue("name", descriptor);
	if( descriptor.compare("sift") == 0 )
	{
		descID = DES_SIFT;
		if( set.exists("params") )
		{
			int nFeatures, nOctaveLayers;
			double threshold, edgeThreshold, sigma;

			set["params"].lookupValue("nFeatures",nFeatures);
			set["params"].lookupValue("nOctaveLayers",nOctaveLayers);
			set["params"].lookupValue("threshold",threshold);
			set["params"].lookupValue("edgeThreshold",edgeThreshold);
			set["params"].lookupValue("sigma",sigma);
			_fh->setDescriptorExtractorSIFT(nFeatures, nOctaveLayers, threshold, edgeThreshold, sigma);
		}
	}
	else if( descriptor.compare("surf") == 0 )
	{
		descID = DES_SURF;
		_fh->setDescriptorExtractorSURF();
	}
	else if( descriptor.compare("brief") == 0 )
	{
		descID = DES_BRIEF;
		_fh->setDescriptorExtractorBRIEF();
	}
	else if( descriptor.compare("orb") == 0 )
	{
		descID = DES_ORB;
		_fh->setDescriptorExtractorORB();
	}
	else if( descriptor.compare("freak") == 0 )
	{
		descID = DES_FRE;
		_fh->setDescriptorExtractorFREAK();
	}

	return descID;
}

int parseMatcher( libconfig::Setting& set )
{
	std::string matcher;
	int matchID = -1;

	set.lookupValue("name", matcher);
	if( matcher.compare("BFL1") == 0 )
	{
		matchID = DESM_BF1;
		_fh->setDescriptorMatcherBF(cv::NORM_L1);
	}
	else if( matcher.compare("BFL2") == 0 )
	{
		matchID = DESM_BF2;
		_fh->setDescriptorMatcherBF(cv::NORM_L2);
	}
	else if( matcher.compare("BFH1") == 0 )
	{
		matchID = DESM_BFH1;
		_fh->setDescriptorMatcherBF(cv::NORM_HAMMING);
	}
	else if( matcher.compare("BFH2") == 0 )
	{
		matchID = DESM_BFH2;
		_fh->setDescriptorMatcherBF(cv::NORM_HAMMING2);
	}
	else if( matcher.compare("flannBase") == 0 )
	{
		matchID = DESM_FB;
		_fh->setDescriptorMatcherFB();
	}

	return matchID;
}

void initLog( std::string path )
{
	el::Configurations logConf(path);
	el::Loggers::reconfigureAllLoggers(logConf);
}
