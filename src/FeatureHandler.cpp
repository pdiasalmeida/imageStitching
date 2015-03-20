#include "FeatureHandler.hpp"

#include "libs/easylogging++.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

FeatureHandler::FeatureHandler()
{
	_fDetector = NULL;
	_fDescriptor = NULL;
	_fMatcher = NULL;

	_fDetectorID = -1;
	_fDescriptorID = -1;
	_fMatcherID = -1;

	_ratioTestThreshold = -1;
}

FeatureHandler::FeatureHandler( Detector det, DescriptorExtractor des, DescriptorMatcher match, float ratioTestThresh )
{
	init( (Detector)det, (DescriptorExtractor)des, (DescriptorMatcher)match);

	_ratioTestThreshold = ratioTestThresh;
}

void FeatureHandler::init( Detector det, DescriptorExtractor des, DescriptorMatcher match )
{
	if( ((des == DES_BRIEF) || (des == DES_ORB) || (des == DES_FRE)) &&
				(match == DESM_FB) )
	{
		LOG(WARNING) << "Invalid combination of descriptors and matcher.";
		LOG(WARNING) << "FlannBased Descriptor Matcher is compatible with float based descriptors only";
		LOG(INFO) << "Setting default matcher";
		match = DESM_BFH2;
	}

	_fDetectorID = det;
	_fDescriptorID = des;
	_fMatcherID = match;

	setFeatureDetector(det);
	setDescriptorExtractor(des);
	setDescriptorMatcher(match);
}

void FeatureHandler::setFeatureDetector(Detector detectorID)
{
	switch(detectorID)
	{
		case 0:
			setFeatureDetectorSIFT(); break;
		case 1:
			setFeatureDetectorSURF(); break;
		case 2:
			setFeatureDetectorFAST(); break;
		case 3:
			setFeatureDetectorGFTT(); break;
		default:
			LOG(WARNING) << "Invalid feature detector ID, setting default.";
			setFeatureDetectorSIFT();
			break;
	}
}

void FeatureHandler::setDescriptorExtractor(DescriptorExtractor extractorID)
{
	switch(extractorID)
	{
		case 0:
			setDescriptorExtractorSIFT(); break;
		case 1:
			setDescriptorExtractorSURF(); break;
		case 2:
			setDescriptorExtractorBRIEF(); break;
		case 3:
			setDescriptorExtractorORB(); break;
		case 4:
			setDescriptorExtractorFREAK(); break;
		default:
			LOG(WARNING) << "Invalid descriptor extractor ID, setting default.";
			setDescriptorExtractorSIFT();
			break;
	}
}

void FeatureHandler::setDescriptorMatcher( DescriptorMatcher matcherID )
{
	switch(matcherID)
	{
	case 0:
		setDescriptorMatcherBF(cv::NORM_L1); break;
	case 1:
		setDescriptorMatcherBF(cv::NORM_L2); break;
	case 2:
		setDescriptorMatcherBF(cv::NORM_HAMMING); break;
	case 3:
		setDescriptorMatcherBF(cv::NORM_HAMMING2); break;
	case 4:
		setDescriptorMatcherFB(); break;
	default:
		LOG(WARNING) << "Invalid descriptor matcher ID, setting default.";
		setDescriptorMatcherBF(cv::NORM_L2);
		break;
	}
}

void FeatureHandler::setFeatureDetectorSIFT( int nFeatures, int nOctaveLayers, double threshold,
		double edgeThreshold, double sigma )
{
	_fDetector = new cv::SiftFeatureDetector( nFeatures, nOctaveLayers, threshold, edgeThreshold, sigma );
	_fDetectorName = "SIFT Feature Detector";

	LOG(INFO) << "Feature detector set to SIFT: [" << nFeatures << "; " << nOctaveLayers << "; " <<
		threshold << "; " << edgeThreshold << "; " << sigma << "]";
}

void FeatureHandler::setFeatureDetectorSURF( double hessianThreshold, int octaves, int octaveLayers,
		bool extended, bool upright )
{
	_fDetector = new cv::SurfFeatureDetector(hessianThreshold, octaves, octaveLayers);
	_fDetectorName = "SURF Feature Detector";

	LOG(INFO) << "Feature detector set to SURF: [" << hessianThreshold << "; " << octaves <<
			"; " << octaveLayers << "]";
}

void FeatureHandler::setFeatureDetectorFAST( int threshold, bool nmaxSupre )
{
	_fDetector = new cv::FastFeatureDetector( threshold, nmaxSupre );
	_fDetectorName = "FAST Feature Detector";

	std::string nmaxS = nmaxSupre ? "true" : "false";
	LOG(INFO) << "Feature detector set to FAST: [" << threshold << "; " << nmaxS  << "]";
}

void FeatureHandler::setFeatureDetectorGFTT( int maxCorners, double qualityLevel, double minDistance,
		int blockSize, bool useHarrisDetector, double k )
{
	_fDetector = new cv::GoodFeaturesToTrackDetector( maxCorners, qualityLevel, minDistance, blockSize,
			useHarrisDetector, k );
	_fDetectorName = "GoodFeaturesToTrack Feature Detector";

	LOG(INFO) << "Feature detector set to GoodFeaturesToTrack: [" << maxCorners << ";" << qualityLevel <<
			"; " << minDistance << "; " << blockSize << "; " << useHarrisDetector << "; " << k << "]";
}

void FeatureHandler::setDescriptorExtractorSIFT( int nFeatures, int nOctaveLayers, double threshold,
		double edgeThreshold, double sigma )
{
	_fDescriptor = new cv::SiftDescriptorExtractor;
	_fDescriptorName = "SIFT Descriptor Extractor";

	LOG(INFO) << "Descriptor Extractor set to SIFT: [" << nFeatures << "; " << nOctaveLayers << "; " <<
		threshold << "; " << edgeThreshold << "; " << sigma << "]";
}

void FeatureHandler::setDescriptorExtractorSURF( double hessianThreshold, int octaves,
		int octaveLayers, bool extended, bool upright )
{
	_fDescriptor = new cv::SurfDescriptorExtractor;
	_fDescriptorName = "SURF Descriptor Extractor";

	LOG(INFO) << "Descriptor Extractor set to SURF: [" << hessianThreshold << "; " << octaves <<
			"; " << octaveLayers << "]";
}

void FeatureHandler::setDescriptorExtractorBRIEF(int bytes)
{
	_fDescriptor = new cv::BriefDescriptorExtractor( bytes );
	_fDescriptorName = "BRIEF Descriptor Extractor";

	LOG(INFO) << "Descriptor Extractor set to BRIEF: [" << bytes <<  "]";
}

void FeatureHandler::setDescriptorExtractorORB(int nfeatures, float scaleFactor, int nlevels,
		int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize)
{
	_fDescriptor = new cv::OrbDescriptorExtractor( nfeatures, scaleFactor, nlevels, edgeThreshold,
			firstLevel, WTA_K, scoreType, patchSize );
	_fDescriptorName = "ORB Descriptor Extractor";

	LOG(INFO) << "Descriptor Extractor set to ORB: [" << nfeatures << "; " << scaleFactor << "; " <<
			nlevels << "; " << edgeThreshold << "; " << firstLevel << "; " << WTA_K << "; " <<
			scoreType << "; " << patchSize <<  "]";
}

void FeatureHandler::setDescriptorExtractorFREAK( bool orientationNormalized, bool scaleNormalized,
		float patternScale, int nOctaves, const std::vector<int>& selectedPairs )
{
	_fDescriptor = new cv::FREAK( orientationNormalized, scaleNormalized, patternScale, nOctaves,
			selectedPairs );
	_fDescriptorName = "FREAK Descriptor Extractor";

	LOG(INFO) << "Descriptor Extractor set to FREAK: [" << orientationNormalized << "; " << scaleNormalized <<
			"; " << patternScale << "; " << nOctaves <<  "]";
}

void FeatureHandler::setDescriptorMatcherBF( int normType, bool crossCheck )
{
	_fMatcher = new cv::BFMatcher( normType, crossCheck );
	_fMatcherName = "BruteForce Descriptor Matcher";

	LOG(INFO) << "Descriptor Matcher set to BruteForce: [" << normType << "; " << crossCheck <<  "]";
}

void FeatureHandler::setDescriptorMatcherFB( const cv::Ptr<cv::flann::IndexParams>& indexParams,
        const cv::Ptr<cv::flann::SearchParams>& searchParams )
{
	_fMatcher = new cv::FlannBasedMatcher( indexParams, searchParams );
	_fMatcherName = "FlannBased Descriptor Matcher";

	LOG(INFO) << "Descriptor Matcher set to FlannBased";
}

int FeatureHandler::getDetectorID()
{
	return _fDetectorID;
}

int FeatureHandler::getDescriptorID()
{
	return _fDescriptorID;
}

int FeatureHandler::getMatcherID()
{
	return _fMatcherID;
}

FeatureHandler::~FeatureHandler()
{
	delete _fDetector;
	delete _fDescriptor;
	delete _fMatcher;
}
