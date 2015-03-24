#ifndef FEATUREHANDLER_HPP_
#define FEATUREHANDLER_HPP_

#include <opencv2/features2d/features2d.hpp>

enum Detector{ DET_SIFT, DET_SURF, DET_FAST, DET_GFTT };
enum DescriptorExtractor{ DES_SIFT, DES_SURF, DES_BRIEF, DES_ORB, DES_FRE };
enum DescriptorMatcher{ DESM_BF1, DESM_BF2, DESM_BFH1, DESM_BFH2, DESM_FB };

class FeatureHandler
{
public:
	FeatureHandler();
	FeatureHandler( Detector det, DescriptorExtractor des, DescriptorMatcher match, float ratioTestThresh );

	std::vector< cv::KeyPoint > detectKeypoints( cv::Mat image );
	cv::Mat computeDescriptors(cv::Mat image, std::vector< cv::KeyPoint > keypoints);
	std::vector< cv::DMatch > getImageMatches(cv::Mat descriptors1, cv::Mat descriptors2);
	std::vector< std::vector< cv::DMatch > > getImageKnnMatches(cv::Mat descriptors1, cv::Mat descriptors2, int k);
	std::vector< cv::DMatch > ratioTest( std::vector< std::vector< cv::DMatch > > knnMatches );
	std::vector< cv::DMatch > symmetryTest( std::vector< cv::DMatch > img1Toimg2,
			std::vector< cv::DMatch > img2Toimg1 );

	int getDetectorID();
	int getDescriptorID();
	int getMatcherID();

	void setRatioTestThreshold(float ratioTestThreshold){_ratioTestThreshold=ratioTestThreshold;};

	void setFeatureDetectorSIFT( int nFeatures=0, int nOctaveLayers=3, double threshold=0.04,
			double edgeThreshold=10.0, double sigma=1.6 );
	void setFeatureDetectorSURF( double hessianThreshold=400, int octaves=3,
			int octaveLayers=4, bool extended=true, bool upright=true );
	void setFeatureDetectorFAST( int threshold=20, bool nmaxSupre=true );
	void setFeatureDetectorGFTT( int maxCorners=1000, double qualityLevel=0.01, double minDistance=1,
            int blockSize=3, bool useHarrisDetector=false, double k=0.04 );

	void setDescriptorExtractorSIFT( int nFeatures=0, int nOctaveLayers=3, double threshold=0.04,
				double edgeThreshold=10.0, double sigma=1.6 );
	void setDescriptorExtractorSURF( double hessianThreshold=400, int octaves=3,
				int octaveLayers=4, bool extended=true, bool upright=true );
	void setDescriptorExtractorBRIEF( int bytes=32 );
	void setDescriptorExtractorORB( int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31,
			int firstLevel=0, int WTA_K=2, int scoreType=cv::ORB::HARRIS_SCORE, int patchSize=31 );
	void setDescriptorExtractorFREAK( bool orientationNormalized=true, bool scaleNormalized=true,
			float patternScale=22.0f, int nOctaves=4, const std::vector<int>& selectedPairs=std::vector<int>() );

	void setDescriptorMatcherBF( int normType=cv::NORM_L2, bool crossCheck=false );
	void setDescriptorMatcherFB( const cv::Ptr<cv::flann::IndexParams>& indexParams=new cv::flann::KDTreeIndexParams(),
	        const cv::Ptr<cv::flann::SearchParams>& searchParams=new cv::flann::SearchParams() );

	~FeatureHandler();

protected:
	cv::FeatureDetector* _fDetector;
	cv::DescriptorExtractor* _fDescriptor;
	cv::DescriptorMatcher* _fMatcher;

	float _ratioTestThreshold;

	std::string _fDetectorName;
	std::string _fDescriptorName;
	std::string _fMatcherName;

	int _fDetectorID;
	int _fDescriptorID;
	int _fMatcherID;

private:
	void init( Detector det, DescriptorExtractor des, DescriptorMatcher match );
	void setFeatureDetector(Detector detectorID);
	void setDescriptorExtractor(DescriptorExtractor extractorID);
	void setDescriptorMatcher( DescriptorMatcher matcherID );
};

#endif
