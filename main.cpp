#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

#define DICTIONARY_BUILD 1 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2

int main(int argc, char* argv[])
{	
#if DICTIONARY_BUILD == 1

	//Step 1 - Obtain the set of bags of features.

	//to store the input file names
	char * filenameplane = new char[100];		
	char * filenamebike = new char[100];		
	//to store the current input image
	Mat inputplane,inputbike;	

	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypointsplane,keypointsbike;
	//To store the SIFT descriptor of current image
	Mat descriptorplane,descriptorbike;
	//To store all the descriptors that are extracted from all the images.
	Mat allfeaturesUnclustered;
	//The SIFT feature extractor and descriptor
	SiftDescriptorExtractor detectorplane,detectorbike;	
	
	//I select 75 training images 
	for(int f=100;f<=800;f+=10){		
		//create the file name of an image
		sprintf(filenameplane,"./dataset/training/airplanes_side/0%i.jpg",f);
		sprintf(filenamebike,"./dataset/training/motorbikes_side/0%i.jpg",f);
		//open the file
		inputplane = imread(filenameplane, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale				
		inputbike = imread(filenamebike, CV_LOAD_IMAGE_GRAYSCALE);
		//detect feature points
		detectorplane.detect(inputplane, keypointsplane);
		detectorbike.detect(inputbike, keypointsbike);
		//compute the descriptors for each keypoint
		detectorplane.compute(inputplane, keypointsplane,descriptorplane);
		detectorbike.compute(inputbike, keypointsbike,descriptorbike);		
		//put the all feature descriptors in a single Mat object 

		allfeaturesUnclustered.push_back(descriptorplane);		
		allfeaturesUnclustered.push_back(descriptorbike);		
		//print the percentage
		printf("%i percent done\n",f/10);
	}	


	//Construct BOWKMeansTrainer
	//the number of bags
	int dictionarySize=200;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=1;
	//necessary flags
	int flags=KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);


	//cluster the feature vectors for all training set
	Mat dictionaryplane=bowTrainer.cluster(allfeaturesUnclustered);	
	//store the vocabulary
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "descriptors" << dictionary;
	fs.release();

#else
	//Step 2 - Obtain the BoF descriptor for given image/video frame. 

    //prepare BOW descriptor extractor from the dictionary    
	Mat dictionary; 
	FileStorage fs("dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();	
    
	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);	
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor,matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	//To store the image file name
	char * filename = new char[100];
	//To store the image tag name - only for save the descriptor in a file
	char * imageTag = new char[10];

	//open the file to write the resultant descriptor
	FileStorage fs1("descriptor.yml", FileStorage::WRITE);	
	
	//the image file with the location. 
	sprintf(filename,"./128.JPG");		
	//read the image
	Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);		
	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypoints;		
	//Detect SIFT keypoints (or feature points)
	detector->detect(img,keypoints);
	//To store the BoW (or BoF) representation of the image
	Mat bowDescriptor;		
	//extract BoW (or BoF) descriptor from given image
	bowDE.compute(img,keypoints,bowDescriptor);

	//prepare the yml (some what similar to xml) file
	sprintf(imageTag,"img1");			
	//write the new BoF descriptor to the file
	fs1 << imageTag << bowDescriptor;
	//release the file storage
	fs1.release();	

	//You may use this descriptor for classifying the image.
	Mat image_yml_descriptor; 
	FileStorage fs2("descriptor.yml", FileStorage::READ);
	fs2["img1"] >> image_yml_descriptor;
	fs2.release();
	imshow( "Display Frame", image_yml_descriptor );
#endif
	printf("\ndone\n");	
    return 0;
}