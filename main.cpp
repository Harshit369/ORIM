#include <stdio.h>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

//#define DICTIONARY_BUILD 1 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2

int main(int argc, char* argv[])
{	
//#if DICTIONARY_BUILD == 1

	//Step 1 - Obtain the set of bags of features.

	//to store the input file names
	char * filename = new char[1000];

	//to store the current input image
	Mat input;	

	//To store the keypoints that will be extracted by SIFT
	vector<KeyPoint> keypoints;

	//To store the SIFT descriptor of current image
	Mat descriptor;

	//To store all the descriptors that are extracted from all the images.
	Mat allfeaturesUnclustered;

	//The SIFT feature extractor and descriptor
	SiftDescriptorExtractor detector;	
	
	//I select 20 (1000/50) images from 1000 images to extract feature descriptors and build the vocabulary
	for(int f=0;f<=75;f++){		
		//create the file name of an image
		sprintf(filename,"./dataset/training/%i.JPG",f);
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale				
		//detect feature points
		detector.detect(input, keypoints);
		//compute the descriptors for each keypoint
		detector.compute(input, keypoints,descriptor);		
		//put the all feature descriptors in a single Mat object 
		allfeaturesUnclustered.push_back(descriptor);		
		//print the percentage
		printf("%f percent training done\n",f*((float)100/(float)75));
	}	


	
    return 0;
}