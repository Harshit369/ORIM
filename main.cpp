#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

//#define DICTIONARY_BUILD 1 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2
void readme()
{ 
	cout << " Usage: ./SURF_descriptor <scene_image_path>" << std::endl; 
}


int main(int argc, char* argv[])
{	
	if( argc != 2 ) { 
		readme(); 
		return -1;
	}

	Mat scene_descriptor;
	Mat img_scene = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	if( !img_scene.data ) {
		cout<< " --(!) Error reading images " << std::endl; return -1; 
	}		

	//------------------------------------------------

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

	//detect feature points for scene
	detector.detect(input, keypoints);
	//compute the descriptors for each keypoint for scene
	detector.compute(img_scene, keypoints,scene_descriptor);

	//create a flann based matcher and a match vector to match descriptors and store them
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	vector<int> match_count;
	
	//I select 20 (1000/50) images from 1000 images to extract feature descriptors and build the vocabulary
	for(int f=1;f<=75;f++){		
		//create the file name of an image
		sprintf(filename,"./dataset/training/%i.JPG",f);
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale				
		//detect feature points
		detector.detect(input, keypoints);
		//compute the descriptors for each keypoint
		detector.compute(input, keypoints,descriptor);

		//put the all feature descriptors in a single Mat object 
		/*cv::FileStorage fs("descriptors.yml", cv::FileStorage::WRITE);
		fs << "descriptors" << descriptor;
		fs << "keypoints" << keypoints;
		fs.release();*/

		allfeaturesUnclustered.push_back(descriptor);

		//matcher.match( descriptor, scene_descriptor, matches );
		int n = (int)matches.size();
  		cout<<n;

		//print the percentage
		printf("%f percent training done\n",f*((float)100/(float)75));
	}	

	//int descriptor_power_set = allfeaturesUnclustered.size();

	for(int j=0;j < 75;j++){

		//match descriptors
  		
  		//match_count.push_back(matches.size());

  		//double max_dist = 0; double min_dist = 100;
	}

	
    return 0;
}

