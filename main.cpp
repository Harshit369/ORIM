#include <stdio.h>
#include <map>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/legacy/legacy.hpp>

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
	Mat descriptorplane,descriptorbike,descrplane,descrbike;
	//To store all the descriptors that are extracted from all the images.
	Mat allfeaturesUnclustered;
	//The SIFT feature extractor and descriptor
	SiftFeatureDetector detectorplane,detectorbike;
	SiftDescriptorExtractor extractorplane,extractorbike;	
	
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
		extractorplane.compute(inputplane, keypointsplane,descriptorplane);
		extractorbike.compute(inputbike, keypointsbike,descriptorbike);		
		//put the all feature descriptors in a single Mat object 

		allfeaturesUnclustered.push_back(descriptorplane);		
		allfeaturesUnclustered.push_back(descriptorbike);		
		//print the percentage
		printf("%f percent training done\n",((float)f/(float)8));
	}	

	printf("-----------Training complete------\n\n");
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
	Mat dictionarywrite = bowTrainer.cluster(allfeaturesUnclustered);	
	printf("-------------clusters created-------------------\n\n");
	//store the vocabulary
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "descriptors" << dictionarywrite;
	fs.release();
	printf("-----------Dictionary of descriptors created---------------\n\n");


	//Step 2 - Obtain the BoF descriptor for given image and train svm according to it. 

    //prepare BOW descriptor extractor from the dictionary    
	Mat dictionary; 
	FileStorage fs1("dictionary.yml", FileStorage::READ);
	fs1["vocabulary"] >> dictionary;
	fs1.release();	


	vector< KeyPoint > keypointsp,keypointsb;
	Mat response_histplane;
	Mat response_hist;
	Mat imgplane,imgbike;
	map<string,Mat> classes_training_data;
	vector< string > classes_names;
	//To store the image file name
	//char *filenameplane = new char[100];
    
	//create a nearest neighbor matcher
	//Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12, 20, 2));
	//Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<uchar> >());
	//FlannBasedMatcher matcher;
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);	
	//create BoF (or BoW) descriptor extractor
	//BOWImgDescriptorExtractor bowide(extractor,matcher);
	Ptr<BOWImgDescriptorExtractor> bowide(new BOWImgDescriptorExtractor(extractor,matcher));
	//Set the dictionary with the vocabulary we created in the first step
	bowide->setVocabulary(dictionary);


	printf("----------vocabulary loaded--------------\n\n");

	int total_samples=0;
	// adds histograam data to the hash table corresponding to plane and bike class
	for(int f=100;f<=800;f+=10) {

		
   		sprintf(filenameplane,"./dataset/training/motorplane_side/0%i.jpg",f);
   		imgplane = imread(filenameplane,CV_LOAD_IMAGE_GRAYSCALE);
   		detector->detect(imgplane,keypointsp);
   		extractor->compute(imgplane,keypointsp,response_histplane);
   		bowide->compute(imgplane, keypointsp, response_histplane);
    	if(classes_training_data.count("plane") == 0) { //not yet created...
        	classes_training_data["plane"].create(0,response_histplane.cols,response_histplane.type());
        	classes_names.push_back("plane");
      	}
      	classes_training_data["plane"].push_back(response_histplane);
   		total_samples++;
	}
	//cout<<total_samples<<endl;
	printf("-----------plane histograms hash created----------\n\n");

	/*for(int f=100;f<=800;f+=10) {

		sprintf(filenamebike,"./dataset/training/motorbikes_side/0%i.jpg",f);
		
   		imgbike = imread(filenamebike,CV_LOAD_IMAGE_GRAYSCALE);

   		detector->detect(imgbike,keypointsb);
   		extractor->compute(imgbike,keypointsb,response_histbike);
   		bowide->compute(imgbike, keypointsb, response_histbike);

    	if(classes_training_data.count("bike") == 0) { //not yet created...
        	classes_training_data["bike"].create(0,response_histbike.cols,response_histbike.type());
        	classes_names.push_back("bike");
      	}
      	classes_training_data["bike"].push_back(response_histbike);
   		//total_samples++;
	}
	printf("-----------bike histograms hash created----------\n\n");*/
	//------------------------------

      	cout<<classes_training_data.size()<<endl;
      	//cout<<classes_training_data["plane"].count()<<endl;

	for (int i=0;i<classes_names.size();i++) {
   		string class_ = classes_names[i];
   		cout << " training class: " << class_ << ".." << endl;
         
   		Mat samples(0,response_histplane.cols,response_histplane.type());
   		Mat labels(0,1,CV_32FC1);
         
  		//copy class samples and label
   		cout << "adding " << classes_training_data[class_].rows << " positive" << endl;
   		samples.push_back(classes_training_data[class_]);
   		Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
   		labels.push_back(class_label);
         
   		//copy rest samples and label
   		for (map<string,Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
      		string not_class_ = (*it1).first;
      		if(not_class_.compare(class_)==0) continue; //skip class itself
      		samples.push_back(classes_training_data[not_class_]);
      		class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32FC1);
      		labels.push_back(class_label);
   		}
    
   		cout << "Trained.." << endl;
   		Mat samples_32f; 
   		samples.convertTo(samples_32f, CV_32F);
   		if(samples.rows == 0) continue; //phantom class?!
   		CvSVM classifier; 
   		classifier.train(samples_32f,labels);
 
	}

	//---------------------------------------------------------------------------
	Mat resposne_histscene;
	map<string,CvSVM> classes_classifiers; //This we created earlier
 
	//vector<string> file; //load up with images
	//vector<string> class1; //load up with the respective classes
 
   	Mat img = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
   	
    
   	vector<KeyPoint> keypoints;
   	detector->detect(img,keypoints);
   	extractor->compute(img,keypoints,response_hist);
   	bowide->compute(img, keypoints, response_hist);
 
   	float minf = FLT_MAX; string minclass;
   	for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
    	float res = (*it).second.predict(response_hist,true);
    	if (res < minf) {
         	minf = res;
         	minclass = (*it).first;
    	}
    	cout<<res;
   	}
   	//confusion_matrix[minclass][classes[]]++; */ 
	

#endif
	printf("\ndone\n");	
    return 0;
}