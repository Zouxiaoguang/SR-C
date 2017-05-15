#include <iostream> 
#include <string>   
#include <iomanip> 
#include <fstream>
#include <sstream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp> 

using namespace cv;
using namespace std;

ofstream lr_example("lr_example.txt");
ofstream input_example("input_example.txt");
ofstream Weight("GSsd.txt");


void img2patch( Mat img,Mat &patches,Mat &pi,Mat &pj);
double getPSNR(const Mat& I1,const Mat& I2);
Mat IBP(const Mat &ImageGT,const Mat &SR,const Mat &LR,int iters);
double GSSD(const Mat &lr_patch,const Mat &lr_example);

int main()
{
  Mat LR=imread("1.png",0);
  if(!LR.data)
  {
      cout<<"NO Image Data~"<<endl; 
	  return false;
  }
 
  //Gaussian blur;
  Mat src;
  GaussianBlur( LR, src, Size( 3, 3 ), 0, 0 );

  //downsample the src img;
  int N = src.rows;
  int M = src.cols;
  int scale=2;
  float gradualScale=1.25;
  int Mid=3;
  Mat level_img[7];
  //char img_name[10];
  for(int i=0;i!=Mid;++i)
   {
	int diff=i-Mid;
	float down=pow(gradualScale,diff);
	int down_rows=int(N*down);
	int down_cols=int(M*down); 
	//level_img[i]=Mat::zeros(down_rows,down_cols,LR.type());
    resize(src, level_img[i], Size(down_cols, down_rows),(0,0),(0,0),2);
    //sprintf(img_name, "%s%d%s", "downimg", i, ".jpg");
	//imwrite(img_name,level_img[i]);
   }

  level_img[Mid]=LR;

  //initial the upscale image
  for(int j=Mid+1;j!=7;j++)
  {
    int diff=j-Mid;
	float up=pow(gradualScale,diff);
	int up_rows=int(N*up);
	int up_cols=int(M*up); 
    level_img[j]=Mat::zeros(up_rows, up_cols, LR.type());
  }

  int patchSize=5;
  //change the input image to patches
  Mat *patches_p,pi,pj;
  int inputPatches=(level_img[Mid].size().width-patchSize+1)*(level_img[Mid].size().height-patchSize+1);
  patches_p =new Mat(inputPatches, patchSize*patchSize, CV_32F);
  pi=Mat::zeros(inputPatches, 1, CV_8U);
  pj=Mat::zeros(inputPatches, 1, CV_8U);
  img2patch(level_img[Mid],*patches_p,pi,pj);

  //write the patches txt
 /* for(int i=0;i!=(*patches_p).rows;++i)
  {
     input_example<<i<<"  "<<(*patches_p).row(i)<<endl;
  }*/

  //runtime
  clock_t start,end;
  double dur;
  start = clock();
  int K=1;
  //change the downimg to patch
  for(int i=Mid-1;i!=-1;i--)
 {
   int downPatches=(level_img[i].size().width-patchSize+1)*(level_img[i].size().height-patchSize+1);

   Mat *patches_q,qi,qj;
   //downscaling image patch index
   patches_q =new Mat(downPatches, patchSize*patchSize, CV_32F);
   //patch center coordinate index
   qi=Mat::zeros(downPatches, 1, CV_8U);
   qj=Mat::zeros(downPatches, 1, CV_8U);

   img2patch(level_img[i],*patches_q,qi,qj);
   //cout<<(*patches_q).row(0)<<endl;
  /* if(i==2)
   {
     //write the patches txt
     for(int r=0;r!=(*patches_q).rows;++r)
     {
       lr_example<<r<<"  "<<(*patches_q).row(r)<<endl;
     }
   }*/

   
   flann::Index* tree;
   tree = new flann::Index(*patches_q, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);

  
	for(int j=0;j!=(*patches_p).rows;++j)
	{
	  Mat ind;
	 /*Eculidean distances of nearest patches*/
	 /* NOTE!: DISTANCE IS EUCLIDEAN DISTANCE SQUARED! */
	  Mat dist;

	 /*search the tree for the K nearest patches of 'patch'*/
	  tree->knnSearch((*patches_p).row(j), ind, dist, K);
	  //double* scaleVec = new double[3];
	  for (int k = 0; k< K; k++)
	  {
		/* for each K-nearest patches*/
		 double d = (double)dist.at<float>(0, k);
		 int index = ind.at<int>(0, k);
		 double weight=GSSD((*patches_p).row(j),(*patches_q).row(index));
		 Weight<<weight<<endl;
	  }
	   
	}

	

  }
   end=clock();
   dur=double(end-start);
   cout<<dur/CLOCKS_PER_SEC<<endl;

  

  //CvKNearest knn();
  system("pause");
  waitKey(0);
  return 0;
}

void img2patch( Mat img,Mat &patch,Mat &pi,Mat &pj)
{
   //Mat patch(1, patchSize*patchSize, CV_8U);

   int patchIndex=0;
   for(int x=2;x!=img.size().height-2;x++)
   {
      for(int y=2;y!=img.size().width-2;y++)
	  {
	     int pixelIndex=0;
		 for(int i=-2;i!=3;i++)
		 {
		    for(int j=-2;j!=3;j++)
			{
			  int sampleIndexX=x+i;
	          int sampleIndexY=y+j;
			  float interSample=(float)img.at<uchar>(sampleIndexX,sampleIndexY);
			  //patch.at<uchar>(0,pixelIndex)=interSample;
			  patch.at<float>(patchIndex,pixelIndex)=interSample;
			  pixelIndex++;
			}
		 }
		 pi.at<uchar>(patchIndex,0)=x;
		 pj.at<uchar>(patchIndex,0)=y;
		 patchIndex++;
	  }
   }
}

double getPSNR(const Mat& I1,const Mat& I2)
{
  Mat s1;
  absdiff(I1,I2,s1);
  s1.convertTo(s1,CV_32F);
  s1=s1.mul(s1);

  Scalar s=sum(s1);

  double sse=s.val[0]+s.val[1]+s.val[2];
  if(sse<1e-10)
     return 0;
  else
  {
     double mse=sse/ (double)(I1.channels() * I1.total());
	 double psnr=10*log10(255*255/mse);
	 return psnr;	 
  }
}

Mat IBP(const Mat &ImageGT,const Mat &SR,const Mat &LR,int iters)
{
  Mat H_IBP=Mat::zeros(ImageGT.rows, ImageGT.cols, ImageGT.type());
  SR.copyTo(H_IBP);
  double SR_psnr=0.0f;
  for(int i=0;i!=iters;++i)
  {
    Mat blur_H=Mat::zeros(ImageGT.rows, ImageGT.cols, ImageGT.type());
    Mat downSample=Mat::zeros(LR.rows,LR.cols, LR.type());
    GaussianBlur(H_IBP,blur_H, Size(3,3),0,0);
    resize(blur_H,downSample,downSample.size(),0,0,2);

    Mat error;
    error=LR-downSample;

    Mat error_cubic;
    resize(error,error_cubic,ImageGT.size(),0,0,2);
    H_IBP=H_IBP+error_cubic;
	SR_psnr=getPSNR(ImageGT,H_IBP);

	cout<<i+1<<"  IBP's PSNR:  "<<SR_psnr<<endl;
  }
  return H_IBP;
}



double GSSD(const Mat &lr_patch,const Mat &lr_example)
{
    Mat mean;  
    Mat stddev;  
    meanStdDev(lr_example, mean, stddev);

	double sigma=stddev.at<double>(0,0);

	float ssd=0;
	for(int i=0;i<lr_patch.rows;i++)
	{
	  for(int j=0;j<lr_patch.cols;j++)
	  {
	     float recSample = lr_patch.at<float>(i, j);
		 float gtSample = lr_example.at<float>(i, j);
	     float t = recSample - gtSample;
		 ssd+=t*t;
	  }
	}
	double dist=exp(-ssd/(pow(sigma,2)*255));
	return dist;
}

/*void nearestNeighborPatchs(Mat* patch, int patchSize,flann::Index* tree, Mat* pLow, Mat* pHigh, int K, double weight, Mat* atomsLow, Mat* atomsHigh)
{
	//index of the nearest patches , sorted by distance/
	Mat ind;
	//Eculidean distances of nearest patches/
	// NOTE!: DISTANCE IS EUCLIDEAN DISTANCE SQUARED! /
	Mat dist;

	long temp = getTickCount();
	//search the tree for the K nearest patches of 'patch'/
	tree->knnSearch(*patch, ind, dist, K);
	long t = getTickCount() - temp;

	double* scaleVec = new double[K];
	for (int i = 0; i < K; i++)
	{
		// for each K-nearest patches/
		double d = (double)dist.at<float>(0, i);

		//compute the atom weighting/
		double e = -(d) / weight;
		double scale = exp(e);
		scaleVec[i] = scale;
	}

	//compute the sum of the atoms weights for normalization/
	double scaleSum = 0.0;
	for (int i = 0; i < K; i++)
	{
		scaleSum = scaleSum + scaleVec[i];
	}
	double tempSum = 0.0;
	for (int i = 0; i < K; i++)
	{
		scaleVec[i] = scaleVec[i] / scaleSum;
		tempSum += scaleVec[i];
	}

	for (int k = 0; k < K; k++)
	{
		int index = ind.at<int>(0, k);
		//if an error exists, append a zero atom/
		if (index < pLow->rows && index >= 0 && scaleSum != 0.0)
		{
			//append the LR atoms to "atomsLow', and HR atoms to 'atomsHigh'. Weight and normalize/
			
			for (int j = 0; j < patchSize*patchSize; j++)
			{
				atomsLow->at<double>(j, k) = (double)(pLow->at<float>(index, j)*(scaleVec[k]));
				atomsHigh->at<double>(j, k) = (double)(pHigh->at<float>(index, j)*(scaleVec[k]));
			}
		}
		else
		{
			for (int j = 0; j < patchSize*patchSize; j++)
			{
				atomsLow->at<double>(j, k) = 0.0;
				atomsHigh->at<double>(j, k) = 0.0;
			}
		}
	}

	//clean up/
	delete scaleVec;
	ind.release();
	dist.release();
}*/
