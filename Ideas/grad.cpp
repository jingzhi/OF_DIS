#include <opencv2/core/core.hpp>
#include <opencv2/highgui//highgui.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include "matplotlibcpp.h"
#include <fstream>
#include <limits>
using namespace std;
using namespace cv;
using namespace optflow;
namespace plt = matplotlibcpp;

static Mat flowToDisplay(const Mat flow)
{
    Mat flow_split[2];
    Mat magnitude, angle;
    Mat hsv_split[3], hsv, bgr;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, bgr, COLOR_HSV2BGR);
    return bgr;
}

bool warping_error(Mat &window_error,
		          const Rect& roi, 
		          const Mat& window_data, 
		          const Mat& window_flow_u,
			  const Mat& window_flow_v,
			  const Mat& img_bo_mat)
{
    Size sz = img_bo_mat.size();
    int w= sz.width;   
    int h = sz.height; 
    for(int x =roi.x, i =0; i < roi.width; x++,i++){
         for(int y=roi.y, j=0; j <  roi.height; y++,j++){
             float u=window_flow_u.at<float>(j,i);
             float v=window_flow_v.at<float>(j,i);
	     int x_new=round(x+u), y_new=round(y+v);
	     //cout<<"x:"<<x<<", y:"<<y<<",\tu:"<<u<<", v:"<<v<<",\t x_new:"<<x_new<<", y_new:"<<y_new<<endl;
	     if( x_new >= w || x_new <0 || y_new >= h || y_new <0){
                 return false;
	     }
	     else{
		 Vec3b color;
		 cv::absdiff(window_data.at<Vec3b>(j,i),img_bo_mat.at<Vec3b>(y_new,x_new),color);
                 window_error.at<Vec3b>(j,i)=color;
	         //cout<<"current: "<< window_data.at<Vec3b>(j,i)<<"\tx:"<<x<<", y:"<<y<<endl;
	         //cout<<"target : "<<img_bo_mat.at<Vec3b>(y_new,x_new)<<"\tx_new:"<<x_new<<", y_new:"<<y_new<<endl;
	         //cout<<"error pt="<<color<<endl<<endl;

		     
	     }
	 }
    }
    //cout<<"Error in function: ="<<endl<<window_error<<endl<<endl;   
    return true;

}
Mat warping_error_norm_glb;
Mat warping_error_norm_glb_l1;
Mat warping_error_norm_glb_l2;
Mat warping_error_norm_glb_linf;
Mat img_ao_mat_glb;
int thresholdSlider = 10;
int thresholdSlider_l1 = 10;
int thresholdSlider_l2 = 10;
int thresholdSlider_linf = 10;

void warpingErrorThreshold(int,void*)
{
    Mat error_mask = warping_error_norm_glb >thresholdSlider;
    Mat masked_img_ao(img_ao_mat_glb.size(),CV_32FC1,Scalar(0));
    img_ao_mat_glb.copyTo(masked_img_ao,error_mask);
    imshow( "Error Map", masked_img_ao);
    return ;
}
void warpingErrorThreshold_l1(int,void*)
{
    Mat error_mask = warping_error_norm_glb_l1 >thresholdSlider_l1;
    Mat masked_img_ao(img_ao_mat_glb.size(),CV_32FC1,Scalar(0));
    img_ao_mat_glb.copyTo(masked_img_ao,error_mask);
    imshow( "Error Map l1", masked_img_ao);
    return ;
}
void warpingErrorThreshold_l2(int,void*)
{
    Mat error_mask = warping_error_norm_glb_l2 >thresholdSlider_l2;
    Mat masked_img_ao(img_ao_mat_glb.size(),CV_32FC1,Scalar(0));
    img_ao_mat_glb.copyTo(masked_img_ao,error_mask);
    imshow( "Error Map l2", masked_img_ao);
    return ;
}
void warpingErrorThreshold_linf(int,void*)
{
    Mat error_mask = warping_error_norm_glb_linf >thresholdSlider_linf;
    Mat masked_img_ao(img_ao_mat_glb.size(),CV_32FC1,Scalar(0));
    img_ao_mat_glb.copyTo(masked_img_ao,error_mask);
    imshow( "Error Map linf", masked_img_ao);
    return ;
}
Mat warping_count_glb;
int countThresholdSlider = 1;
void warpingCountThreshold(int,void*)
{
    Mat count_mask = warping_count_glb == countThresholdSlider;
    Mat masked_img_ao(img_ao_mat_glb.size(),CV_32FC1,Scalar(0));
    img_ao_mat_glb.copyTo(masked_img_ao,count_mask);
    //imshow( "Warping count Map", masked_img_ao);
    imshow( "Warping count Map", count_mask);
    return ;
}
float my_mean(Mat & in_data){
    int w=in_data.cols;
    int h=in_data.rows;
    int crop=1;
    float sum=0.f;
    for(int i = crop;i < w-crop;i++){
       for(int j = crop; j<h-crop;j++){
           if(in_data.at<float>(i,j)<22.5) sum+=360.f-in_data.at<float>(i,j);
	   else sum+=in_data.at<float>(i,j);
       }
    }

    return sum/((w-2*crop)*(h-2*crop));

}
/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh ;
int lowThreshold = 20;
int const max_lowThreshold = 300;
int ratio = 5;
int kernel_size = 3;
string window_name = "Edge Map";


void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 2x3
  blur( src_gray, detected_edges, Size(6,6) );
  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(-1);
  src.copyTo( dst, detected_edges);
  imshow( window_name, detected_edges );
 }

void myCanny(Mat & out_mat, Mat & in_mat)
{
  src=in_mat.clone();
  src.convertTo(src_gray,CV_8U);
  dst.create( src.size(), src.type() );
  /// Convert the image to grayscale
  cvtColor( src_gray, src_gray, CV_BGR2GRAY );
  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );
  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
 /// Show the image
  CannyThreshold(0, 0);
  /// Wait until user exit program by pressing a key
  waitKey(0);
  dst.copyTo( out_mat);
}

int main( int argc, char** argv )
{
    //****pathes****//
    string imgfile_ao = "/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/alley_1/frame_0021.png";
    string imgfile_bo = "/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/alley_1/frame_0022.png";
    //string flow_path = "./frame_0021.flo";
    string flow_path_gt = "/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/flow/alley_1/frame_0021.flo";
    string outfile = "./grad_frame_0021.flo";
    string flow_path = "/scratch/lijingz/MasterThesis/OF_DIS/build/frame_0021.flo";
   
    bool display_images=true;
    
    //****params****//
    int window_sz=9; //9x9 sliding window
    int stride=1;
    float std_thres=25;
    float warping_error_thres=10.f;
    //****load uin8 img****//
    Mat img_ao_mat, img_bo_mat, img_viz, flow_viz ;
    Mat img_ao_mat_gray, img_bo_mat_gray ;
    int rpyrtype, nochannels, incoltype;
    incoltype = CV_LOAD_IMAGE_COLOR;
    rpyrtype = CV_32FC3;
    nochannels = 3;      
    img_ao_mat = cv::imread(imgfile_ao, incoltype);   // Read the file
    img_bo_mat = cv::imread(imgfile_bo, incoltype);   // Read the file    
    img_ao_mat_glb=img_ao_mat.clone();
    //cout<<"type"<<img_ao_mat.type()<<endl;
    cvtColor(img_bo_mat,img_bo_mat_gray,COLOR_RGB2GRAY);
    cvtColor(img_ao_mat,img_ao_mat_gray,COLOR_RGB2GRAY);
    img_viz=img_ao_mat.clone();
    Size sz = img_ao_mat.size();
    int width_org = sz.width;   // unpadded original image size
    int height_org = sz.height;  // unpadded original image size 
    //****float img****//
    Mat img_ao_fmat, img_bo_fmat;
    img_ao_mat.convertTo(img_ao_fmat, CV_32FC3); // convert to float
    img_bo_mat.convertTo(img_bo_fmat, CV_32FC3);
    //cout<<"img_ao_mat ="<<endl<<img_ao_mat<<endl<<endl;
    //cout<<"img_ao_fmat ="<<endl<<img_ao_fmat<<endl<<endl;
    //namedWindow("float ao",CV_WINDOW_AUTOSIZE);
    //imshow("float ao",img_ao_fmat);
    //namedWindow("original ao",CV_WINDOW_AUTOSIZE);
    //imshow("original ao",img_ao_mat);
  
    //Mat img_ao_edge;
    //myCanny(img_ao_edge,img_ao_mat);
    Mat img_ao_channel[3];
    split(img_ao_fmat,img_ao_channel);
    Mat img_ao_mag;
    Mat img_ao_dx;
    Mat img_ao_dy;
    Mat img_ao_grad_mag;
    Mat img_ao_grad_angle;
    //cvtColor(img_ao_fmat,img_ao_mag,COLOR_RGB2GRAY);
    img_ao_mag=img_ao_channel[2].clone();
    normalize(img_ao_mag,img_ao_mag, 0, 1, NORM_MINMAX);
    Sobel(img_ao_mag,img_ao_dx,CV_32F,1,0,3,1,0,BORDER_DEFAULT);
    Sobel(img_ao_mag,img_ao_dy,CV_32F,0,1,3,1,0,BORDER_DEFAULT);
    cartToPolar(img_ao_dx,img_ao_dy,img_ao_grad_mag, img_ao_grad_angle,true); //true= angle in degree;
    //normalize(img_ao_dx,img_ao_dx, 0, 1, NORM_MINMAX);
    //normalize(img_ao_dy,img_ao_dy, 0, 1, NORM_MINMAX);
    //normalize(img_ao_grad_mag,img_ao_grad_mag, 0, 1, NORM_MINMAX);
    //namedWindow("img_ao grad mag",CV_WINDOW_AUTOSIZE);
    //imshow("img_ao grad mag",img_ao_grad_mag);
    //namedWindow("img_ao dx",CV_WINDOW_AUTOSIZE);
    //imshow("img_ao dx",img_ao_dx);
    //namedWindow("img_ao dy",CV_WINDOW_AUTOSIZE);
    //imshow("img_ao dy",img_ao_dy);
    //namedWindow("original ao gray",CV_WINDOW_AUTOSIZE);
    //imshow("original ao gray",img_ao_mag);
    //BRG
    //namedWindow("original ao 0",CV_WINDOW_AUTOSIZE);
    //imshow("original ao 0",img_ao_channel[0]);
    //namedWindow("original ao 1",CV_WINDOW_AUTOSIZE);
    //imshow("original ao 1",img_ao_channel[1]);
    //namedWindow("original ao 2",CV_WINDOW_AUTOSIZE);
    //imshow("original ao 2",img_ao_channel[2]);
    //cout<<"type: "<<img_ao_mag.type()<<endl;


    Mat in_flow,out_flow, in_flow_edge,flow_image,flow_mag,flow_angle;
    vector<Mat> in_flow_split;
    in_flow = readOpticalFlow(flow_path);
    split(in_flow, in_flow_split);
    //GT
    Mat gt_flow;
    vector<Mat> gt_flow_split;
    gt_flow = readOpticalFlow(flow_path_gt);
    split(gt_flow, gt_flow_split);

    if(display_images)
    {
        Mat flow_image = flowToDisplay(in_flow);
        flow_viz=flow_image.clone();
        namedWindow( "Input flow from OF_DIS", WINDOW_AUTOSIZE );
        imshow( "Input flow from OF_DIS", flow_image );
    }
    Mat flow_dx;
    Mat flow_dy;
    Mat flow_grad_mag;
    Mat flow_grad_angle;
    cartToPolar(in_flow_split[0], in_flow_split[1], flow_mag, flow_angle, true);
    Sobel(flow_mag,flow_dx,CV_32F,1,0,3,1,0,BORDER_DEFAULT);
    Sobel(flow_mag,flow_dy,CV_32F,0,1,3,1,0,BORDER_DEFAULT);
    cartToPolar(flow_dx,flow_dy,flow_grad_mag, flow_grad_angle,true); //true= angle in degree;
    //normalize(flow_mag,flow_mag, 0, 1, NORM_MINMAX);
    //normalize(flow_dx,flow_dx, 0, 1, NORM_MINMAX);
    //normalize(flow_dy,flow_dy, 0, 1, NORM_MINMAX);
    //normalize(flow_grad_mag,flow_grad_mag, 0, 1, NORM_MINMAX);
    //namedWindow("flow grad mag",CV_WINDOW_AUTOSIZE);
    //imshow("flow grad mag",flow_grad_mag);
    //namedWindow("flow dx",CV_WINDOW_AUTOSIZE);
    //imshow("flow dx",flow_dx);
    //namedWindow("flow dy",CV_WINDOW_AUTOSIZE);
    //imshow("flow dy",flow_dy);


    
    ////****Perturb the flow within sliding window****//
    int counter =0;
    //int col =700,row=399;
   // if(true){
   //             Mat window_img, 
   //     	    window_img_dx,
   //     	    window_img_dy,
   //     	    window_img_grad_mag,
   //     	    window_img_grad_angle,
   //                mean_img,
   //                std_img;
   //            Rect roi(col,row,window_sz,window_sz);
   //            for(int c =0;c<3;c++){
   //                 window_img = img_ao_channel[c](roi);
   //         	    Sobel(window_img,window_img_dx,CV_32F,1,0,3,1,0,BORDER_DEFAULT);
   //         	    Sobel(window_img,window_img_dy,CV_32F,0,1,3,1,0,BORDER_DEFAULT);
   //         	    cartToPolar(window_img_dx,window_img_dy,window_img_grad_mag,window_img_grad_angle,true); //true= angle in degree;
   //            meanStdDev(window_img_grad_mag,mean_img,std_img);
   //                    rectangle(img_viz, roi, Scalar(255,0,0), 1, 8, 0);
   //                    rectangle(flow_viz, roi, Scalar(2550,0,0), 1, 8, 0);    
   //    		cout<<"x:"<<col<<"y:"<<row<<"channel:"<<c<<endl;
   //        	        cout<<"img  ="<<endl<<window_img<<endl<<endl;
   //        	        cout<<"img dx ="<<endl<<window_img_dx<<endl<<endl;
   //        	        cout<<"img dy ="<<endl<<window_img_dy<<endl<<endl;
   //        	        cout<<"img grad mag ="<<endl<<window_img_grad_mag<<endl<<endl;
   //        	        cout<<"Mean:"<<mean_img.at<double>(0,0)<<endl;
   //        	        cout<<"Std:"<<std_img.at<double>(0,0)<<endl;
   //            }
   // }
   // col =709,row=397;
   // col =718,row=392;
   // col =726,row=384;
   // col =735,row=381;
   // col =742,row=372;


   //*** construct error map*****//
   Mat warping_error(img_ao_fmat.size(),CV_32FC3);
   Mat warping_error_norm_l1(img_ao_fmat.size(),CV_32FC1);
   Mat warping_error_norm_l2(img_ao_fmat.size(),CV_32FC1);
   Mat warping_error_norm_linf(img_ao_fmat.size(),CV_32FC1);
   Mat warping_count(img_ao_fmat.size(),CV_32FC1,Scalar(0));
   Mat warping_count_gt(img_ao_fmat.size(),CV_32FC1,Scalar(0));
   Mat flow_u_mat=in_flow_split[0];
   Mat flow_v_mat=in_flow_split[1];
   for(int iter =0;iter<1;iter++){
       for (int col = 0; col <= width_org - window_sz;col+=window_sz){
           for (int row = 0; row<=height_org - window_sz;row+=window_sz){
   
                Mat window_img, 
                    window_error, 
        	    window_img_dx,
        	    window_img_dy,
        	    window_img_grad_mag,
        	    window_img_grad_angle,
                    mean_img,
                    mean_dx,
                    mean_dy,
                    std_img,
                    std_dx,
                    std_dy;
               Rect roi(col,row,window_sz,window_sz);
	       //***each pixel in the patch//
               for(int x =roi.x, i =0; i < roi.width; x++,i++){
                   for(int y=roi.y, j=0; j <  roi.height; y++,j++){
                     float u=flow_u_mat.at<float>(y,x);
                     float v=flow_v_mat.at<float>(y,x);
               	     int x_new=round(x+u), y_new=round(y+v);
                     float u_gt=gt_flow_split[0].at<float>(y,x);
                     float v_gt=gt_flow_split[1].at<float>(y,x);
               	     int x_new_gt=round(x+u_gt), y_new_gt=round(y+v_gt);
               	     //cout<<"x:"<<x<<", y:"<<y<<",\tu:"<<u<<", v:"<<v<<",\t x_new:"<<x_new<<", y_new:"<<y_new<<endl;
               	     if( x_new >= width_org || x_new <0 || y_new >= height_org || y_new <0){
                                warping_error.at<Vec3f>(y,x)[0]=0;
                                warping_error.at<Vec3f>(y,x)[1]=0;
                                warping_error.at<Vec3f>(y,x)[2]=0;
               	     }
               	     else{
		         warping_count.at<float>(y_new,x_new)+=1;
               		 Vec3f diff;
               		 cv::absdiff(img_ao_fmat.at<Vec3f>(y,x),img_bo_fmat.at<Vec3f>(y_new,x_new),diff);
                         warping_error.at<Vec3f>(y,x)=diff;
                         warping_error_norm_l1.at<float>(y,x)=norm(diff,NORM_L1);
                         warping_error_norm_l2.at<float>(y,x)=norm(diff,NORM_L2);
                         warping_error_norm_linf.at<float>(y,x)=norm(diff,NORM_INF);
               	         //cout<<"current: "<< img_ao_fmat.at<Vec3f>(y,x)<<"\tx:"<<x<<", y:"<<y<<endl;
               	         //cout<<"target : "<<img_bo_fmat.at<Vec3f>(y_new,x_new)<<"\tx_new:"<<x_new<<", y_new:"<<y_new<<endl;
               	         //cout<<"error pt="<<diff<<endl<<endl;
               		     
               	         }
               	     if( x_new_gt >= width_org || x_new_gt <0 || y_new_gt >= height_org || y_new_gt <0){
                                warping_error.at<Vec3f>(y,x)[0]=0;
                                warping_error.at<Vec3f>(y,x)[1]=0;
                                warping_error.at<Vec3f>(y,x)[2]=0;
               	     }
               	     else{
		         warping_count_gt.at<float>(y_new_gt,x_new_gt)+=1;
               	         }
                   }
               }
                       //rectangle(img_viz, roi, Scalar(255,0,0), 1, 8, 0);
                       //rectangle(flow_viz, roi, Scalar(2550,0,0), 1, 8, 0);    
           }    
       }
    ////in_flow=out_flow.clone();

    }
    //writeOpticalFlow(outfile,out_flow);
    //if(display_images)
    //{
    //    Mat flow_image = flowToDisplay(out_flow);
    //    namedWindow( "Output flow", WINDOW_AUTOSIZE );
    //    imshow( "Output flow", flow_image );
    //}
    //

    //****Warping error map****//
    warping_error_norm_glb=warping_error_norm_l2.clone();
    //warping_error_norm_glb_l1=warping_error_norm_l1.clone();
    //warping_error_norm_glb_l2=warping_error_norm_l2.clone();
    //warping_error_norm_glb_linf=warping_error_norm_linf.clone();
    //namedWindow( "Error Map", WINDOW_AUTOSIZE );
    //namedWindow( "Error Map l1", WINDOW_AUTOSIZE );
    //namedWindow( "Error Map l2", WINDOW_AUTOSIZE );
    //namedWindow( "Error Map linf", WINDOW_AUTOSIZE );
    //int maxThreshold = 500;
    //createTrackbar( "lower error bound:", "Error Map", &thresholdSlider,maxThreshold, warpingErrorThreshold);
    //createTrackbar( "lower error bound:", "Error Map l1", &thresholdSlider_l1,maxThreshold, warpingErrorThreshold_l1);
    //createTrackbar( "lower error bound:", "Error Map l2", &thresholdSlider_l2,maxThreshold, warpingErrorThreshold_l2);
    //createTrackbar( "lower error bound:", "Error Map linf", &thresholdSlider_linf,maxThreshold, warpingErrorThreshold_linf);
    //warpingErrorThreshold(thresholdSlider, 0);
    //warpingErrorThreshold_l1(thresholdSlider_l1, 0);
    //warpingErrorThreshold_l2(thresholdSlider_l2, 0);
    //warpingErrorThreshold_linf(thresholdSlider_linf, 0);

    
    
    //****warping count map****//
    //warping_count_glb=warping_count.clone();
    //namedWindow( "Warping count Map", WINDOW_AUTOSIZE );
    //createTrackbar( "Warping count:", "Warping count Map", &countThresholdSlider,20, warpingCountThreshold);
    //warpingCountThreshold(countThresholdSlider, 0);
    //waitKey(0);
    //
    int erosion_size=1;
    int dilation_size=1;
    //int morph_type =MORPH_CROSS;
    int erosion_type =MORPH_CROSS;
    //int dilation_type =MORPH_RECT;
    int dilation_type =MORPH_CROSS;
    Mat erosion_element = getStructuringElement( erosion_type,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
    Mat dilation_element = getStructuringElement( dilation_type,
                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       Point( dilation_size, dilation_size ) );

    Mat count_mask, count_mask_gt;
    count_mask = warping_count == 0;
    dilate(count_mask,count_mask,dilation_element);
    erode(count_mask,count_mask,erosion_element);
    erode(count_mask,count_mask,erosion_element);
    dilate(count_mask,count_mask,dilation_element);
    erode(count_mask,count_mask,erosion_element);
    dilate(count_mask,count_mask,dilation_element);
    erode(count_mask,count_mask,erosion_element);
    dilate(count_mask,count_mask,dilation_element);
    imshow( "Warping count =0", count_mask);
    cout<<"no. of count 0 pixel in_flow:\t"<<cv::countNonZero(count_mask)<<endl;

    count_mask = warping_count == 1;
    erode(count_mask,count_mask,erosion_element);
    dilate(count_mask,count_mask,dilation_element);
    imshow( "Warping count =1", count_mask);
    cout<<"no. of count 1 pixel in_flow:\t"<<cv::countNonZero(count_mask)<<endl;

    count_mask = warping_count >= 2;
    dilate(count_mask,count_mask,dilation_element);
    erode(count_mask,count_mask,erosion_element);
    erode(count_mask,count_mask,erosion_element);
    dilate(count_mask,count_mask,dilation_element);
    erode(count_mask,count_mask,erosion_element);
    dilate(count_mask,count_mask,dilation_element);
    erode(count_mask,count_mask,erosion_element);
    dilate(count_mask,count_mask,dilation_element);
    imshow( "Warping count >=2", count_mask);
    cout<<"no. of count >=2 pixel in_flow:\t"<<cv::countNonZero(count_mask)<<endl;

    count_mask_gt = warping_count_gt == 0;
    imshow( "GT:Warping count =0", count_mask_gt);
    cout<<"no. of count 0 pixel gt_flow:\t"<<cv::countNonZero(count_mask_gt)<<endl;
    count_mask_gt = warping_count_gt == 1;
    imshow( "GT:Warping count =1", count_mask_gt);
    cout<<"no. of count 1 pixel gt_flow:\t"<<cv::countNonZero(count_mask_gt)<<endl;
    count_mask_gt = warping_count_gt >= 2;
    imshow( "GT:Warping count >=2", count_mask_gt);
    cout<<"no. of count >=2 pixel gt_flow:\t"<<cv::countNonZero(count_mask_gt)<<endl;

     
    //namedWindow( "Visualization", WINDOW_AUTOSIZE );
    //imshow( "Visualization", img_viz );
    //namedWindow( "Flow Visualization", WINDOW_AUTOSIZE );
    //imshow( "Flow Visualization", flow_viz );

    if(display_images) // wait for the user to see all the images
        waitKey(0);
    return 0;

}
