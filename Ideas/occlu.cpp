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
    Mat hsv_split[3], hsv, rgb;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
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

void warp_img(Mat& warpped_img,
              const Rect& roi, 
              const Mat& window_data, 
              const Mat& window_flow_u,
              const Mat& window_flow_v,
              const Mat& img_ao_mat)
{
    Size sz = img_ao_mat.size();
    int w= sz.width;   
    int h = sz.height; 
    for(int x =roi.x, i =0; i < roi.width; x++,i++){
         for(int y=roi.y, j=0; j <  roi.height; y++,j++){
             double u=window_flow_u.at<double>(j,i);
             double v=window_flow_v.at<double>(j,i);
	     int x_new=floor(x+u), y_new=floor(y+u);
	     if( x_new >= w || x_new <0 || y_new >= h || y_new <0){
                 return ;
	     }
	     else{
		 for(int c=0; c<=2;c++){
		     //cout<<"error: "<<(int)window_error.at<Vec3b>(j,i)[c]<<endl; 
		 }
	     }
	 }
    }
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

int main( int argc, char** argv )
{
    //****pathes****//
    string imgfile_ao = "/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/alley_1/frame_0021.png";
    string imgfile_bo = "/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/alley_1/frame_0022.png";
    string flow_path = "./frame_0021.flo";
    string outfile = "./tuned_frame_0021.flo";
   
    bool display_images=true;
    
    //****params****//
    int window_sz=9; //9x9 sliding window
    int stride=9;
    float std_thres=0.7;
    float warpping_error_thres=10.f;
    //****load uin8 img****//
    Mat img_ao_mat, img_bo_mat, img_viz, flow_viz ;
    Mat img_ao_mat_gray, img_bo_mat_gray ;
    int rpyrtype, nochannels, incoltype;
    incoltype = CV_LOAD_IMAGE_COLOR;
    rpyrtype = CV_32FC3;
    nochannels = 3;      
    img_ao_mat = cv::imread(imgfile_ao, incoltype);   // Read the file
    img_bo_mat = cv::imread(imgfile_bo, incoltype);   // Read the file    
    cvtColor(img_bo_mat,img_bo_mat_gray,COLOR_RGB2GRAY);
    cvtColor(img_ao_mat,img_ao_mat_gray,COLOR_RGB2GRAY);
    img_viz=img_ao_mat.clone();
    Size sz = img_ao_mat.size();
    int width_org = sz.width;   // unpadded original image size
    int height_org = sz.height;  // unpadded original image size 
    //****float img****//
    //Mat img_ao_fmat, img_bo_fmat;
    //img_ao_mat.convertTo(img_ao_fmat, CV_32FC3); // convert to float
    //img_bo_mat.convertTo(img_bo_fmat, CV_32FC3);
    //cout<<"img_ao_mat ="<<endl<<img_ao_mat<<endl<<endl;
    //cout<<"img_ao_fmat ="<<endl<<img_ao_fmat<<endl<<endl;
    //namedWindow("original ao",CV_WINDOW_AUTOSIZE);
    //namedWindow("float ao",CV_WINDOW_AUTOSIZE);
    //imshow("original ao",img_ao_mat);
    //imshow("float ao",img_ao_fmat);
  
    Mat_<Point2f> in_flow, out_flow;
    Mat computed_errors;
    in_flow = readOpticalFlow(flow_path);
    out_flow=in_flow.clone();
    vector<Mat> in_flow_split;
    split(in_flow, in_flow_split);
    if(display_images)
    {
        Mat flow_image = flowToDisplay(in_flow);
        flow_viz=flow_image.clone();
        namedWindow( "Input flow from OF_DIS", WINDOW_AUTOSIZE );
        imshow( "Input flow from OF_DIS", flow_image );
    }
    //****Perturb the flow within sliding window****//
    cout<<"width:"<<width_org<<endl;
    cout<<"height:"<<height_org<<endl;
    for(int iter =0;iter<1;iter++){
        for (int col = 500; col <= width_org - window_sz;col+=stride){
            for (int row = 0; row<=height_org-300 - window_sz;row+=stride){
                Mat window_data, 
            	window_data_gray, 
            	window_flow, //[0],[1]
            	window_flow_u,
            	window_flow_v, 
            	window_error,
                    mean_u,
                    mean_v,
            	mean_error,
                    std_u,
                    std_v,
            	std_error;
                Rect roi(col,row,window_sz,window_sz);
                window_data = img_ao_mat(roi);
                //window_error = img_ao_mat(roi);
                window_error = window_data.clone();
                window_flow_u = in_flow_split[0](roi);
                window_flow_v = in_flow_split[1](roi);
                window_flow = in_flow(roi);
                //bool warp_valid = warping_error(window_error, roi, window_data, window_flow_u, window_flow_v, img_bo_mat);
                meanStdDev(window_flow_u,mean_u,std_u);
                meanStdDev(window_flow_v,mean_v,std_v);
                if(std_u.at<double>(0,0) > std_thres || std_v.at<double>(0,0) >std_thres){
                    bool warp_valid = warping_error(window_error, roi, window_data, window_flow_u, window_flow_v, img_bo_mat);
                    if(warp_valid){
                        meanStdDev(window_error,mean_error,std_error);
                        //rectangle(img_viz, roi, Scalar(255,0,0), 1, 8, 0);
                        //rectangle(flow_viz, roi, Scalar(2550,0,0), 1, 8, 0);
            	    Mat padded_window_flow,
            		window_flow_mag,
            		window_flow_angle,
            		window_flow_mag_dx,
            		window_flow_mag_dy,
            		window_flow_mag_grad,
            		window_flow_mag_grad_angle;
            	    cartToPolar(window_flow_u,window_flow_v,window_flow_mag,window_flow_angle,true); //true= angle in degree;
            	    Sobel(window_flow_mag,window_flow_mag_dx,CV_32F,1,0,3,1,0,BORDER_DEFAULT);
            	    Sobel(window_flow_mag,window_flow_mag_dy,CV_32F,0,1,3,1,0,BORDER_DEFAULT);
            	    cartToPolar(window_flow_mag_dx,window_flow_mag_dy,window_flow_mag_grad,window_flow_mag_grad_angle,true); //true= angle in degree;
            	    float meanDirection=my_mean(window_flow_mag_grad_angle);
            	    Scalar meanGrad=cv::mean(window_flow_mag_grad);
            	    Scalar meanFlowDirection=cv::mean(window_flow_angle);
            	    Scalar meanFlowMag=cv::mean(window_flow_mag);
            	    //cout<<"Flow mag ="<<endl<<window_flow_mag<<endl<<endl;
            	    //cout<<"Flow angle ="<<endl<<window_flow_angle<<endl<<endl;
            	    //cout<<"Mean direction:"<<meanDirection<<endl;
            	    //cout<<"Mean grad:"<<meanGrad.val[0]<<endl;
            	    //cout<<"Mean flow direction:"<<meanFlowDirection.val[0]<<endl;
            	    //cout<<"Mean flow mag:"<<meanFlowMag.val[0]<<endl;


            	    Mat_<int> speed_vec(2,1),flow_vec(2,1);
            	    //****SPEED PATTERN****//

            	    if(22.5<= meanDirection && meanDirection< 67.5){
            		    cout<<"speed pattern: \\v"<<endl;
            		    speed_vec<<1,-1;
            	    }
            	    else if(67.5<= meanDirection && meanDirection< 112.5){
            		    cout<<"speed pattern: |v"<<endl;
            		    speed_vec<<0,-1;
            	    }
            	    else if(112.5<= meanDirection && meanDirection< 157.5){
            		    cout<<"speed pattern: /v "<<endl;
            		    speed_vec<<-1,-1;
            	    }
            	    else if(157.5<= meanDirection && meanDirection< 202.5){
            		    cout<<"speed pattern: <--"<<endl;
            		    speed_vec<<-1,0;
            	    }
            	    else if(202.5<= meanDirection && meanDirection< 247.5){
            		    cout<<"speed pattern: ^\\"<<endl;
            		    speed_vec<<-1,1;
            	    }
            	    else if(247.5<= meanDirection && meanDirection< 292.5){
            		    cout<<"speed pattern: ^|"<<endl;
            		    speed_vec<<0,1;
            	    }
            	    else if(292.5<= meanDirection && meanDirection< 337.5){
            		    cout<<"speed pattern: ^/"<<endl;
            		    speed_vec<<1,1;
                        //rectangle(flow_viz, roi, Scalar(255,0,255), 1, 8, 0);
                        //rectangle(img_viz, roi, Scalar(255,0,255), 1, 8, 0);
            	    }
            	    else{
            		    cout<<"speed pattern: -->"<<endl;
            		    speed_vec<<1,0;
                            //rectangle(flow_viz, roi, Scalar(0,0,255), 1, 8, 0);
                            //rectangle(img_viz, roi, Scalar(0,0,255), 1, 8, 0);
            	    }
            	    //****FLOW DIRECTION****//
            	    if(23.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 67.5){
            		    cout<<"flow direction: \\v"<<endl;
            		    flow_vec<<1,-1;
            		    
            	    }
            	    else if(67.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 112.5){
            		    cout<<"flow direction: |v"<<endl;
            		    flow_vec<<0,-1;
            	    }
            	    else if(112.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 157.5){
            		    cout<<"flow direction: /v "<<endl;
            		    flow_vec<<-1,-1;
            	    }
            	    else if(157.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 202.5){
            		    cout<<"flow direction: <--"<<endl;
            		    flow_vec<<-1,0;
            	    }
            	    else if(202.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 247.5){
            		    cout<<"flow direction: ^\\"<<endl;
            		    flow_vec<<-1,1;
            	    }
            	    else if(247.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 292.5){
            		    cout<<"flow direction: ^|"<<endl;
            		    flow_vec<<0,1;
            	    }
            	    else if(292.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 337.5){
            		    cout<<"flow direction: ^/"<<endl;
            		    flow_vec<<1,1;
            	    }
            	    else{
            		    cout<<"flow direction: -->"<<endl;
            		    flow_vec<<1,0;
            	    }
            	    //****PERTURB FLOW****//
             	    if(flow_vec.dot(speed_vec)>0)//SAME direction
            	    {
                            rectangle(flow_viz, roi, Scalar(0,0,255), 1, 8, 0);
                            rectangle(img_viz, roi, Scalar(0,0,255), 1, 8, 0);
                            //cout<<"Flow mag ="<<endl<<window_flow_mag<<endl<<endl;
            	            //cout<<"Flow angle ="<<endl<<window_flow_angle<<endl<<endl;
            	            //cout<<"Mean direction:"<<meanDirection<<endl;
            	            //cout<<"Mean grad:"<<meanGrad.val[0]<<endl;
            	            //cout<<"Mean flow direction:"<<meanFlowDirection.val[0]<<endl;
            	            //cout<<"Mean flow mag:"<<meanFlowMag.val[0]<<endl;
            	            int del_row  = round(std_u.at<double>(0,0)*3),
				del_col  = round(std_v.at<double>(0,0)*3);
            	            copyMakeBorder(window_flow, padded_window_flow, del_row, del_row, del_col, del_col, BORDER_REPLICATE);
            		    int col_new = del_col - del_col*speed_vec.at<int>(0,0);
            		    int row_new = del_row + del_row*speed_vec.at<int>(1,0);
                            Rect roi_target(col_new,row_new,window_sz,window_sz);
            		    Mat test = ((0.3*padded_window_flow(roi_target))+(0.7*out_flow(roi)));
			    test.copyTo(out_flow(roi));
            	    }
		    else if(flow_vec.dot(speed_vec)<0)//OPPOSITE Direction
            	    {
                            rectangle(flow_viz, roi, Scalar(255,0,0), 1, 8, 0);
                            rectangle(img_viz, roi,  Scalar(255,0,0), 1, 8, 0);
                            //cout<<"Flow mag ="<<endl<<window_flow_mag<<endl<<endl;
            	            //cout<<"Flow angle ="<<endl<<window_flow_angle<<endl<<endl;
            	            //cout<<"Mean direction:"<<meanDirection<<endl;
            	            //cout<<"Mean grad:"<<meanGrad.val[0]<<endl;
            	            //cout<<"Mean flow direction:"<<meanFlowDirection.val[0]<<endl;
            	            //cout<<"Mean flow mag:"<<meanFlowMag.val[0]<<endl;
            	            int del_row  = round(std_u.at<double>(0,0)*5),
				del_col  = round(std_v.at<double>(0,0)*5);
            	            copyMakeBorder(window_flow, padded_window_flow, del_row, del_row, del_col, del_col, BORDER_REPLICATE);
            		    int col_new = del_col - del_col*speed_vec.at<int>(0,0);
            		    int row_new = del_row + del_row*speed_vec.at<int>(1,0);
                            Rect roi_target(col_new,row_new,window_sz,window_sz);
            		    Mat test = ((0.3*padded_window_flow(roi_target))+(0.7*out_flow(roi)));
			    test.copyTo(out_flow(roi));
            	    }
		    else{
			//cout<<"PERP dirc"<<endl;
                        //rectangle(flow_viz, roi, Scalar(0,255,0), 1, 8, 0);
                        //rectangle(img_viz, roi, Scalar(0,255,0), 1, 8, 0);
		    }
		    cout<<endl;

                    }    
                }
                

            
            }    
        }
    in_flow=out_flow.clone();
    }
    writeOpticalFlow(outfile,out_flow);
    if(display_images)
    {
        Mat flow_image = flowToDisplay(out_flow);
        namedWindow( "Output flow", WINDOW_AUTOSIZE );
        imshow( "Output flow", flow_image );
    }

     
    namedWindow( "Visualization", WINDOW_AUTOSIZE );
    imshow( "Visualization", img_viz );
    namedWindow( "Flow Visualization", WINDOW_AUTOSIZE );
    imshow( "Flow Visualization", flow_viz );

    if(display_images) // wait for the user to see all the images
        waitKey(0);
    return 0;

}
