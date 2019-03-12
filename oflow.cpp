

#include <iostream>
#include <string>
#include <vector>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

//#include <opencv2/core/core.hpp> // needed for verbosity >= 3, DISVISUAL
//#include <opencv2/highgui/highgui.hpp> // needed for verbosity >= 3, DISVISUAL
//#include <opencv2/imgproc/imgproc.hpp> // needed for verbosity >= 3, DISVISUAL
#include <opencv2/core/core.hpp>
#include <opencv2/highgui//highgui.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include <fstream>
#include <limits>
#include <sys/time.h>    // timeof day
#include <stdio.h>  

#include "oflow.h"
#include "patchgrid.h"
#include "refine_variational.h"


using std::cout;
using std::endl;
using std::vector;
using namespace cv;


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
Mat flowToDisplay(const Mat flow)
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
namespace OFC
{

  OFClass::OFClass(const float ** im_ao_in, const float ** im_ao_dx_in, const float ** im_ao_dy_in, // expects #sc_f_in pointers to float arrays for images and gradients. 
                                                                                       // E.g. im_ao[sc_f_in] will be used as coarsest coarsest, im_ao[sc_l_in] as finest scale
                                                                                       // im_ao[  (sc_l_in-1) : 0 ] can be left as nullptr pointers
                                                                                       // IMPORTANT assumption: mod(width,2^sc_f_in)==0  AND mod(height,2^sc_f_in)==0, 
                  const float ** im_bo_in, const float ** im_bo_dx_in, const float ** im_bo_dy_in,
                  const int imgpadding_in,    
                  float * outflow,
                  const float * initflow,
                  const int width_in, const int height_in, 
                  const int sc_f_in, const int sc_l_in,
                  const int max_iter_in, const int min_iter_in,
                  const float  dp_thresh_in,
                  const float  dr_thresh_in,
                  const float res_thresh_in,            
                  const int p_samp_s_in,//patch size
                  const float patove_in,//patch overlap rate in %
                  const bool usefbcon_in, 
                  const int costfct_in,//cost function                   
                  const int noc_in,//no of channel
                  const int patnorm_in, 
                  const bool usetvref_in,
                  const float tv_alpha_in,
                  const float tv_gamma_in,
                  const float tv_delta_in,
                  const int tv_innerit_in,
                  const int tv_solverit_in,
                  const float tv_sor_in,
                  const int verbosity_in)
  : im_ao(im_ao_in), im_ao_dx(im_ao_dx_in), im_ao_dy(im_ao_dy_in),  
    im_bo(im_bo_in), im_bo_dx(im_bo_dx_in), im_bo_dy(im_bo_dy_in)
{
 
  
  #ifdef WITH_OPENMP
    if (verbosity_in>1)
      cout <<  "OPENMP is ON - used in pconst, pinit, potim";
    #ifdef USE_PARALLEL_ON_FLOWAGGR
    if (verbosity_in>1)
      cout << ", cflow ";
    #endif                                                                  
    if (verbosity_in>1) cout << endl;
  #endif //DWITH_OPENMP  
                                                                
  // Parse optimization parameters
  #if (SELECTMODE==1)
  op.nop = 2;
  #else
  op.nop = 1;
  #endif
  op.p_samp_s = p_samp_s_in;  // patch has even border length, center pixel is at (p_samp_s/2, p_samp_s/2) (ZERO INDEXED!) 
  op.outlierthresh = (float)op.p_samp_s;     
  op.patove = patove_in;
  op.sc_f = sc_f_in;
  op.sc_l = sc_l_in;
  op.max_iter = max_iter_in;
  op.min_iter = min_iter_in;
  op.dp_thresh = dp_thresh_in*dp_thresh_in; // saves the square to compare with squared L2-norm (saves sqrt operation)
  op.dr_thresh = dr_thresh_in;
  op.res_thresh = res_thresh_in;
  op.steps = std::max(1,  (int)floor(op.p_samp_s*(1-op.patove)));  
  op.novals = noc_in * (p_samp_s_in)*(p_samp_s_in);
  op.usefbcon = usefbcon_in;
  op.costfct = costfct_in;
  op.noc = noc_in;
  op.patnorm = patnorm_in;
  op.verbosity = verbosity_in;
  op.noscales = op.sc_f-op.sc_l+1;
  op.usetvref = usetvref_in;
  op.tv_alpha = tv_alpha_in;
  op.tv_gamma = tv_gamma_in;
  op.tv_delta = tv_delta_in;
  op.tv_innerit = tv_innerit_in;
  op.tv_solverit = tv_solverit_in;
  op.tv_sor = tv_sor_in;
  op.normoutlier_tmpbsq = (v4sf) {op.normoutlier*op.normoutlier, op.normoutlier*op.normoutlier, op.normoutlier*op.normoutlier, op.normoutlier*op.normoutlier};
  op.normoutlier_tmp2bsq = __builtin_ia32_mulps(op.normoutlier_tmpbsq, op.twos);
  op.normoutlier_tmp4bsq = __builtin_ia32_mulps(op.normoutlier_tmpbsq, op.fours);

  
  // Variables for algorithm timings
  struct timeval tv_start_all, tv_end_all, tv_start_all_global, tv_end_all_global;
  if (op.verbosity>0)
    gettimeofday(&tv_start_all_global, nullptr);
  
  // ... per each scale
  double tt_patconstr[op.noscales], tt_patinit[op.noscales], tt_patoptim[op.noscales], tt_compflow[op.noscales], tt_tvopt[op.noscales], tt_all[op.noscales];
  for (int sl=op.sc_f; sl>=op.sc_l; --sl) 
  {
    tt_patconstr[sl-op.sc_l]=0;
    tt_patinit[sl-op.sc_l]=0;
    tt_patoptim[sl-op.sc_l]=0;
    tt_compflow[sl-op.sc_l]=0;
    tt_tvopt[sl-op.sc_l]=0;
    tt_all[sl-op.sc_l]=0;
  }

  if (op.verbosity>1) gettimeofday(&tv_start_all, nullptr);
 
  
  // Create grids on each scale
  vector<OFC::PatGridClass*> grid_fw(op.noscales);
  vector<OFC::PatGridClass*> grid_bw(op.noscales); // grid for backward OF computation, only needed if 'usefbcon' is set to 1.
  vector<float*> flow_fw(op.noscales);
  vector<float*> flow_bw(op.noscales);
  cpl.resize(op.noscales);
  cpr.resize(op.noscales);
  for (int sl=op.sc_f; sl>=op.sc_l; --sl) 
  {
    int i = sl-op.sc_l;

    float sc_fct = pow(2,-sl); // scaling factor at current scale
    cpl[i].sc_fct = sc_fct;
    cpl[i].height = height_in * sc_fct;
    cpl[i].width = width_in * sc_fct;
    cpl[i].imgpadding = imgpadding_in;
    cpl[i].tmp_lb = -(float)op.p_samp_s/2; 
    cpl[i].tmp_ubw = (float) (cpl[i].width +op.p_samp_s/2-2);
    cpl[i].tmp_ubh = (float) (cpl[i].height+op.p_samp_s/2-2);
    cpl[i].tmp_w = cpl[i].width + 2*imgpadding_in;
    cpl[i].tmp_h = cpl[i].height+ 2*imgpadding_in;
    cpl[i].curr_lv = sl;
    cpl[i].camlr = 0;

    
    cpr[i] = cpl[i];
    cpr[i].camlr = 1;
    
    flow_fw[i]   = new float[op.nop * cpl[i].width * cpl[i].height]; 
    grid_fw[i]   = new OFC::PatGridClass(&(cpl[i]), &(cpr[i]), &op);
   
    if (op.usefbcon) // for merging forward and backward flow 
    {
      flow_bw[i] = new float[op.nop * cpr[i].width * cpr[i].height];
      grid_bw[i] = new OFC::PatGridClass(&(cpr[i]), &(cpl[i]), &op);
      
      // Make grids known to each other, necessary for AggregateFlowDense();
      grid_fw[i]->SetComplGrid( grid_bw[i] );
      grid_bw[i]->SetComplGrid( grid_fw[i] ); 
    }
  }
  
  
  // Timing, Grid memory allocation
  if (op.verbosity>1)
  {
    gettimeofday(&tv_end_all, nullptr);
    double tt_gridconst = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
    printf("TIME (Grid Memo. Alloc. ) (ms): %3g\n", tt_gridconst);          
  }
  

  // *** Main loop; Operate over scales, coarse-to-fine
  for (int sl=op.sc_f; sl>=op.sc_l; --sl)  
  {
    int ii = sl-op.sc_l;

    if (op.verbosity>1) gettimeofday(&tv_start_all, nullptr);

    // Initialize grid (Step 1 in Algorithm 1 of paper)
    grid_fw[ii]->  InitializeGrid(im_ao[sl], im_ao_dx[sl], im_ao_dy[sl]);
    grid_fw[ii]->  SetTargetImage(im_bo[sl], im_bo_dx[sl], im_bo_dy[sl]);
    if (op.usefbcon)
    {
      grid_bw[ii]->InitializeGrid(im_bo[sl], im_bo_dx[sl], im_bo_dy[sl]);
      grid_bw[ii]->SetTargetImage(im_ao[sl], im_ao_dx[sl], im_ao_dy[sl]);
    }

    // Timing, Grid construction
    if (op.verbosity>1)
    {
      gettimeofday(&tv_end_all, nullptr);
      tt_patconstr[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
      tt_all[ii] += tt_patconstr[ii];
      gettimeofday(&tv_start_all, nullptr);
    }
    
    // Initialization from previous scale, or to zero at first iteration. (Step 2 in Algorithm 1 of paper)                                          
    if (sl < op.sc_f)
    {
      grid_fw[ii]->InitializeFromCoarserOF(flow_fw[ii+1]); // initialize from flow at previous coarser scale
      
      // Initialize backward flow
      if (op.usefbcon)
        grid_bw[ii]->InitializeFromCoarserOF(flow_bw[ii+1]);
    } 
    else if (sl == op.sc_f && initflow != nullptr) // initialization given input flow
    {
      grid_fw[ii]->InitializeFromCoarserOF(initflow); // initialize from flow at coarser scale
    }

    // Timing, Grid initialization
    if (op.verbosity>1)
    {    
      gettimeofday(&tv_end_all, nullptr);
      tt_patinit[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
      tt_all[ii] += tt_patinit[ii];                                                                                                                                
      gettimeofday(&tv_start_all, nullptr);
    }      
    
    
    // Dense Inverse Search. (Step 3 in Algorithm 1 of paper)                                          
    grid_fw[ii]->Optimize();
    if (op.usefbcon)
      grid_bw[ii]->Optimize();
      
//     if (op.verbosity==4) // needed for verbosity >= 3, DISVISUAL
//     {
//       grid_fw[ii]->OptimizeAndVisualize(pow(2, sl));
//       if (op.usefbcon)
//         grid_bw[ii]->Optimize();
//     }
//     else
//     {
//       grid_fw[ii]->Optimize();
//       if (op.usefbcon)
//         grid_bw[ii]->Optimize();
//     }

    
    // Timing, DIS
    if (op.verbosity>1)
    {    
      gettimeofday(&tv_end_all, nullptr);
      tt_patoptim[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
      tt_all[ii] += tt_patoptim[ii];                                                                                                                                                                                                          
      
      gettimeofday(&tv_start_all, nullptr);
    }

                                                              
    // Densification. (Step 4 in Algorithm 1 of paper)                                                                    
    float *tmp_ptr = flow_fw[ii];
    if (sl == op.sc_l)
      tmp_ptr = outflow;
    
    grid_fw[ii]->AggregateFlowDense(tmp_ptr);
    
    if (op.usefbcon && sl > op.sc_l )  // skip at last scale, backward flow no longer needed
      grid_bw[ii]->AggregateFlowDense(flow_bw[ii]);
      
    
    // Timing, Densification
    if (op.verbosity>1)
    {    
      gettimeofday(&tv_end_all, nullptr);
      tt_compflow[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
      tt_all[ii] += tt_compflow[ii];                                                                                                                                                                                                          
      
      gettimeofday(&tv_start_all, nullptr);
    }


    // Variational refinement, (Step 5 in Algorithm 1 of paper)
    for(int i=0;i<1;i++){
    //for(int i=0;i<15;i++){
    if (op.usetvref)
    {
      OFC::VarRefClass varref_fw(im_ao[sl], im_ao_dx[sl], im_ao_dy[sl], 
                                im_bo[sl], im_bo_dx[sl], im_bo_dy[sl]
                                ,&(cpl[ii]), &(cpr[ii]), &op, tmp_ptr);
      
      if (op.usefbcon  && sl > op.sc_l )    // skip at last scale, backward flow no longer needed
          OFC::VarRefClass varref_bw(im_bo[sl], im_bo_dx[sl], im_bo_dy[sl], 
                                    im_ao[sl], im_ao_dx[sl], im_ao_dy[sl]
                                    ,&(cpr[ii]), &(cpl[ii]), &op, flow_bw[ii]);
    }
    }
    
    
    // Timing, Variational Refinement
    if (op.verbosity>1)
    {        
      gettimeofday(&tv_end_all, nullptr);
      tt_tvopt[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
      tt_all[ii] += tt_tvopt[ii];                                                                                                                                                                                                                                                                     
      printf("TIME (Sc: %i, #p:%6i, pconst, pinit, poptim, cflow, tvopt, total): %8.2f %8.2f %8.2f %8.2f %8.2f -> %8.2f ms.\n", sl, grid_fw[ii]->GetNoPatches(), tt_patconstr[ii], tt_patinit[ii], tt_patoptim[ii], tt_compflow[ii], tt_tvopt[ii], tt_all[ii]);
    }
                                                                
    // occlusion refinement
    if(false){
     //****params****//
    int window_sz=5; //9x9 sliding window
    int stride=1;
    float std_thres=0.6;
    //****load uin8 img****//
  
    int width_org=cpl[ii].width;
    int height_org=cpl[ii].height;
    Mat in_flow(cpl[ii].height,cpl[ii].width,CV_32FC2, tmp_ptr), out_flow(width_org,height_org,CV_32FC2);
    out_flow=in_flow.clone();
    vector<Mat> in_flow_split;
    split(in_flow, in_flow_split);
    //****Perturb the flow within sliding window****//
    for(int iter =0;iter<1;iter++){
        for (int col = 0; col <= width_org - window_sz;col+=stride){
            for (int row = 0; row<=height_org - window_sz;row+=stride){
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
                window_flow_u = in_flow_split[0](roi);
                window_flow_v = in_flow_split[1](roi);
                window_flow = in_flow(roi);
                meanStdDev(window_flow_u,mean_u,std_u);
                meanStdDev(window_flow_v,mean_v,std_v);
                if(std_u.at<double>(0,0) > std_thres || std_v.at<double>(0,0) >std_thres){
                    bool warp_valid = true;//warping_error(window_error, roi, window_data, window_flow_u, window_flow_v, img_bo_mat);
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
		    Point max_dx_loc,
			  max_dy_loc,
			  min_dx_loc,
			  min_dy_loc;
		    double max_dx_val,
			   max_dy_val,
			   min_dx_val,
			   min_dy_val;
            	    cartToPolar(window_flow_u,window_flow_v,window_flow_mag,window_flow_angle,true); //true= angle in degree;
            	    Sobel(window_flow_mag,window_flow_mag_dx,CV_32F,1,0,3,1,0,BORDER_DEFAULT);
            	    Sobel(window_flow_mag,window_flow_mag_dy,CV_32F,0,1,3,1,0,BORDER_DEFAULT);
            	    cartToPolar(window_flow_mag_dx,window_flow_mag_dy,window_flow_mag_grad,window_flow_mag_grad_angle,true); //true= angle in degree;
            	    float meanDirection=my_mean(window_flow_mag_grad_angle);
            	    Scalar meanGrad=cv::mean(window_flow_mag_grad);
            	    Scalar meanFlowDirection=cv::mean(window_flow_angle);
            	    Scalar meanFlowMag=cv::mean(window_flow_mag);
            	    Mat_<int> speed_vec(2,1),flow_vec(2,1);
		    minMaxLoc(window_flow_mag_dx,&min_dx_val,&max_dx_val,&min_dx_loc,&max_dx_loc);
		    minMaxLoc(window_flow_mag_dy,&min_dy_val,&max_dy_val,&min_dy_loc,&max_dy_loc);
            	    //****SPEED PATTERN****//

            	    if(22.5<= meanDirection && meanDirection< 67.5){
            		    speed_vec<<1,-1;
            	    }
            	    else if(67.5<= meanDirection && meanDirection< 112.5){
            		    speed_vec<<0,-1;
            	    }
            	    else if(112.5<= meanDirection && meanDirection< 157.5){
            		    speed_vec<<-1,-1;
            	    }
            	    else if(157.5<= meanDirection && meanDirection< 202.5){
            		    speed_vec<<-1,0;
            	    }
            	    else if(202.5<= meanDirection && meanDirection< 247.5){
            		    speed_vec<<-1,1;
            	    }
            	    else if(247.5<= meanDirection && meanDirection< 292.5){
            		    speed_vec<<0,1;
            	    }
            	    else if(292.5<= meanDirection && meanDirection< 337.5){
            		    speed_vec<<1,1;
            	    }
            	    else{
            		    speed_vec<<1,0;
            	    }
            	    //****FLOW DIRECTION****//
            	    if(23.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 67.5){
            		    flow_vec<<1,-1;
            		    
            	    }
            	    else if(67.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 112.5){
            		    flow_vec<<0,-1;
            	    }
            	    else if(112.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 157.5){
            		    flow_vec<<-1,-1;
            	    }
            	    else if(157.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 202.5){
            		    flow_vec<<-1,0;
            	    }
            	    else if(202.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 247.5){
            		    flow_vec<<-1,1;
            	    }
            	    else if(247.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 292.5){
            		    flow_vec<<0,1;
            	    }
            	    else if(292.5<= meanFlowDirection.val[0] && meanFlowDirection.val[0]< 337.5){
            		    flow_vec<<1,1;
            	    }
            	    else{
            		    flow_vec<<1,0;
            	    }
            	    //****PERTURB FLOW****//
		    float learning = 2.f;
		    float del_step = 20;
		    float offset=1;

             	    if(flow_vec.dot(speed_vec)>0)//SAME direction
            	    {
            	            int del_row  = floor(offset+1/max(abs(min_dy_val),abs(max_dy_val))*del_step),//round(std_u.at<double>(0,0)*1.5),
				del_col  = floor(offset+1/max(abs(min_dx_val),abs(max_dx_val))*del_step);//round(std_v.at<double>(0,0)*1.5);
            	            copyMakeBorder(window_flow, padded_window_flow, del_row, del_row, del_col, del_col, BORDER_REPLICATE);
            		    int col_new = del_col - del_col*speed_vec.at<int>(0,0);
            		    int row_new = del_row + del_row*speed_vec.at<int>(1,0);
			    double weight = 1/learning;
                            Rect roi_target(col_new,row_new,window_sz,window_sz);
            		    Mat test = ((weight*padded_window_flow(roi_target))+((1-weight)*out_flow(roi)));
			    //Mat blur;
            		    //GaussianBlur(test,blur,cv::Size(0,0),3);
			    //addWeighted(blur,1.5,test,-0.5,0,test);
			    test.copyTo(out_flow(roi));
            	    }
		    else if(flow_vec.dot(speed_vec)<0)//OPPOSITE Direction
            	    {
            	            int del_row  = floor(offset+1/max(abs(min_dy_val),abs(max_dy_val))*del_step),//round(std_u.at<double>(0,0)*1.5),
				del_col  = floor(offset+1/max(abs(min_dx_val),abs(max_dx_val))*del_step);//round(std_v.at<double>(0,0)*1.5);
            	            copyMakeBorder(window_flow, padded_window_flow, del_row, del_row, del_col, del_col, BORDER_REPLICATE);
            		    int col_new = del_col - del_col*speed_vec.at<int>(0,0);
            		    int row_new = del_row + del_row*speed_vec.at<int>(1,0);
			    double weight = 1/learning;
                            Rect roi_target(col_new,row_new,window_sz,window_sz);
            		    Mat test = ((weight*padded_window_flow(roi_target))+((1-weight)*out_flow(roi)));
			    //Mat blur;
            		    //GaussianBlur(test,blur,cv::Size(0,0),3);
			    //addWeighted(blur,1.5,test,-0.5,0,test);
			    test.copyTo(out_flow(roi));
            	    }
		    else{
			//cout<<"PERP dirc"<<endl;
		    }
                    }    
                }
            }    
        }
    // Copy flow result back
    for (int iy = 0; iy < height_org; iy++){
      for (int ix = 0; ix < width_org; ix++){
        int i  = iy * width_org+ ix;
        for (int j = 0; j < 2; j++){
	  //cout<<"Before: "<<tmp_ptr[i*2+j]<<endl;
          tmp_ptr[i*2 + j] = out_flow.at<Vec2f>(iy,ix)[j];
	  //cout<<"After: "<<tmp_ptr[i*2+j]<<endl;
	}
      }
    }
    
    // Variational refinement, (Step 5 in Algorithm 1 of paper)
    for(int i =0;i<15;i++){
      if (op.usetvref)
      {
        OFC::VarRefClass varref_fw(im_ao[sl], im_ao_dx[sl], im_ao_dy[sl], 
                                  im_bo[sl], im_bo_dx[sl], im_bo_dy[sl]
                                  ,&(cpl[ii]), &(cpr[ii]), &op, tmp_ptr);
        
        if (op.usefbcon  && sl > op.sc_l )    // skip at last scale, backward flow no longer needed
            OFC::VarRefClass varref_bw(im_bo[sl], im_bo_dx[sl], im_bo_dy[sl], 
                                      im_ao[sl], im_ao_dx[sl], im_ao_dy[sl]
                                      ,&(cpr[ii]), &(cpl[ii]), &op, flow_bw[ii]);
      }
    }
    Mat tmp_flow(cpl[ii].height,cpl[ii].width,CV_32FC2, tmp_ptr);
    in_flow=tmp_flow.clone();
    split(in_flow, in_flow_split);
    window_sz-=2;
    }
    for(int i =0;i<55;i++){
      if (op.usetvref)
      {
        OFC::VarRefClass varref_fw(im_ao[sl], im_ao_dx[sl], im_ao_dy[sl], 
                                  im_bo[sl], im_bo_dx[sl], im_bo_dy[sl]
                                  ,&(cpl[ii]), &(cpr[ii]), &op, tmp_ptr);
        
        if (op.usefbcon  && sl > op.sc_l )    // skip at last scale, backward flow no longer needed
            OFC::VarRefClass varref_bw(im_bo[sl], im_bo_dx[sl], im_bo_dy[sl], 
                                      im_ao[sl], im_ao_dx[sl], im_ao_dy[sl]
                                      ,&(cpr[ii]), &(cpl[ii]), &op, flow_bw[ii]);
      }
    }
    }

//     if (op.verbosity==3) // Display displacement result of this scale // needed for verbosity >= 3, DISVISUAL
//     {
//       // Display Grid on current scale
//       float sc_fct_tmp = pow(2, sl); // upscale factor
// 
//       cv::Mat src(cpl[ii].height+2*cpl[ii].imgpadding, cpl[ii].width+2*cpl[ii].imgpadding, CV_32FC1, (void*) im_ao[sl]);  
//       cv::Mat img_ao_mat = src(cv::Rect(cpl[ii].imgpadding, cpl[ii].imgpadding, cpl[ii].width, cpl[ii].height));
// 
//       cv::Mat outimg;
//       img_ao_mat.convertTo(outimg, CV_8UC1);
//       cv::cvtColor(outimg, outimg, CV_GRAY2RGB);
//       cv::resize(outimg, outimg, cv::Size(), sc_fct_tmp, sc_fct_tmp, cv::INTER_NEAREST);
//       for (int i = 0; i < grid_fw[ii]->GetNoPatches() ; ++i)
//         DisplayDrawPatchBoundary(outimg, grid_fw[ii]->GetRefPatchPos(i), sc_fct_tmp);
//                           
//       for (int i = 0; i < grid_fw[ii]->GetNoPatches(); ++i)
//       {
//         // Show displacement vector
//         const Eigen::Vector2f pt_ref = grid_fw[ii]->GetRefPatchPos(i);
//         const Eigen::Vector2f pt_ret = grid_fw[ii]->GetQuePatchPos(i);
// 
//         Eigen::Vector2f pta, ptb;
//         cv::line(outimg, cv::Point( (pt_ref[0]+.5)*sc_fct_tmp, (pt_ref[1]+.5)*sc_fct_tmp ), cv::Point( (pt_ret[0]+.5)*sc_fct_tmp, (pt_ret[1]+.5)*sc_fct_tmp ), cv::Scalar(0,255,0),  2);
//       }
//       cv::namedWindow( "Img_ao", cv::WINDOW_AUTOSIZE );
//       cv::imshow( "Img_ao", outimg);
//       
//       cv::waitKey(0);
//     }
                                                              
  }
  
  // Clean up
  for (int sl=op.sc_f; sl>=op.sc_l; --sl) 
  {                                        

    delete[] flow_fw[sl-op.sc_l];    
    delete grid_fw[sl-op.sc_l];

    if (op.usefbcon) 
    {
      delete[] flow_bw[sl-op.sc_l];
      delete grid_bw[sl-op.sc_l];
    }
  }
  
   
  // Timing, total algorithm run-time
  if (op.verbosity>0)
  {       
    gettimeofday(&tv_end_all_global, nullptr);
    double tt = (tv_end_all_global.tv_sec-tv_start_all_global.tv_sec)*1000.0f + (tv_end_all_global.tv_usec-tv_start_all_global.tv_usec)/1000.0f;
    printf("TIME (O.Flow Run-Time   ) (ms): %3g\n", tt);            
  }

  
}

// // needed for verbosity >= 3, DISVISUAL
// void OFClass::DisplayDrawPatchBoundary(cv::Mat img, const Eigen::Vector2f pt, const float sc) 
// {
//   cv::line(img, cv::Point( (pt[0]+.5)*sc, (pt[1]+.5)*sc ), cv::Point( (pt[0]+.5)*sc, (pt[1]+.5)*sc ), cv::Scalar(0,0,255),  4);
//   
//   float lb = -op.p_samp_s/2;
//   float ub = op.p_samp_s/2-1;     
//   
//   cv::line(img, cv::Point( ((pt[0]+lb)+.5)*sc, ((pt[1]+lb)+.5)*sc ), cv::Point( ((pt[0]+ub)+.5)*sc, ((pt[1]+lb)+.5)*sc ), cv::Scalar(0,0,255),  1);
//   cv::line(img, cv::Point( ((pt[0]+ub)+.5)*sc, ((pt[1]+lb)+.5)*sc ), cv::Point( ((pt[0]+ub)+.5)*sc, ((pt[1]+ub)+.5)*sc ), cv::Scalar(0,0,255),  1);
//   cv::line(img, cv::Point( ((pt[0]+ub)+.5)*sc, ((pt[1]+ub)+.5)*sc ), cv::Point( ((pt[0]+lb)+.5)*sc, ((pt[1]+ub)+.5)*sc ), cv::Scalar(0,0,255),  1);
//   cv::line(img, cv::Point( ((pt[0]+lb)+.5)*sc, ((pt[1]+ub)+.5)*sc ), cv::Point( ((pt[0]+lb)+.5)*sc, ((pt[1]+lb)+.5)*sc ), cv::Scalar(0,0,255),  1);
// }

}














