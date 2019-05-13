
#include <opencv2/core/core.hpp> // needed for verbosity >= 3, DISVISUAL
#include <opencv2/highgui/highgui.hpp> // needed for verbosity >= 3, DISVISUAL
#include <opencv2/imgproc/imgproc.hpp> // needed for verbosity >= 3, DISVISUAL

#include <iostream>
#include <string>
#include <vector>
#include <valarray>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>  
#include<cstdlib>

#include "patch.h"
#include "patchgrid.h"


using std::cout;
using std::endl;
using std::vector;
using namespace cv; 


namespace OFC
{
    
  PatGridClass::PatGridClass(
    const camparam* cpt_in,
    const camparam* cpo_in,
    const optparam* op_in)
  : 
    cpt(cpt_in),
    cpo(cpo_in),
    op(op_in)
  {

  // Generate grid on current scale
  steps = op->steps;
  nopw = ceil( (float)cpt->width /  (float)steps );
  noph = ceil( (float)cpt->height / (float)steps );
  const int offsetw = floor((cpt->width - (nopw-1)*steps)/2);
  const int offseth = floor((cpt->height - (noph-1)*steps)/2);

  nopatches = nopw*noph;
  //std::vector<Eigen::Vector2f> pt_ref; // Midpoints for reference patches
  pt_ref.resize(nopatches);
  //std::vector<Eigen::Vector2f> p_init; // starting parameters for query patches
  p_init.resize(nopatches);
  //std::vector<OFC::PatClass*> pat; // Patch Objects
  pat.reserve(nopatches);
  
  im_ao_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_ao_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_ao_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);

  im_bo_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_bo_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_bo_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);

  int patchid=0;
  for (int x = 0; x < nopw; ++x)
  {
    for (int y = 0; y < noph; ++y)
    {
      int i = x*noph + y;

      pt_ref[i][0] = x * steps + offsetw;
      pt_ref[i][1] = y * steps + offseth;
      p_init[i].setZero();
      
      pat.push_back(new OFC::PatClass(cpt, cpo, op, patchid));    
      patchid++;
    }
  }
}

PatGridClass::~PatGridClass()
{
  delete im_ao_eg;
  delete im_ao_dx_eg;
  delete im_ao_dy_eg;

  delete im_bo_eg;
  delete im_bo_dx_eg;
  delete im_bo_dy_eg;

  for (int i=0; i< nopatches; ++i)
    delete pat[i];
}

void PatGridClass::SetComplGrid(PatGridClass *cg_in)
{
  cg = cg_in;
}


void PatGridClass::InitializeGrid(const float * im_ao_in, const float * im_ao_dx_in, const float * im_ao_dy_in)
{
  im_ao = im_ao_in;
  im_ao_dx = im_ao_dx_in;
  im_ao_dy = im_ao_dy_in;
  
  new (im_ao_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao,cpt->height,cpt->width); // new placement operator
  new (im_ao_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dx,cpt->height,cpt->width);  
  new (im_ao_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dy,cpt->height,cpt->width);  
  
  
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nopatches; ++i)
  {
    pat[i]->InitializePatch(im_ao_eg, im_ao_dx_eg, im_ao_dy_eg, pt_ref[i]);
    p_init[i].setZero();    //set starting param of query patch to be zero
  }

}

void PatGridClass::SetTargetImage(const float * im_bo_in, const float * im_bo_dx_in, const float * im_bo_dy_in)
{
  im_bo = im_bo_in;
  im_bo_dx = im_bo_dx_in;
  im_bo_dy = im_bo_dy_in;
  
  new (im_bo_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo,cpt->height,cpt->width); // new placement operator
  new (im_bo_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dx,cpt->height,cpt->width); // new placement operator
  new (im_bo_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dy,cpt->height,cpt->width); // new placement operator
  
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nopatches; ++i)
    pat[i]->SetTargetImage(im_bo_eg, im_bo_dx_eg, im_bo_dy_eg);
  
}

void PatGridClass::Optimize()
{
    #pragma omp parallel for schedule(dynamic,10)
    for (int i = 0; i < nopatches; ++i)
    {
      pat[i]->OptimizeIter(p_init[i], true); // optimize until convergence  
    }
}  

// void PatGridClass::OptimizeAndVisualize(const float sc_fct_tmp) // needed for verbosity >= 3, DISVISUAL
// {
//   bool allconverged=0;
//   int cnt = 0;
//   while (!allconverged)
//   {
//     cnt++;
// 
//     allconverged=1;
// 
//     for (int i = 0; i < nopatches; ++i)
//     {
//       if (pat[i]->isConverged()==0)
//       {
//         pat[i]->OptimizeIter(p_init[i], false); // optimize, only one iterations
//         allconverged=0;
//       }
//     }
//     
// 
//     // Display original image
//     const cv::Mat src(cpt->height+2*cpt->imgpadding, cpt->width+2*cpt->imgpadding, CV_32FC1, (void*) im_ao);  
//     cv::Mat img_ao_mat = src(cv::Rect(cpt->imgpadding,cpt->imgpadding,cpt->width,cpt->height));
//     cv::Mat outimg;
//     img_ao_mat.convertTo(outimg, CV_8UC1);
//     cv::cvtColor(outimg, outimg, CV_GRAY2RGB);
//     cv::resize(outimg, outimg, cv::Size(), sc_fct_tmp, sc_fct_tmp, cv::INTER_NEAREST);
// 
//     for (int i = 0; i < nopatches; ++i)
//     {
//       // Show displacement vector
//       const Eigen::Vector2f pt_ret = pat[i]->GetPointPos();
//       
//       Eigen::Vector2f pta, ptb;
//       
//       cv::line(outimg, cv::Point( (pt_ref[i][0]+.5)*sc_fct_tmp, (pt_ref[i][1]+.5)*sc_fct_tmp ), cv::Point( (pt_ret[0]+.5)*sc_fct_tmp, (pt_ret[1]+.5)*sc_fct_tmp ), cv::Scalar(255*pat[i]->isConverged() ,255*(!pat[i]->isConverged()),0),  2);
//       
//       cv::line(outimg, cv::Point( (cpt->cx+.5)*sc_fct_tmp, (cpt->cy+.5)*sc_fct_tmp ), cv::Point( (cpt->cx+.5)*sc_fct_tmp, (cpt->cy+.5)*sc_fct_tmp ), cv::Scalar(0,0, 255),  2);
// 
//     }
// 
//     char str[200];
//     sprintf(str,"Iter: %i",cnt);
//     cv::putText(outimg, str, cv::Point2f(20,20), cv::FONT_HERSHEY_PLAIN, 1,  cv::Scalar(0,0,255,255), 2);
// 
//     cv::namedWindow( "Img_iter", cv::WINDOW_AUTOSIZE );
//     cv::imshow( "Img_iter", outimg);
//     
//     cv::waitKey(500);
//   }
// } 

void PatGridClass::InitializeFromCoarserOF(const float * flow_prev)
{
  #pragma omp parallel for schedule(dynamic,10)
  for (int ip = 0; ip < nopatches; ++ip)
  {
    int x = floor(pt_ref[ip][0] / 2); // better, but slower: use bil. interpolation here
    int y = floor(pt_ref[ip][1] / 2); 
    int i = y*(cpt->width/2) + x;
    
    #if (SELECTMODE==1)
    p_init[ip](0) = flow_prev[2*i  ]*2;
    p_init[ip](1) = flow_prev[2*i+1]*2;
    #else
    p_init[ip](0) = flow_prev[  i  ]*2;      
    #endif
  }
}

float normal_pdf(float x, float m, float s)
{
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

void PatGridClass::AggregateFlowDense(float *flowout, float * varout) const
{
  float* we = new float[cpt->width * cpt->height];
  float* u_lb = new float[cpt->width * cpt->height];
  float* u_ub = new float[cpt->width * cpt->height];
  float* v_lb = new float[cpt->width * cpt->height];
  float* v_ub = new float[cpt->width * cpt->height];
  int array_size=cpt->width * cpt->height;
  //vector<Point>* all_flow = new vector<Point> [array_size];
  vector<Point3f>* all_flow = new vector<Point3f> [array_size];
  float* point_valid_flag = new float[cpt->width * cpt->height];

  memset(flowout, 0, sizeof(float) * (op->nop * cpt->width * cpt->height) );
  memset(varout, 0, sizeof(float) * (op->nop * cpt->width * cpt->height) );
  memset(we,      0, sizeof(float) * (          cpt->width * cpt->height) );
  memset(u_lb,      0, sizeof(float) * (          cpt->width * cpt->height) );
  memset(u_ub,      0, sizeof(float) * (          cpt->width * cpt->height) );
  memset(v_lb,      0, sizeof(float) * (          cpt->width * cpt->height) );
  memset(v_ub,      0, sizeof(float) * (          cpt->width * cpt->height) );
  memset(point_valid_flag,      1, sizeof(float) * (          cpt->width * cpt->height) );
  #ifdef USE_PARALLEL_ON_FLOWAGGR // Using this enables OpenMP on flow aggregation. This can lead to race conditions. Experimentally we found that the result degrades only marginally. However, for our experiments we did not enable this.
    #pragma omp parallel for schedule(static)  
  #endif
  for (int ip = 0; ip < nopatches; ++ip)
  {       
    
    if (pat[ip]->IsValid())
    {
      #if (SELECTMODE==1)
      const Eigen::Vector2f*            fl = pat[ip]->GetParam(); // flow displacement of this patch
      Eigen::Vector2f flnew;
      #else
      const Eigen::Matrix<float, 1, 1>* fl = pat[ip]->GetParam(); // horz. displacement of this patch
      Eigen::Matrix<float, 1, 1> flnew;
      #endif
      
      const float * pweight = pat[ip]->GetpWeightPtr(); // use image error as weight
      int lb = -op->p_samp_s/2;
      int ub = op->p_samp_s/2-1;
      float patch_homo_weight=(float) (std::max(std::sqrt(pat[ip]->GetpVar()),op->minerrval));
      
      for (int y = lb; y <= ub; ++y)
      {   
        for (int x = lb; x <= ub; ++x, ++pweight)
        {
          int yt = (y + pt_ref[ip][1]);
          int xt = (x + pt_ref[ip][0]);

          if (xt >= 0 && yt >= 0 && xt < cpt->width && yt < cpt->height)
          {
  
            int i = yt*cpt->width + xt;
              
            //****float minerrval = 2.0f;       // 1/max(this, error) for pixel averaging weight
            #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // single channel/gradient image 
            float absw = 1.0f /  (float)(std::max(op->minerrval  ,*pweight));
            #else  // RGB image
            float absw = (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                  absw+= (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                  absw+= (float)(std::max(op->minerrval  ,*pweight));
            absw = 1.0f / (patch_homo_weight);
            //absw = 1.0f / absw;
            #endif
              
            flnew = (*fl) * absw;
            we[i] += absw;
	    //if((*fl)[0]!=0 && (*fl)[1]!=0){
		    //cout<<"fl[0]"<<fl[0]<<endl;
		    //cout<<"fl[1]"<<fl[1]<<endl;
                all_flow[i].push_back(Point3f((*fl)[0],(*fl)[1],1.0f/absw));
            //patch_error_std[i]=patch_homo_weight;
	    //}
            #if (SELECTMODE==1)
            flowout[2*i]   += flnew[0];
            flowout[2*i+1] += flnew[1];
            #else
            flowout[i] += flnew[0]; 
            #endif
          }
        }
      }
    }
  }
 
  
  // if complementary (forward-backward merging) is given, integrate negative backward flow as well
  if (cg)
  {  
      Eigen::Vector4f wbil; // bilinear weight vector
      Eigen::Vector4i pos;
      
      #ifdef USE_PARALLEL_ON_FLOWAGGR
        #pragma omp parallel for schedule(static)  
      #endif    
      for (int ip = 0; ip < cg->nopatches; ++ip)
      {
        if (cg->pat[ip]->IsValid())
        {
          #if (SELECTMODE==1)
          const Eigen::Vector2f*            fl = (cg->pat[ip]->GetParam()); // flow displacement of this patch
          Eigen::Vector2f flnew;
          #else
          const Eigen::Matrix<float, 1, 1>* fl = (cg->pat[ip]->GetParam()); // horz. displacement of this patch
          Eigen::Matrix<float, 1, 1> flnew;
          #endif
        
          const Eigen::Vector2f rppos = cg->pat[ip]->GetPointPos(); // get patch position after optimization
          const float * pweight = cg->pat[ip]->GetpWeightPtr(); // use image error as weight
          
          Eigen::Vector2f resid;

          // compute bilinear weight vector
          pos[0] = ceil(rppos[0] +.00001); // make sure they are rounded up to natural number
          pos[1] = ceil(rppos[1] +.00001); // make sure they are rounded up to natural number
          pos[2] = floor(rppos[0]);
          pos[3] = floor(rppos[1]);

          resid[0] = rppos[0] - pos[2];
          resid[1] = rppos[1] - pos[3];
          wbil[0] = resid[0]*resid[1];
          wbil[1] = (1-resid[0])*resid[1];
          wbil[2] = resid[0]*(1-resid[1]);
          wbil[3] = (1-resid[0])*(1-resid[1]);

          int lb = -op->p_samp_s/2;
          int ub = op->p_samp_s/2-1;

          
          for (int y = lb; y <= ub; ++y)
          {   
            for (int x = lb; x <= ub; ++x, ++pweight)
            {
          
              int yt = y + pos[1]; 
              int xt = x + pos[0];
              if (xt >= 1 && yt >= 1 && xt < (cpt->width-1) && yt < (cpt->height-1))
              {
                
                #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // single channel/gradient image
                float absw = 1.0f /  (float)(std::max(op->minerrval  ,*pweight));
                #else  // RGB
                float absw = (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                      absw+= (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                      absw+= (float)(std::max(op->minerrval  ,*pweight));
                //absw = 1.0f / (absw + patch_homo_weight);
                absw = 1.0f / absw;
                #endif
              
              
                flnew = (*fl) * absw;
                
                int idxcc =  xt    +  yt   *cpt->width;
                int idxfc = (xt-1) +  yt   *cpt->width;
                int idxcf =  xt    + (yt-1)*cpt->width;
                int idxff = (xt-1) + (yt-1)*cpt->width;            
        
                we[idxcc] += wbil[0] * absw;
                we[idxfc] += wbil[1] * absw;
                we[idxcf] += wbil[2] * absw;
                we[idxff] += wbil[3] * absw;

                #if (SELECTMODE==1)
                flowout[2*idxcc  ] -= wbil[0] * flnew[0];   // use reversed flow 
                flowout[2*idxcc+1] -= wbil[0] * flnew[1];

                flowout[2*idxfc  ] -= wbil[1] * flnew[0];
                flowout[2*idxfc+1] -= wbil[1] * flnew[1];

                flowout[2*idxcf  ] -= wbil[2] * flnew[0];
                flowout[2*idxcf+1] -= wbil[2] * flnew[1];

                flowout[2*idxff  ] -= wbil[3] * flnew[0];
                flowout[2*idxff+1] -= wbil[3] * flnew[1];
                #else
                flowout[idxcc] -= wbil[0] * flnew[0]; // simple averaging of inverse horizontal displacement
                flowout[idxfc] -= wbil[1] * flnew[0];
                flowout[idxcf] -= wbil[2] * flnew[0];
                flowout[idxff] -= wbil[3] * flnew[0];
                #endif
              }
            }
          }
        }
      }
  } 
  
  #pragma omp parallel for schedule(static, 100)    
  // normalize each pixel by dividing displacement by aggregated weights from all patches
  for (int yi = 0; yi < cpt->height; ++yi)
  {
    for (int xi = 0; xi < cpt->width; ++xi)
    { 
      int i    = yi*cpt->width + xi;
      if (we[i]>0)
      {
        #if (SELECTMODE==1)        
        flowout[2*i  ] /= we[i];
        flowout[2*i+1] /= we[i];
        #else
        flowout[i] /= we[i];
        #endif
      }
    }
  }
  if(false && cpt->width == 1024){
	    float conflict_cnt=0;
    for (int j =0;j<cpt->height;j++){
	    for(int i =0;i<cpt->width;i++){
	        int index=j*cpt->width+i;
		//cout<<"At location x="<<i<<",y="<<j<<", the number of flow candidates: "<<all_flow[index].size()<<endl;
                Size sz;
		sz.height=all_flow[index].size();
		sz.width=1;
		Mat all_u(sz,CV_32FC1);
	        Mat all_v(sz,CV_32FC1); 
	        Mat all_error_std(sz,CV_32FC1); 
		for(unsigned f=0;f<all_flow[index].size();f++){
			//cout<<"\t"<<all_flow[index][f]<<endl;
			all_u.at<float>(f,0)=all_flow[index][f].x / cpt->sc_fct;
			all_v.at<float>(f,0)=all_flow[index][f].y / cpt->sc_fct;
			all_error_std.at<float>(f,0)=all_flow[index][f].z ;
		}
		if(all_flow[index].size()==0){
                    point_valid_flag[index]=0;  
		    float margin=3;
                    varout[2*index]   = margin*margin;
                    varout[2*index+1] = margin*margin;
		} 
		else if(all_flow[index].size()<=4){
		    float margin=1;
		    u_lb[index]=flowout[2*index]   - 1 * margin;
		    u_ub[index]=flowout[2*index]   + 1 * margin;
		    v_lb[index]=flowout[2*index+1] - 1 * margin;
		    v_ub[index]=flowout[2*index+1] + 1 * margin;
                    varout[2*index]   = margin*margin;
                    varout[2*index+1] = margin*margin;
		}
		else{
		    double min_u, max_u, min_v, max_v;
		    Mat mean_u, std_u, mean_v, std_v;
		    Point min_u_loc,max_u_loc,min_v_loc,max_v_loc;
		    minMaxLoc(all_u,&min_u,&max_u,&min_u_loc,&max_u_loc);
		    minMaxLoc(all_v,&min_v,&max_v,&min_v_loc,&max_v_loc);
		    meanStdDev(all_u,mean_u,std_u);
		    meanStdDev(all_v,mean_v,std_v);
	    //if(i==20){// && (varout[2*index]<0.25 && varout[2*index+1]<0.25)){
	            //cout<<"At location x="<<i<<",y="<<j<<", the number of flow candidates: "<<all_flow[index].size()<<endl;
	            //cout<<"std_u:"<<(float) std_u.at<double>(0,0)<<endl;
	            //cout<<"std_v:"<<(float) std_v.at<double>(0,0)<<endl;
		    //cout<<"patch error std:"<<endl<<all_error_std<<endl;
	            //cout<<"u:"<<endl<<all_u<<endl;
	            //cout<<"v:"<<endl<<all_v<<endl;
	            //cout<<"max u:"<<max_u<<", min u:"<<min_u<<endl;
	            //cout<<"max v:"<<max_v<<", min v:"<<min_v<<endl;
	            //cout<<"decided u:"<<flowout[2*index]  / cpt->sc_fct<<endl;
	            //cout<<"decided v:"<<flowout[2*index+1]/ cpt->sc_fct<<endl;
	    //}
		    u_lb[index]=flowout[2*index]   - 1 * (float) std_u.at<double>(0,0);
		    u_ub[index]=flowout[2*index]   + 1 * (float) std_u.at<double>(0,0);
		    v_lb[index]=flowout[2*index+1] - 1 * (float) std_v.at<double>(0,0);
		    v_ub[index]=flowout[2*index+1] + 1 * (float) std_v.at<double>(0,0);
                    varout[2*index]   = (float) std_u.at<double>(0,0);//*(float) std_u.at<double>(0,0);
                    varout[2*index+1] = (float) std_v.at<double>(0,0);//*(float) std_v.at<double>(0,0);
		}

		//if((float) patch_error_std[index]>5.0){
		//    //cout<<"At location x="<<i<<",y="<<j<<", the number of flow candidates: "<<all_flow[index].size()<<endl;
		//    //cout<<"std_u:"<<(float) std_u.at<double>(0,0)<<endl;
		//    //cout<<"std_v:"<<(float) std_v.at<double>(0,0)<<endl;
		//    //cout<<"u:"<<all_u<<endl;
		//    //cout<<"v:"<<all_v<<endl;
		//    //cout<<"decided u:"<<flowout[2*index]  / cpt->sc_fct<<endl;
		//    //cout<<"decided v:"<<flowout[2*index+1]/ cpt->sc_fct<<endl;
                //    flowout[2*index]   = 0;
                //    flowout[2*index+1] = 0;
		//}
	    }
    }
    // bilateral param
    int k_size = 4;
    int half_k=4;
    float sigma=1;
    float mu=0;
    float thres=0.5;
if(false){
    // row-wise bilateral
    for (int j =0;j<cpt->height;j++){
	//left to right
        for(int i =k_size;i<cpt->width-k_size;i++){
	    float smoothed_u=0;
	    float weight_u=0;
	    float smoothed_v=0;
	    float weight_v=0;
            int index_p=j*cpt->width+i;
	    for(int k=-half_k;k<=half_k;k++){
                int index_q=index_p+k;
		float G_spatial=normal_pdf(k,mu,sigma);
		//float G_value_u=normal_pdf(flowout[2*index],mu,sigma);
		float G_std_u=1;
		//float G_value_v=normal_pdf(flowout[2*index],mu,sigma);
		float G_std_v=1;
		if(varout[2*index_q]>thres){
		    G_std_u=0.001;//normal_pdf(10,mu,sigma);
		}
		else{
	        G_std_u=normal_pdf(varout[2*index_q],mu,sigma);
		}
		if(varout[2*index_q+1]>thres){
		    G_std_v=normal_pdf(10,mu,sigma);
		}
		else{
	            G_std_v=0.001;//normal_pdf(varout[2*index_q+1],mu,sigma);
		}
		weight_u += G_spatial*G_std_u;
		weight_v += G_spatial*G_std_v;
		smoothed_u += flowout[2*index_q]  *G_spatial*G_std_u;
		smoothed_v += flowout[2*index_q+1]*G_spatial*G_std_v;
	    }
	    if(weight_u!=0)
	    flowout[2*index_p] =smoothed_u/weight_u;
	    if(weight_v!=0)
	    flowout[2*index_p+1] =smoothed_v/weight_v;

        }
	//right to left
	//for(int i = cpt->width-half_k-1;i>=half_k;i--){
	//    float smoothed_u=0;
	//    float weight_u=0;
	//    float smoothed_v=0;
	//    float weight_v=0;
        //    int index_p=j*cpt->width+i;
	//    for(int k=-half_k;k<=half_k;k++){
        //        int index_q=index_p+k;
	//	float G_spatial=normal_pdf(k,mu,sigma);
	//	//float G_value_u=normal_pdf(flowout[2*index],mu,sigma);
	//	float G_std_u;
	//	//float G_value_v=normal_pdf(flowout[2*index],mu,sigma);
	//	float G_std_v;
	//	if(varout[2*index_q]>thres){
	//	    G_std_u=normal_pdf(10,mu,sigma);
	//	}
	//	else{
	//            G_std_u=normal_pdf(varout[2*index_q],mu,sigma);
	//	}
	//	if(varout[2*index_q+1]>thres){
	//	    G_std_v=normal_pdf(10,mu,sigma);
	//	}
	//	else{
	//            G_std_v=normal_pdf(varout[2*index_q+1],mu,sigma);
	//	}
	//	weight_u += G_spatial*G_std_u;
	//	weight_v += G_spatial*G_std_v;
	//	smoothed_u += flowout[2*index_q]  *G_spatial*G_std_u;
	//	smoothed_v += flowout[2*index_q+1]*G_spatial*G_std_v;
	//    }
	//    flowout[2*index_p] =smoothed_u/weight_u;
	//    flowout[2*index_p+1] =smoothed_v/weight_v;
	//}
     }

    // col-wise bilateral
    for (int j =k_size;j<cpt->width-k_size;j++){
        //top to bottom
        for(int i =0+half_k;i<cpt->height-half_k;i++){
            float smoothed_u=0;
            float weight_u=0;
            float smoothed_v=0;
            float weight_v=0;
            int index_p=i*cpt->width+j;
            for(int k=-half_k;k<=half_k;k++){
                int index_q=index_p+k;
        	float G_spatial=normal_pdf(k,mu,sigma);
        	//float G_value_u=normal_pdf(flowout[2*index],mu,sigma);
        	float G_std_u;
        	//float G_value_v=normal_pdf(flowout[2*index],mu,sigma);
        	float G_std_v;
        	if(varout[2*index_q]>thres){
        	    G_std_u=normal_pdf(10,mu,sigma);
        	}
        	else{
                    G_std_u=normal_pdf(varout[2*index_q],mu,sigma);
        	}
        	if(varout[2*index_q+1]>thres){
        	    G_std_v=normal_pdf(10,mu,sigma);
        	}
        	else{
                    G_std_v=normal_pdf(varout[2*index_q+1],mu,sigma);
        	}
        	weight_u += G_spatial*G_std_u;
        	weight_v += G_spatial*G_std_v;
        	smoothed_u += flowout[2*index_q]  *G_spatial*G_std_u;
        	smoothed_v += flowout[2*index_q+1]*G_spatial*G_std_v;
            }
	    if(weight_u!=0)
            flowout[2*index_p] =smoothed_u/weight_u;
	    if(weight_v!=0)
            flowout[2*index_p+1] =smoothed_v/weight_v;

        }
        //bottom to top
        //for(int i = cpt->height-half_k-1;i>=half_k;i--){
        //    float smoothed_u=0;
        //    float weight_u=0;
        //    float smoothed_v=0;
        //    float weight_v=0;
        //    int index_p=i*cpt->width+j;
        //    for(int k=-half_k;k<=half_k;k++){
        //        int index_q=index_p+k;
        //	float G_spatial=normal_pdf(k,mu,sigma);
        //	//float G_value_u=normal_pdf(flowout[2*index],mu,sigma);
        //	float G_std_u;
        //	//float G_value_v=normal_pdf(flowout[2*index],mu,sigma);
        //	float G_std_v;
        //	if(varout[2*index_q]>thres){
        //	    G_std_u=normal_pdf(10,mu,sigma);
        //	}
        //	else{
        //            G_std_u=normal_pdf(varout[2*index_q],mu,sigma);
        //	}
        //	if(varout[2*index_q+1]>thres){
        //	    G_std_v=normal_pdf(10,mu,sigma);
        //	}
        //	else{
        //            G_std_v=normal_pdf(varout[2*index_q+1],mu,sigma);
        //	}
        //	weight_u += G_spatial*G_std_u;
        //	weight_v += G_spatial*G_std_v;
        //	smoothed_u += flowout[2*index_q]  *G_spatial*G_std_u;
        //	smoothed_v += flowout[2*index_q+1]*G_spatial*G_std_v;
        //    }
        //    flowout[2*index_p] =smoothed_u/weight_u;
        //    flowout[2*index_p+1] =smoothed_v/weight_v;


        //}
     }

}
    //cout<<"no. of conflicting px: "<<conflict_cnt<<", "<<conflict_cnt/(cpt->width*cpt->height)<<" of total px."<<endl;
  }
    
  delete[] we;
}

}

