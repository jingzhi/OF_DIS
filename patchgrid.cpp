
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

void PatGridClass::AggregateFlowDense(float *flowout, float * varout) const
{
  float* we = new float[cpt->width * cpt->height];
  int array_size=cpt->width * cpt->height;
  //vector<Point>* all_flow = new vector<Point> [array_size];
  vector<Point2f>* all_flow = new vector<Point2f> [array_size];
  float* point_conflict_flag = new float[cpt->width * cpt->height];

  memset(flowout, 0, sizeof(float) * (op->nop * cpt->width * cpt->height) );
  memset(varout, 0, sizeof(float) * (op->nop * cpt->width * cpt->height) );
  memset(we,      0, sizeof(float) * (          cpt->width * cpt->height) );
  memset(point_conflict_flag,      0, sizeof(float) * (          cpt->width * cpt->height) );
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
      float patch_homo_weight=std::sqrt(pat[ip]->GetpVar());
      
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
            //absw = 1.0f / (absw + patch_homo_weight);
            absw = 1.0f / absw;
            #endif
              
            flnew = (*fl) * absw;
            we[i] += absw;
	    if((*fl)[0]!=0 && (*fl)[1]!=0){
		    //cout<<"fl[0]"<<fl[0]<<endl;
		    //cout<<"fl[1]"<<fl[1]<<endl;
                all_flow[i].push_back(Point2f((*fl)[0],(*fl)[1]));
	    }
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
          float patch_homo_weight=std::sqrt(pat[ip]->GetpVar());

          
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
  if(true || cpt->width == 1024){
	    float conflict_cnt=0;
    for (int j =0;j<cpt->height;j++){
	    for(int i =0;i<cpt->width;i++){
	        int index=j*cpt->width+i;
		//cout<<"At location x="<<i<<",y="<<j<<", the number of flow candidates: "<<all_flow[index].size()<<endl;
		//
                Size sz;
		sz.height=all_flow[index].size();
		sz.width=1;
		Mat all_u(sz,CV_32FC1);
	        Mat all_v(sz,CV_32FC1); 
		for(unsigned f=0;f<all_flow[index].size();f++){
			//cout<<"\t"<<all_flow[index][f]<<endl;
			all_u.at<float>(f,0)=all_flow[index][f].x / cpt->sc_fct;
			all_v.at<float>(f,0)=all_flow[index][f].y / cpt->sc_fct;
		}
		double min_u, max_u, min_v, max_v;
		Mat mean_u, std_u, mean_v, std_v;
		Point min_u_loc,max_u_loc,min_v_loc,max_v_loc;
		minMaxLoc(all_u,&min_u,&max_u,&min_u_loc,&max_u_loc);
		minMaxLoc(all_v,&min_v,&max_v,&min_v_loc,&max_v_loc);
		vector<Mat> merge_vec={all_u,all_v};
	        Mat merged_flow;
		merge(merge_vec,merged_flow);
		//cout<<"u:"<<all_u<<endl;
		//cout<<"v:"<<all_v<<endl;
		//cout<<"merged = "<<endl<<endl<<merged_flow<<endl;
		Mat hist;
	        double mode;
		Point mode_loc;
		//int histSize[]={(int)max_u-(int)min_u+1,(int)max_v-(int)min_v+1};
		//float u_range[]={(float)min_u,(float)max_u+1};
		//float v_range[]={(float)min_v,(float)max_v+1};
		int histSize[]={320,320};
		float u_range[]={floor(min_u),ceil(max_u)};
		float v_range[]={floor(min_v),ceil(max_v)};
		const float* histRange[]={u_range, v_range};
		int channels[]={0,1};
                //calcHist(&merged_flow,1,channels,Mat(),hist,2,histSize,histRange,true,false);
		//minMaxLoc(hist,0,&mode,0,&mode_loc);

		float mode_u, mode_v;
		mode_u=((u_range[1]-u_range[0])/320)*mode_loc.y+u_range[0];
		mode_v=((v_range[1]-v_range[0])/320)*mode_loc.x+v_range[0];
		meanStdDev(all_u,mean_u,std_u);
		meanStdDev(all_v,mean_v,std_v);
		//cout<<"max u:"<<max_u<<", min u:"<<min_u<<endl;
		//cout<<"max v:"<<max_v<<", min v:"<<min_v<<endl;
		//cout<<"mode:"<<mode<<"; location:"<<mode_loc<<endl;

		if((float) std_u.at<double>(0,0)>5 || (float) std_v.at<double>(0,0) >5){
		    //cout<<"Conflict at location x="<<i<<",y="<<j<<", the number of flow candidates: "<<all_flow[index].size()<<endl;
		    //cout<<"u:"<<all_u<<endl;
		    //cout<<"v:"<<all_v<<endl;
                    //flowout[2*index]   = 0;//(0.5*(double) rand()/(double) RAND_MAX+0.75);
                    //flowout[2*index+1] = 0;//(0.5*(double) rand()/(double) RAND_MAX+0.75);
                   //if(cpt->width == 1024){
                   // flowout[2*index]   = 0;
                   // flowout[2*index+1] = 0;
		   // }
		}
		//if(false||max_u-min_u > 8 || max_v-min_v > 8 ){
		if((max_u-min_u > 8 && (max_u-mean_u.at<double>(0,0)>3 || mean_u.at<double>(0,0)-min_u>5)) || (max_v-min_v >8 && (max_v-mean_v.at<double>(0,0)>5 || mean_v.at<double>(0,0)-min_v>5))){
                    point_conflict_flag[index]=1;   
		    conflict_cnt+=1;
		    //cout<<"Conflict at location x="<<i<<",y="<<j<<", the number of flow candidates: "<<all_flow[index].size()<<endl;
		    //cout<<"max u:"<<max_u<<", min u:"<<min_u<<endl;
		    //cout<<"max v:"<<max_v<<", min v:"<<min_v<<endl;
                    varout[2*index]   = (float) std_u.at<double>(0,0)*(float) std_u.at<double>(0,0);
                    varout[2*index+1] = (float) std_v.at<double>(0,0)*(float) std_v.at<double>(0,0);
		    //if(varout[2*index]>2 || varout[2*index]>2){
		    //cout<<"var u:"<<varout[2*index]  <<endl;
		    //cout<<"var v:"<<varout[2*index+1]<<endl;
                   //if(cpt->width == 1024){
                    //flowout[2*index]   = (0.5*(double) rand()/(double) RAND_MAX+0.75);
                    //flowout[2*index+1] = (0.5*(double) rand()/(double) RAND_MAX+0.75);
		   //}
		    //}
		}
                //cout<<"org u"<<flowout[2*index]<<",org v:"<<flowout[2*index+1]<<endl;
		//cout<<"mode u:"<<mode_u<<", mode v:"<<mode_v<<endl;
		//cout<<"mean u:"<<mode_u<<", mean v:"<<mode_v<<endl;
	    }
    }
    //cout<<"no. of conflicting px: "<<conflict_cnt<<", "<<conflict_cnt/(cpt->width*cpt->height)<<" of total px."<<endl;
    }
    
  delete[] we;
}

}


