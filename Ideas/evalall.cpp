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

const String keys = "{help h usage ? |      | print this message   }"
        "{@image1        |      | image1               }"
        "{@image2        |      | image2               }"
        "{@algorithm     |      | [farneback, simpleflow, tvl1, deepflow, sparsetodenseflow, pcaflow, DISflow_ultrafast, DISflow_fast, DISflow_medium] }"
        "{@groundtruth   |      | path to the .flo file  (optional), Middlebury format }"
        "{m measure      |endpoint| error measure - [endpoint or angular] }"
        "{r region       |all   | region to compute stats about [all, discontinuities, untextured, smalld,central] }"
        "{d display      |      | display additional info images (pauses program execution) }"
        "{g gpu          |      | use OpenCL}"
        "{prior          |      | path to a prior file for PCAFlow}";

inline bool isFlowCorrect( const Point2f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9);
}
inline bool isFlowCorrect( const Point3f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && !cvIsNaN(u.z) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9)
            && (fabs(u.z) < 1e9);
}
static Mat endpointError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    Mat result(flow1.size(), CV_32FC1);
    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1 = flow1(i, j);
            const Point2f u2 = flow2(i, j);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
            {
                const Point2f diff = u1 - u2;
                result.at<float>(i, j) = sqrt((float)diff.ddot(diff)); //distance
            } else
                result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
        }
    }
    return result;
}
static Mat angularError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    Mat result(flow1.size(), CV_32FC1);

    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1_2d = flow1(i, j);
            const Point2f u2_2d = flow2(i, j);
            const Point3f u1(u1_2d.x, u1_2d.y, 1);
            const Point3f u2(u2_2d.x, u2_2d.y, 1);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
                result.at<float>(i, j) = acos((float)(u1.ddot(u2) / norm(u1) * norm(u2)));
            else
                result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
        }
    }
    return result;
}
// what fraction of pixels have errors higher than given threshold?
static float stat_RX( Mat errors, float threshold, Mat mask )
{
    CV_Assert(errors.size() == mask.size());
    CV_Assert(mask.depth() == CV_8U);

    int count = 0, all = 0;
    for ( int i = 0; i < errors.rows; ++i )
    {
        for ( int j = 0; j < errors.cols; ++j )
        {
            if ( mask.at<char>(i, j) != 0 )
            {
                ++all;
                if ( errors.at<float>(i, j) > threshold )
                    ++count;
            }
        }
    }
    return (float)count / all;
}
static float stat_AX( Mat hist, int cutoff_count, float max_value )
{
    int counter = 0;
    int bin = 0;
    int bin_count = hist.rows;
    while ( bin < bin_count && counter < cutoff_count )
    {
        counter += (int) hist.at<float>(bin, 0);
        ++bin;
    }
    return (float) bin / bin_count * max_value;
}
static void calculateStats( Mat errors, Mat mask = Mat(), bool display_images = false )
{
    float R_thresholds[] = { 0.f, 0.5f, 1.f, 2.f, 4.f, 8.f}; 
    vector<float> R_thresholds_vec = { 0.f, 0.5f, 1.f, 2.f, 4.f, 8.f};
    float A_thresholds[] = { 0.5f, 0.75f, 0.95f };
    if ( mask.empty() )
        mask = Mat::ones(errors.size(), CV_8U);
    CV_Assert(errors.size() == mask.size());
    CV_Assert(mask.depth() == CV_8U);

    //displaying the mask
    //if(display_images)
    //{
    //    namedWindow( "Region mask", WINDOW_AUTOSIZE );
    //    imshow( "Region mask", mask );
    //}

    //mean and std computation
    Scalar s_mean, s_std;
    float mean, std;
    meanStdDev(errors, s_mean, s_std, mask);
    mean = (float)s_mean[0];
    std = (float)s_std[0];
    printf("Average: %.2f\nStandard deviation: %.2f\n", mean, std);

    //RX stats - displayed in percent
    int R_thresholds_count = sizeof(R_thresholds) / sizeof(float);
    float R;
    vector<float> R_vec;
    for ( int i = 0; i < R_thresholds_count; ++i )
    {
        R = stat_RX(errors, R_thresholds[i], mask);
	R_vec.push_back(R*100);
        printf("R%.1f: %.2f%%\n", R_thresholds[i], R * 100);
    }
    if(display_images){
    //plt::scatter(R_thresholds_vec,R_vec,3);
    //plt::show();
    }
 
    //AX stats
    double max_value;
    minMaxLoc(errors, NULL, &max_value, NULL, NULL, mask);

    Mat hist;
    const int n_images = 1;
    const int channels[] = { 0 };
    const int n_dimensions = 1;
    const int hist_bins[] = { 1024 };
    const float iranges[] = { 0, (float) max_value };
    const float* ranges[] = { iranges };
    const bool uniform = true;
    const bool accumulate = false;
    calcHist(&errors, n_images, channels, mask, hist, n_dimensions, hist_bins, ranges, uniform,
            accumulate);
    int all_pixels = countNonZero(mask);
    int cutoff_count;
    float A;
    int A_thresholds_count = sizeof(A_thresholds) / sizeof(float);
    for ( int i = 0; i < A_thresholds_count; ++i )
    {
        cutoff_count = (int) (floor(A_thresholds[i] * all_pixels + 0.5f));
        A = stat_AX(hist, cutoff_count, (float) max_value);
        printf("A%.2f: %.2f\n", A_thresholds[i], A);
    }
}

static Mat flowToDisplay(const Mat flow)
{
    Mat flow_split[2];
    Mat magnitude, angle;
    Mat hsv_split[3], hsv, rgb;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
	threshold(magnitude,magnitude,1,0,2);
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
    //hue,saturation,color
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
}
static void flowToOut(const Mat flow)
{
    Mat flow_split[2];
    split(flow, flow_split);
    cout<<"Mat u ="<<endl<<endl<<flow_split[0]<<endl;
    cout<<"Mat v ="<<endl<<endl<<flow_split[1]<<endl;
    return ;
}
//  gt_path calc_path 
int main( int argc, char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV optical flow evaluation app");
    if ( parser.has("help") || argc < 3 )
    {
        parser.printMessage();
        printf("EXAMPLES:\n");
        printf("./example_optflow_optical_flow_evaluation im1.png im2.png farneback -d \n");
        printf("\t - compute flow field between im1 and im2 with farneback's method and display it");
        printf("./example_optflow_optical_flow_evaluation im1.png im2.png simpleflow groundtruth.flo \n");
        printf("\t - compute error statistics given the groundtruth; all pixels, endpoint error measure");
        printf("./example_optflow_optical_flow_evaluation im1.png im2.png farneback groundtruth.flo -m=angular -r=untextured \n");
        printf("\t - as before, but with changed error measure and stats computed only about \"untextured\" areas");
        printf("\n\n Flow file format description: http://vision.middlebury.edu/flow/code/flow-code/README.txt\n\n");
        return 0;
    }
    String groundtruth_path = parser.get<String>(0);
    String flow_path = parser.get<String>(1);
    String error_measure = parser.get<String>("measure");
    String region = parser.get<String>("region");
    bool display_images = parser.has("display");
    const bool useGpu = parser.has("gpu");

    if ( !parser.check() )
    {
        parser.printErrors();
        return 0;
    }
    cv::ocl::setUseOpenCL(useGpu);
    //printf("OpenCL Enabled: %u\n", useGpu && cv::ocl::haveOpenCL());

    Mat_<Point2f> flow, ground_truth;
    Mat computed_errors;
    if ( !flow_path.empty() )
    {
        flow = readOpticalFlow(flow_path);
    }
    if(display_images)
    {
        Mat flow_image = flowToDisplay(flow);
	//flowToOut(flow);
        namedWindow( "Computed flow", WINDOW_AUTOSIZE );
        imshow( "Computed flow", flow_image );
    }

    if ( !groundtruth_path.empty() )
    { // compare to ground truth
        ground_truth = readOpticalFlow(groundtruth_path);
        if ( flow.size() != ground_truth.size() || flow.channels() != 2
                || ground_truth.channels() != 2 )
        {
            printf("Dimension mismatch between the computed flow and the provided ground truth\n");
            return -1;
        }
        if ( error_measure == "endpoint" )
            computed_errors = endpointError(flow, ground_truth);
        else if ( error_measure == "angular" )
            computed_errors = angularError(flow, ground_truth);
        else
        {
            printf("Invalid error measure! Available options: endpoint, angular\n");
            return -1;
        }

        Mat mask;
        int valid_count;
        if( region == "all" )
            mask = Mat::ones(ground_truth.size(), CV_8U) * 255;
        else if ( region == "discontinuities" )
        {
            Mat truth_merged, grad_x, grad_y, gradient;
            vector<Mat> truth_split;
            split(ground_truth, truth_split);
            truth_merged = truth_split[0] + truth_split[1];

            Sobel( truth_merged, grad_x, CV_16S, 1, 0, -1, 1, 0, BORDER_REPLICATE );
            grad_x = abs(grad_x);
            Sobel( truth_merged, grad_y, CV_16S, 0, 1, 1, 1, 0, BORDER_REPLICATE );
            grad_y = abs(grad_y);
            addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient); //approximation!

            Scalar s_mean;
            s_mean = mean(gradient);
            double threshold = s_mean[0]; // threshold value arbitrary
            mask = gradient > threshold;
            dilate(mask, mask, Mat::ones(9, 9, CV_8U));
        }
	    else if(region == "smalld")
	    {
                    vector<Mat> truth_split;
	    	Mat magnitude,angle;
                    split(ground_truth, truth_split);
                    cartToPolar(truth_split[0], truth_split[1], magnitude, angle, true);
            mask = Mat::zeros(ground_truth.size(), CV_8U) * 255;
	    	mask = (magnitude <= 8);
	    	valid_count=countNonZero(mask);
	    	if (valid_count==0){
                        float R_thresholds[] = { 0.f, 0.5f, 1.f, 2.f, 4.f, 8.f}; 
                        vector<float> R_thresholds_vec = { 0.f, 0.5f, 1.f, 2.f, 4.f, 8.f};
                        float A_thresholds[] = { 0.5f, 0.75f, 0.95f };
                        int R_thresholds_count = sizeof(R_thresholds) / sizeof(float);
                        int A_thresholds_count = sizeof(A_thresholds) / sizeof(float);
                        printf("Using %s error measure\n", error_measure.c_str());
                        printf("Average: %.2f\nStandard deviation: %.2f\n", 0, 0);
                        for ( int i = 0; i < R_thresholds_count; ++i ){
                            printf("R%.1f: %.2f%%\n", R_thresholds[i], 0 * 100);
                            	}
                        for ( int i = 0; i < A_thresholds_count; ++i ){
                            printf("A%.2f: %.2f\n", A_thresholds[i], 0);
                        }
                        printf("Valid region: %d \n", valid_count);
	    	}
	    	//cout<<"GT mag ="<<endl<<endl<<magnitude;
	    }
		else if(region == "central")
	    {
            vector<Mat> truth_split;
	    	Mat magnitude,angle;
            split(ground_truth, truth_split);
            cartToPolar(truth_split[0], truth_split[1], magnitude, angle, true);
	    	mask = (magnitude <= 8);
	    	valid_count=countNonZero(mask);
	    	if (valid_count==0){
                        float R_thresholds[] = { 0.f, 0.5f, 1.f, 2.f, 4.f, 8.f}; 
                        vector<float> R_thresholds_vec = { 0.f, 0.5f, 1.f, 2.f, 4.f, 8.f};
                        float A_thresholds[] = { 0.5f, 0.75f, 0.95f };
                        int R_thresholds_count = sizeof(R_thresholds) / sizeof(float);
                        int A_thresholds_count = sizeof(A_thresholds) / sizeof(float);
                        printf("Using %s error measure\n", error_measure.c_str());
                        printf("Average: %.2f\nStandard deviation: %.2f\n", 0, 0);
                        for ( int i = 0; i < R_thresholds_count; ++i ){
                            printf("R%.1f: %.2f%%\n", R_thresholds[i], 0 * 100);
                            	}
                        for ( int i = 0; i < A_thresholds_count; ++i ){
                            printf("A%.2f: %.2f\n", A_thresholds[i], 0);
                        }
                        printf("Valid region: %d \n", valid_count);
	    	}
	    	//cout<<"GT mag ="<<endl<<endl<<magnitude;
	    }
        else
        {
            printf("Invalid region selected! Available options: all, discontinuities, untextured");
            return -1;
        }
        if(display_images)
        {
            namedWindow( "Mask region", WINDOW_AUTOSIZE );
            imshow( "Mask region", mask );
            Mat flow_image = flowToDisplay(ground_truth);
	    //flowToOut(ground_truth);
            namedWindow( "Ground Truth flow", WINDOW_AUTOSIZE );
            imshow( "Ground Truth flow", flow_image );
        }

        //masking out NaNs and incorrect GT values
        Mat truth_split[2];
        split(ground_truth, truth_split);
        Mat abs_mask = Mat((abs(truth_split[0]) < 1e9) & (abs(truth_split[1]) < 1e9));
        Mat nan_mask = Mat((truth_split[0]==truth_split[0]) & (truth_split[1] == truth_split[1]));
        bitwise_and(abs_mask, nan_mask, nan_mask);

        bitwise_and(nan_mask, mask, mask); //including the selected region

        if(display_images) // display difference between computed and GT flow
        {
            Mat difference = ground_truth - flow;
            Mat masked_difference;
            difference.copyTo(masked_difference, mask);
            Mat flow_image = flowToDisplay(masked_difference);
            namedWindow( "Error map", WINDOW_AUTOSIZE );
            imshow( "Error map", flow_image );
        }

	if(region == "smalld" && valid_count !=0 ||region=="all"){
           printf("Using %s error measure\n", error_measure.c_str());
           calculateStats(computed_errors, mask, display_images);
	       valid_count=countNonZero(mask);
           printf("Valid region: %d \n", valid_count);
	}
        //printf("EOF");

    }
    if(display_images) // wait for the user to see all the images
        waitKey(0);
    return 0;

}
