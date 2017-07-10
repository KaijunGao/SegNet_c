#define CHEPAI    1
#define CHEJIAHAO 0

#define USE_OPENCV 1
#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <vector>
#include <chrono> //Just for time measurement

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

class Classifier {
public:
    Classifier(const string& model_file,
               const string& trained_file);

    float* Predict(const cv::Mat& img);

private:
    void SetMean(const string& mean_file);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

    void Visualization(Blob<float>* output_layer, string LUT_file);

private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file) {

    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}


float* Classifier::Predict(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //Just for time measurement

    //net_->Forward();
    net_->ForwardPrefilled();

    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << " sec" <<std::endl; //Just for time measurement


    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];

    return (float *)output_layer->cpu_data();
}


void Classifier::Visualization(Blob<float>* output_layer, string LUT_file) {

    std::cout << "output_blob(n,c,h,w) = " << output_layer->num() << ", " << output_layer->channels() << ", "
              << output_layer->height() << ", " << output_layer->width() << std::endl;

    cv::Mat merged_output_image = cv::Mat(output_layer->height(), output_layer->width(), CV_32F, const_cast<float *>(output_layer->cpu_data()));
    //merged_output_image = merged_output_image/255.0;

    merged_output_image.convertTo(merged_output_image, CV_8U);
    cv::cvtColor(merged_output_image.clone(), merged_output_image, CV_GRAY2BGR);
    cv::Mat label_colours = cv::imread(LUT_file,1);
    cv::Mat output_image;
    LUT(merged_output_image, label_colours, output_image);

    cv::imshow( "Display window", output_image);
    cv::waitKey(0);
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_float, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

static int colormap_bak[] = {
    //255,255,255,  //nothing
    0,0,0,
    204,0,0,      //road
    0,0,153,      //road sign
    51,255,255,    //traffic sign

    153,255,0,    //road line
    153,51,102    //road line 2
};

#if CHEPAI
static char char_map[] =
{
    '\0',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'J',
    'K',
    'L',
    'M',
    'N',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'U',
    'V',
    'W',
    'X',
    'Y',
    'Z'
};
#endif

#if CHEJIAHAO
static char char_map[] =
{
    '\0',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'J',
    'K',
    'L',
    'M',
    'N',
    'P',
    'R',
    'S',
    'T',
    'U',
    'V',
    'W',
    'X',
    'Y',
    'Z'
};
#endif

static int colormap[] =
{
    0,0,0,
    180,120,120,
    6,230,230,
    80,50,50,
    4,200,3,
    120,120,80,
    140,140,140,
    204,5,255,
    230,230,230,
    4,250,7,
    224,5,255,
    235,255,7,
    150,5,61,
    120,120,70,
    8,255,51,
    255,6,82,
    143,255,140,
    204,255,4,
    255,51,7,
    204,70,3,
    0,102,200,
    61,230,250,
    255,6,51,
    11,102,255,
    255,7,71,
    255,9,224,
    9,7,230,
    220,220,220,
    255,9,92,
    112,9,255,
    8,255,214,
    7,255,224,
    255,184,6,
    10,255,71,
    255,41,10,
    7,255,255,
    224,255,8,
};

struct lpr_msg
{
    cv::Rect r;
    int  cnt;
    char c;
};


void get_single_char(cv::Mat &colorMat, int b, int g, int r, std::vector<lpr_msg> &mLpr, char ch)
{
    cv::Mat binImg;

    cv::cvtColor(colorMat, binImg, CV_BGR2GRAY);
    for (int h=0; h<colorMat.rows; h++)
    {
        for (int w=0; w<colorMat.cols; w++)
        {
            int b0 = colorMat.at<cv::Vec3b>(h,w)[0];
            int g0 = colorMat.at<cv::Vec3b>(h,w)[1];
            int r0 = colorMat.at<cv::Vec3b>(h,w)[2];
            if (b0==b && g0==g && r0==r)
            {
                binImg.at<uchar>(h,w) = 255;
            }
            else
            {
                binImg.at<uchar>(h,w) = 0;
            }
        }
    }

    cv::Mat dst;
    cv::dilate(binImg,dst,cv::Mat(11,11,CV_8U),cv::Point(-1,-1),1);
    //cv::dilate(binImg,dst,cv::Mat(3,3,CV_8U),cv::Point(-1,-1),1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dst, contours, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    for (int i=0; i<contours.size(); i++)
    {
        cv::Rect re = cv::boundingRect(contours[i]);
        cv::Mat mRoi = colorMat(re).clone();

        lpr_msg char_lpr;
        char_lpr.c = ch;
        char_lpr.r = re;

        int cnt = 0;
        for (int h=0; h<mRoi.rows; h++)
        {
            for (int w=0; w<mRoi.cols; w++)
            {
                int b0 = mRoi.at<cv::Vec3b>(h,w)[0];
                int g0 = mRoi.at<cv::Vec3b>(h,w)[1];
                int r0 = mRoi.at<cv::Vec3b>(h,w)[2];
                if (b0==b && g0==g && r0==r)
                {
                   cnt++;
                }
            }
        }

        char_lpr.cnt = cnt;
        mLpr.push_back(char_lpr);

    }

}

bool location_x_compare(lpr_msg &a, lpr_msg &b)
{
    return a.r.x < b.r.x;
}

void get_lpr(cv::Mat &colorMat)
{
    std::vector<lpr_msg> mLpr;

    for (int id=1; id<35; id++)
    {
        get_single_char(colorMat, colormap[id*3], colormap[id*3+1], colormap[id*3+2], mLpr, char_map[id]);
    }


    std::sort(mLpr.begin(), mLpr.end(), location_x_compare);

    int cnt = 0;
    for (int i=0; i<mLpr.size(); i++)
    {
        //if (mLpr[i].cnt > 180)
        if (mLpr[i].cnt > 110)
        {
            std::cout << mLpr[i].c << " cnt " << mLpr[i].cnt << std::endl;
            cnt++;
        }
    }
    std::cout << "total cnt:" << cnt << std::endl;
}

int main()
{
    string model_file   = "/home/em-gkj/devdata/chejian_v2/chejian_v2.1/model/baodan_chepai.prototxt";
    string trained_file = "/home/em-gkj/devdata/chejian_v2/chejian_v2.1/model/baodan_chepai.caffemodel";
    std::ifstream file_list("/home/em-gkj/devdata/chejian_v2_train-data/character_det/SegNet-text-chepaichar/CamVid/img/list.txt");

    Classifier classifier(model_file, trained_file);

    std::string file;
    while(file_list >> file)
    {
        std::cout << file << std::endl;

        cv::Mat img = cv::imread(file, 1);
        float *data_ptr = classifier.Predict(img);

        cv::Mat colorMat = img.clone();
        for (int i=0; i<colorMat.rows; i++)
        {
            for (int j=0; j<colorMat.cols; j++)
            {
                int id = -1;
                float max = -1.0;

                for (int n=0;n<34;n++)
                {
                    if (data_ptr[n*colorMat.cols*colorMat.rows+i*colorMat.cols+j] > max)
                    {
                        max = data_ptr[n*colorMat.cols*colorMat.rows+i*colorMat.cols+j];
                        id = n;
                    }
                }

                colorMat.at<cv::Vec3b>(i,j)[0] = colormap[id*3];
                colorMat.at<cv::Vec3b>(i,j)[1] = colormap[id*3+1];
                colorMat.at<cv::Vec3b>(i,j)[2] = colormap[id*3+2];
            }
        }

        get_lpr(colorMat);

        cv::imshow("src", img);
        cv::imshow("color", colorMat);
        cv::waitKey(0);
    }
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV


