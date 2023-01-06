#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Helper function to get the pixel value at a given position in the image
uchar getPixelValue(const Mat& image, int row, int col) {
  // Return 0 if the pixel is outside the bounds of the image
  if (row < 0 || row >= image.rows || col < 0 || col >= image.cols) {
    return 0;
  }
  return image.at<uchar>(row, col);
}

// Inpainting function that estimates the value of each missing pixel as the average of the surrounding pixels
void inpaint(Mat& image, const Mat& mask) {
  // Create a copy of the image to store the inpainted result
  Mat inpainted = image.clone();

  // Iterate through each pixel in the image
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      // If the pixel is not missing (mask value is 0), skip it
      if (mask.at<uchar>(row, col) == 0) {
        continue;
      }

      // Calculate the average of the surrounding pixels
      int sum = 0;
      int count = 0;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          sum += getPixelValue(image, row + i, col + j);
          count++;
        }
      }
      int avg = sum / count;

      // Set the missing pixel value to the average of the surrounding pixels
      inpainted.at<uchar>(row, col) = avg;
    }
  }

  // Copy the inpainted result back into the original image
  image = inpainted;
}

int main(int argc, char** argv) {
  // Load the image and the mask
  Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
  Mat mask = imread("mask.jpg", IMREAD_GRAYSCALE);

  // Inpaint the image
  inpaint(image, mask);

  // Save the inpainted image
  imwrite("inpainted.jpg", image);

  return 0;
}







while(b.empty()!=true){
        int j1=a.top();
        int k1=b.top();
        
        float min=INFINITY;
        
        float sum=0;
        float val[3];
        if(flag[j1][k1+1]==KNOWN){
            for(int i=0;i<image.channels();i++){
                val[i]=image.at<cv::Vec3b>(j1,k1+1)[i];
            }
            sum+=1;
        }
        if(flag[j1][k1-1]==KNOWN){
            for(int i=0;i<image.channels();i++){
                val[i]=image.at<cv::Vec3b>(j1,k1-1)[i];
            }
            sum+=1;
        }
        if(flag[j1+1][k1]==KNOWN){
            for(int i=0;i<image.channels();i++){
                val[i]=image.at<cv::Vec3b>(j1+1,k1)[i];
            }
            sum+=1;
        }
        if(flag[j1-1][k1]==KNOWN){
            for(int i=0;i<image.channels();i++){
                val[i]=image.at<cv::Vec3b>(j1-1,k1)[i];
            }
            sum+=1;
        }
        for(int i=0;i<image.channels();i++){
            val[i]=val[i]/sum;
            helpme.at<cv::Vec3b>(j1,k1)[i]=val[i];
        }
        flag[j1][k1]=KNOWN;        
        for(int i=-1;i<=1;i++){
            for(int j=-1;j<=1;j++){
                if(flag[j1+i][k1+j]==INSIDE){
                    flag[j1+i][k1+j]=BAND;
                    a.push(j1+i);
                    b.push(k1+j);
                }
            }
        }
        a.pop();
        b.pop();
    }