#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <map>
#include <string>
#include <queue>
#include <vector>
#include <cmath>
using namespace std;
#define KNOWN  1  //known outside narrow band
#define BAND   2  //narrow band 
#define INSIDE 3  //unknown

//image to be inpainted is read 
cv::Mat img = cv::imread("C:/Users/ghosh/OneDrive/Desktop/CG_Project/images/image.png");
cv::Mat image=img.clone();
cv::Mat helpme=img.clone();


std::vector<std::vector<int>> flag(img.rows,std::vector<int>(img.cols,0));

//function to initialize distance matrix as infinity
void makeinf(std::vector<std::vector<int>> dists){
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            dists[i][j]=1000000000000;
        }
    }
}


cv::Point cor;
bool draw=false;
//function to draw over image
void mouseCall(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        //if left mouse button is pressed then initial point for line is stored
        draw = true;
        cor.x=x;
        cor.y=y;
        double y1=y;
    }
    else if (event == cv::EVENT_MOUSEMOVE){
        if (draw){
            //if mouse is moved then line is drawn
            cv::line(img, cor, cv::Point(x, y), cv::Scalar(0, 255, 0), 10);
            //final point is stored as initial point for next line
            cor.x=x;
            cor.y=y;
        }
    }
    else if (event == cv::EVENT_LBUTTONUP){
        draw = false;
        //if left mouse button is released then final point for line is stored
        cv::line(img, cor, cv::Point(x, y), cv::Scalar(0, 255, 0), 10);
    }
}
uchar getPixelValue(const cv::Mat& image, int row, int col) {
  // Return 0 if the pixel is outside the bounds of the image
  if (row < 0 || row >= image.rows || col < 0 || col >= image.cols) {
    return 0;
  }
  return image.at<uchar>(row, col);
}
//eikonal equation solver
float solveEikonal(int row1, int row2, int col1, int col2, std::vector<std::vector<int>> flag,std::vector<std::vector<int>> dist){
    if(col1<0 || col1>=image.cols || row1<0 || row1>=image.rows){
        return 1000000000000;
    }
    if(col2<0 || col2>=image.cols || row2<0 || row2>=image.rows){
        return 1000000000000;
    }

    int flag1, flag2;
    flag1 = flag[row1][col1];
    flag2 = flag[row2][col2];
    if(flag1==KNOWN && flag2==KNOWN){
        int dist1=dist[row1][col1];
        int dist2=dist[row2][col2];
        int d=2-(dist1-dist2)*(dist1-dist2);
        if(d<0){
            return 1000000000000;
        }
        else{
            float s=(dist1+dist2+sqrt(d))/2;
            if(s>=dist1 && s>=dist2){
                return s;
            }
            s+=sqrt(d);
            if(s>=dist1 && s<dist2){
                return s;
            }
            return 1000000000000;
        }
    }
    else if(flag1==KNOWN){
        return dist[row1][col1]+1;
    }
    else if(flag2==KNOWN){
        return dist[row2][col2]+1;
    }
    else{
        return  1000000000000;
    }
}
//calculationg gradient due to rows and columns
std::vector<float> gradient(int row, int col, std::vector<std::vector<int>> flag,std::vector<std::vector<int>> dist){
    std::vector<float> grad;
    float gradx, grady;
    int prevrow=row-1;
    int nextrow=row+1;
    float val=dist[row][col];
    if(prevrow<0 && nextrow>image.rows){
        grady= 1000000000000;
    }    
    else{
        int flag1=flag[prevrow][col];
        int flag2=flag[nextrow][col];
        if(flag1!=INSIDE && flag2!=INSIDE){
            grady=(dist[prevrow][col]-dist[nextrow][col])/2;
        }
        else if(flag2!=INSIDE){
            grady=val-dist[nextrow][col];
        }
        else if(flag1==INSIDE){
            grady=val-dist[prevrow][col];
        }
        else{
            grady=0;
        }
    }
    int nextcol=col+1;
    int prevcol=col-1;
    if(prevcol<0 && nextcol>image.cols){
        gradx= 1000000000000;
    }
    else{
        int flag1=flag[row][col-1];
        int flag2=flag[row][col+1];
        if(flag1!=INSIDE && flag2!=INSIDE){
            gradx=(dist[row][col-1]-dist[row][col+1])/2;
        }
        else if(flag2!=INSIDE){
            gradx=val-dist[row][col+1];
        }
        else if(flag1==INSIDE){
            gradx=val-dist[row][col-1];
        }
        else{
            gradx=0;
        }
    }
    grad.push_back(gradx);
    grad.push_back(grady);
    return grad;

}
//compute distances from narrow band point to outside mask using fast marching method
void outside(std::vector<std::vector<int>> flag,std::priority_queue<int> a,std::priority_queue<int> b,std::vector<std::vector<int>> dists){
    std::vector<std::vector<int>>  newflag=flag;
    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            if(flag[i][j]==INSIDE){
                newflag[i][j]==KNOWN;
            }
            else if(flag[i][j]==KNOWN){
                newflag[i][j]==INSIDE;
            }
        }
    }
    float lastdist=0;
    int rad=2;
    float twicerad=2*rad;
    while(a.empty()!=true){
        int row=a.top();
        int col=b.top();
        if(lastdist>=twicerad){
            break;
        }
        a.pop();
        b.pop();
        newflag[row][col]=KNOWN;
        int prevr=row-1;
        int nextr=col+1;
        int prevc=col-1;
        int nextc=col+1;
        int neighbors[4][2];
        neighbors[0][0]=prevr;
        neighbors[0][1]=col;
        neighbors[1][0]=prevr;
        neighbors[1][1]=col;
        neighbors[2][0]=row;
        neighbors[2][1]=prevc;
        neighbors[3][0]=row;
        neighbors[3][1]=nextc;
        for(int i=0;i<4;i++){
            if(neighbors[i][0]<0 || neighbors[i][0]>image.rows && neighbors[i][1]<0 && neighbors[i][1]>image.cols){
                continue;
            }
            if(newflag[neighbors[i][0]][neighbors[i][1]]!=INSIDE){
                continue;
            }
            float a1=solveEikonal(neighbors[i][0]-1,neighbors[i][1],neighbors[i][0],neighbors[i][1]-1,newflag,dists);
            float a2=solveEikonal(neighbors[i][0]-1,neighbors[i][1],neighbors[i][0],neighbors[i][1]+1,newflag,dists);
            float a3=solveEikonal(neighbors[i][0]+1,neighbors[i][1],neighbors[i][0],neighbors[i][1]-1,newflag,dists);
            float a4=solveEikonal(neighbors[i][0]+1,neighbors[i][1],neighbors[i][0],neighbors[i][1]+1,newflag,dists);
            float a5=std::min(a1,a2);
            float a6=std::min(a3,a4);
            lastdist=std::min(a5,a6);
            dists[neighbors[i][0]][neighbors[i][1]]=lastdist;
            a.push(neighbors[i][0]);
            b.push(neighbors[i][1]);
        }

    }
    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            dists[i][j]=-dists[i][j];
        }
    }
}
//compute colour value for a given pixel
std::vector<float> pixelpaint(int row,int col,std::vector<std::vector<int>> flag,std::vector<std::vector<int>> dist){
    float distance=dist[row][col];
    std::vector<float> grad=gradient(row,col,flag,dist);
    float gradx=grad[0];
    float grady=grad[1];
    std::vector<float> pixval;
    float weightsum;
    for(int i=row-1;i<=row+1;i++){
        float weight;
        if(i<0 || i>image.rows){
            continue;
        }
        for(int j=col-1;j<=col+1;j++){
            if(j<0 || j>image.cols){
                continue;
            }
            if(flag[i][j]==INSIDE){
                continue;
            }
            float dirx=i-row;
            float diry=j-col;
            float dirsqr=dirx*dirx+diry*diry;
            float dirnorm=sqrt(dirx*dirx+diry*diry);
            if(dirnorm>2){
                continue;
            }
            //direction factor
            float dirfactor=abs(diry*grady+dirx*gradx);
            if(dirfactor==0){
                dirfactor=1e-6;
            }
            float neighdist=dist[i][j];
            //level factor
            float level_factor=1/(1+abs(neighdist-distance));
            //distance factor
            float dist_factor=1/(dirnorm*dirsqr);
            weight=dirfactor*level_factor*dist_factor;
            pixval.push_back(weight*image.at<cv::Vec3b>(i,j)[0]);
            pixval.push_back(weight*image.at<cv::Vec3b>(i,j)[1]);
            pixval.push_back(weight*image.at<cv::Vec3b>(i,j)[2]);
            weightsum+=weight;
        }
    }
    for(int i=0;i<pixval.size();i++){
        pixval[i]=pixval[i]/weightsum;
    }
    return pixval;

}

void Inpainttelea(){
    //stores band rows and columns
    std::priority_queue<int> a;
    std::priority_queue<int> b;
    std::vector<std::vector<int>> dist(img.rows,std::vector<int>(img.cols,0));
    makeinf(dist);
    cv::Mat mask = cv::imread("C:/Users/ghosh/OneDrive/Desktop/CG_Project/images/mask.jpg");
    
    int channels = image.channels();
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            if (mask.at<cv::Vec3b>(row, col)[0] != 0) {              
                if(mask.at<cv::Vec3b>(row, col+1)[0] == 0 || mask.at<cv::Vec3b>(row, col-1)[0] == 0){
                    // narrow band is stored in a(row) and b(col)
                    flag[row][col]=BAND;
                    a.push(row);
                    b.push(col);    
                }
                else if(mask.at<cv::Vec3b>(row+1, col)[0] == 0|| mask.at<cv::Vec3b>(row-1, col)[0] == 0){
                    flag[row][col]=BAND;
                    a.push(row);
                    b.push(col);    
                }
                else{
                    flag[row][col]=INSIDE;
                }
            }      
            else{
                flag[row][col]=KNOWN;                
            }
        }
    }
    outside(flag,a,b,dist);
    while(b.empty()!=true){

        int row=a.top();
        int col=b.top();
        a.pop();
        b.pop();
        flag[row][col]=KNOWN;
        int prevr=row-1;
        int nextr=col+1;
        int prevc=col-1;
        int nextc=col+1;
        int neighbors[4][2];
        neighbors[0][0]=prevr;
        neighbors[0][1]=col;
        neighbors[1][0]=prevr;
        neighbors[1][1]=col;
        neighbors[2][0]=row;
        neighbors[2][1]=prevc;
        neighbors[3][0]=row;
        neighbors[3][1]=nextc;
        for(int i=0;i<4;i++){
            if(neighbors[i][0]<0 || neighbors[i][0]>image.rows && neighbors[i][1]<0 && neighbors[i][1]>image.cols){
                continue;
            }
            if(flag[neighbors[i][0]][neighbors[i][1]]!=INSIDE){
                continue;
            }
            float a1=solveEikonal(neighbors[i][0]-1,neighbors[i][1],neighbors[i][0],neighbors[i][1]-1,flag,dist);
            float a2=solveEikonal(neighbors[i][0]-1,neighbors[i][1],neighbors[i][0],neighbors[i][1]+1,flag,dist);
            float a3=solveEikonal(neighbors[i][0]+1,neighbors[i][1],neighbors[i][0],neighbors[i][1]-1,flag,dist);
            float a4=solveEikonal(neighbors[i][0]+1,neighbors[i][1],neighbors[i][0],neighbors[i][1]+1,flag,dist);
            float a5=std::min(a1,a2);
            float a6=std::min(a3,a4);
            float lastdist=std::min(a5,a6);
            dist[neighbors[i][0]][neighbors[i][1]]=lastdist;
            std::vector<float> temp;
            temp=pixelpaint(neighbors[i][0],neighbors[i][1],dist,flag);
            helpme.at<cv::Vec3b>(neighbors[i][0],neighbors[i][1])[0]=temp[0];
            helpme.at<cv::Vec3b>(neighbors[i][0],neighbors[i][1])[1]=temp[1];
            helpme.at<cv::Vec3b>(neighbors[i][0],neighbors[i][1])[2]=temp[2];
            flag[neighbors[i][0]][neighbors[i][1]]=BAND;
            a.push(neighbors[i][0]);
            b.push(neighbors[i][1]);    
        }
    }
    cv::namedWindow("inpaint");
    while (char(cv::waitKey(1) != 'q')){
        cv::imshow("inpaint", helpme);
        if (cv::waitKey(1) == 27) break;
    }
    
    
}
//fast digital inpainting
void FDI(){
    cv::Mat img = cv::imread("C:/Users/ghosh/OneDrive/Desktop/CG_Project/images/image.png");
    cv::Mat image=img.clone();
    cv::Mat mask = cv::imread("C:/Users/ghosh/OneDrive/Desktop/CG_Project/images/mask.jpg");
    cv::Mat imask,kernel,Border,inpaint;
    //creating inverse mask
    cv::bitwise_not(mask, imask);
    auto avg = cv::sum(image);

    avg=avg/(image.rows*image.cols);
    //create image with average color of image
    cv::Mat avgMat(1,1,CV_8UC3,avg);
    cv::resize(avgMat, avgMat, img.size());
    //fill blank areas with average color
    inpaint=(imask / 255).mul(img) + (mask/255).mul(avgMat);
    //gaussian kernel is created
    cv::Mat gausk=(cv::Mat_<float>(3, 3) << 0.073235f, 0.176765f, 0.073235f, 0.176765f, 0.0f, 0.176765f, 0.073235f, 0.176765f, 0.073235f);
	cv::cvtColor(gausk, gausk, cv::COLOR_GRAY2BGR);
    inpaint.convertTo(inpaint, CV_32FC3);
    cv::copyMakeBorder(inpaint, Border, 15, 16, 17, 18, cv::BORDER_REPLICATE);

    int chanels=inpaint.channels();
    //repeated convolution is done
    for(int i=0;i<4500;i++){
        cv::copyMakeBorder(inpaint, Border, 15, 16, 17, 18, cv::BORDER_REPLICATE);
        for (int j = 0; j < inpaint.rows; j++){
			const uchar *pMask = mask.ptr(j);
			float *point = inpaint.ptr<float>(j);
			for (int k = 0; k < inpaint.cols; k++){
				if (pMask[chanels * k] != 0){
					cv::Rect rect(k, j, 3, 3);//rectangle with size of size of gaussian kernel
					cv::Mat roi(Border, rect);
                    //gaussian kernel is applied on every coordinate
					cv::Mat result = roi.mul(gausk);
                    int sum[3];
					for(int l=0;l<3;l++){
                        sum[l]=0;
                        for (int m = 0; m < 3; m++){
                            for (int n = 0; n < 3; n++){
                                sum[l] += result.at<cv::Vec3f>(m, n)[l];                                
                            }
                        }
                        point[chanels * k + l] = sum[l];
                    }
					
				}
			}
		}
    }
    inpaint=inpaint/255;
    cv::namedWindow("inpaint");
    while (char(cv::waitKey(1) != 'q')){
        cv::imshow("inpaint", inpaint);
        if (cv::waitKey(1) == 27) break;
    }
}
//main function
int main(int argc, char** argv) {
    if(img.empty()){
        //if image is not found error message is displayed
        std::cout << "Could not read the image: " << "images/image1.png" << std::endl;
        return 1;
    }  
    cv::Mat cast;
    cv::Mat inpaint;
    //new window is created where we will mark areas to be inpainted
    cv::namedWindow("img");
    cv::setMouseCallback("img", mouseCall);
    while (char(cv::waitKey(1) != 'q')){
        cv::imshow("img", img);
        if (cv::waitKey(1) == 27) break;
        //image with red mark is stored as mask.jpg
        cv::imwrite("C:/Users/ghosh/OneDrive/Desktop/CG_Project/images/mask.jpg", img);
    }
    cv::destroyWindow("img");
    //mask image is read
    cv::Mat mask = cv::imread("C:/Users/ghosh/OneDrive/Desktop/CG_Project/images/mask.jpg");
    //mask image is converted to grayscale where green areas become white and rest black
    cv::cvtColor(mask, cast, cv::COLOR_BGR2HSV);
    cv::inRange(mask, cv::Scalar(0, 155, 0), cv::Scalar(40, 255, 40),cast);
    //the bitwise_and function removes some leftover green pixels
    cv::bitwise_and(image, img, img, cast);    
    //mask is created and overwrites mask.jpg
    cv::imwrite("C:/Users/ghosh/OneDrive/Desktop/CG_Project/images/mask.jpg", cast);
    FDI();
    cv::inpaint(image, cast, inpaint, 3, cv::INPAINT_TELEA);
    cv::namedWindow("inpaint");
    while (char(cv::waitKey(1) != 'q')){
        cv::imshow("inpaint", inpaint);
        if (cv::waitKey(1) == 27) break;
    }
    // Inpainttelea();
    return 0;
}
