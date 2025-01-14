// Computer Vision : Image Stitching

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_world4100d")
#else
#pragma comment(lib, "opencv_world4100")
#endif

using namespace cv;
using namespace std;

bool compareMatches(const DMatch& a, const DMatch& b) {
    return a.distance < b.distance;
}

int main() {

    string imagePath1 = "C:/Users/sonyrainy/Desktop/left_desk.jpg";
    string imagePath2 = "C:/Users/sonyrainy/Desktop/right_desk.jpg";
    Mat src1 = imread(imagePath1, IMREAD_COLOR);
    Mat src2 = imread(imagePath2, IMREAD_COLOR);
    if (src1.empty() || src2.empty()) {
        printf(" Error opening image\n");
        return EXIT_FAILURE;
    }

    // 각각의 feature Point, descriptor를 얻는다.
    Ptr<SIFT> sift = SIFT::create(5000);
    vector<KeyPoint> featurePoints1, featurePoints2;
    Mat descriptors1, descriptors2;
    sift->detectAndCompute(src1, noArray(), featurePoints1, descriptors1);
    sift->detectAndCompute(src2, noArray(), featurePoints2, descriptors2);

    // bruthforce 방식으로 matching
    BFMatcher matcher(NORM_L2, false);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // distance 기준 정렬 후, 20% 정도의 match 부분만 고른다.
    sort(matches.begin(), matches.end(), compareMatches);

    int lowNum_GoodMatches = matches.size() * 0.2;
    vector<DMatch> goodMatch(matches.begin(), matches.begin() + lowNum_GoodMatches);

    // drawMatches
    Mat imgMatches;
    drawMatches(src1, featurePoints1, src2, featurePoints2, goodMatch, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Good Matches", imgMatches);
    waitKey(0);

    // 왼쪽, 오른쪽에서의 good match 부분을 별도 배열에 추가
    vector<Point2f> leftPoints, rightPoints;
    for (size_t i = 0; i < goodMatch.size(); i++) {
        leftPoints.push_back(featurePoints1[goodMatch[i].queryIdx].pt);
        rightPoints.push_back(featurePoints2[goodMatch[i].trainIdx].pt);
    }

    Mat HG_RANSAC, HG_LS;

    // RANSAC - Homography 계산
    HG_RANSAC = findHomography(rightPoints, leftPoints, RANSAC, 3.0);


    // Least Squares - Homography 계산
    HG_LS = findHomography(rightPoints, leftPoints, 0);


    cout << "Homography (RANSAC):" << endl;
    cout << HG_RANSAC << "\n";

    cout << "Homography (Least Squares):" << endl;
    cout << HG_LS << "\n";


    // Warp perspective으로 오른쪽 그림 먼저 붙여 넣기
    Size resultSize(src1.cols * 2, src1.rows);
    Mat resultRansac(resultSize, src1.type(), Scalar::all(0));
    Mat resultLS(resultSize, src1.type(), Scalar::all(0));

    // RANSAC - Homography
    warpPerspective(src2, resultRansac, HG_RANSAC, resultSize, INTER_LINEAR, BORDER_CONSTANT, Scalar::all(0));

    Mat roiRansac(resultRansac, Rect(0, 0, src1.cols, src1.rows));
    src1.copyTo(roiRansac);

    // Least Squares - Homography
    warpPerspective(src2, resultLS, HG_LS, resultSize, INTER_LINEAR, BORDER_CONSTANT, Scalar::all(0));

    Mat roiLS(resultLS, Rect(0, 0, src1.cols, src1.rows));
    src1.copyTo(roiLS);

    imshow("RANSAC - Stitched Image", resultRansac);
    waitKey(0);
    
    imshow("Least Squares - Stitched Image", resultLS);
    waitKey(0);

}
