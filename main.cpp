// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;

// http://www.cplusplus.com/forum/windows/189681/
std::vector<std::string> get_filenames(filesystem::path path)
{
    std::vector<std::string> filenames;

    // http://en.cppreference.com/w/cpp/experimental/fs/directory_iterator
    const filesystem::directory_iterator end{};

    for (filesystem::directory_iterator iter{ path }; iter != end; ++iter)
    {
        // http://en.cppreference.com/w/cpp/experimental/fs/is_regular_file
        if (filesystem::is_regular_file(*iter)) // comment out if all names (names of directories tc.) are required
            filenames.push_back(iter->path().string());
    }

    return filenames;
}

int main()
{
    vector<Mat> faces;

    namedWindow("Display Image", WINDOW_AUTOSIZE);
    for (auto name : get_filenames("Faces_FA_FB/fa_H"))
    {
        auto im = imread(name, IMREAD_GRAYSCALE);
        Mat divided;
//        subtract(im, 255, divided, Mat(),CV_64FC1);
//        im /= 255;
        faces.push_back(im);
    }

    //mean
//    imshow("Display Image", faces[0]);
    Mat mean = Mat::zeros(Size(faces[0].cols, faces[0].rows),CV_64FC1);
//    cout << mean.type() << endl;
    for (auto face : faces)
    {
        mean += face;
    }
    mean /= faces.size();
    mean /= 255; //https://stackoverflow.com/questions/9588719/opencv-double-mat-shows-up-as-all-white
//    cout << mean << endl;
//    namedWindow("Display Image", WINDOW_FULLSCREEN);
//    imshow("Display Image", mean);
//    waitKey(0);

    //centered faces
    vector<Mat> centered_faces;
    for (auto face : faces)
    {
//        face /= 255;
        Mat centered;
        subtract(face, mean, centered, Mat(),CV_64FC1);
        centered /= 255;
        centered_faces.push_back(centered);
    }

    cout << faces[0].rows << " " << faces[0].cols << endl;

    //flattened matrix
    Mat flattened = Mat::zeros(Size(faces.size(), faces[0].cols * faces[0].rows), CV_64FC1);
    cout << flattened.rows << " " << flattened.cols << endl;
    cout << faces[0].type() << endl;
//    cout << (double) faces[0].at<uchar>(0,0) << endl;
//    flattened.at<double>(0, 0) = (double) faces[0].at<uchar>(0,0);
    for (int i = 0; i < faces.size(); i++)
    {
        auto face = centered_faces[i];
//        auto face = faces[i]; //faces is uchar, centered_faces is double
        face = face.t();
        for (int j = 0; j < face.cols; j++)
        {
            for (int k = 0; k < face.rows; k++)
            {
//                cout << j << ", " << k << endl;
                flattened.at<double>(k + j * face.rows, i) = (double) face.at<double>(k, j);
            }
        }
    }
//    cout << faces[0] << endl;
//    cout << "_________________" << endl;

    auto t = flattened;
    t = t.t();

    cout << t.at<double>(0,0) << " AAAAAAAAAAA " << flattened.at<double>(0,0) << endl;

    Mat cov = flattened * t;
    cov /= cov.rows;

    cout << cov.rows << " " << cov.cols << endl;
    cout << cov.at<double>(0,2) << " " << cov.at<double>(2,0) << endl;

//    for (int i = 0; i < flattened.rows; i++)
//    {
//        cout << cov.at<double>(i, 0) << " ";
//    }

    //compute eigenvals and eigenvecs
    //todo: dont cheat this
    PCA pca(flattened, Mat(), PCA::DATA_AS_ROW, 10);
    cout << pca.eigenvalues;

//    cv::Mat E, V;
//    cv::eigen(cov,E,V);
//    cout << E.rows << " " << E.cols << endl;
    return 0;
}