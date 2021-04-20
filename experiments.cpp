#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;

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
    /// Reads in all hd faces and stores them unflattened in a vector
    for (string name : get_filenames("Faces_FA_FB/fa_H"))
    {
        Mat im = imread(name,IMREAD_GRAYSCALE);
        // normalize im to contain values between 0 - 1
        normalize(im, im, 0.0, 1.0, NORM_MINMAX, CV_64FC1);
        faces.push_back(im);
    }
    cout << "CP1" << endl;
    imshow("face0", faces[0]);
    waitKey(0);

    Mat mean = imread("mean.pgm", IMREAD_GRAYSCALE);
    normalize(mean, mean, 0.0, 1.0, NORM_MINMAX, CV_64FC1);
    imshow("Mean", mean);
    waitKey(0);

    Mat eigenFaces = imread("eigenfaces.pgm", IMREAD_GRAYSCALE);
    normalize(eigenFaces, eigenFaces, 0.0, 1.0, NORM_MINMAX, CV_64FC1);

    //make unit length
    for (int i = 0; i < eigenFaces.cols; i++)
    {
        multiply(eigenFaces.col(i), 1.0/(sum(eigenFaces.col(i))(0)), eigenFaces.col(i));
    }

    Mat testImg = Mat(faces[0].size(), CV_64FC1);
    for (int i = 0; i < faces[0].rows; i++)
    {
        for (int j = 0; j < faces[0].cols; j++)
        {
            testImg.at<double>(i, j) = eigenFaces.at<double>(j + i * faces[0].cols, 1);
        }
    }
//    imshow("Display window", testImg); //visualization of eigenface
//    waitKey(0);

    //flatten the face

    Mat face = faces[0] - mean;
    cout << "NMF" << norm(faces[0], mean) << endl;
//    cout << face << endl;
//    face = face.t();
    Mat flattened_face = Mat::zeros(Size(faces[0].cols * faces[0].rows, 1), CV_64FC1);

    cout << flattened_face.rows << " " << flattened_face.cols << endl;

    cout << "CP2" << endl;

    // for every pixel
    for (int j = 0; j < face.cols; j++)
    {
        for (int k = 0; k < face.rows; k++)
        {
            // Mat.at() takes in (row,col) indexing
            flattened_face.at<double>(0, k + j * face.rows) = (double)face.at<double>(k, j);
        }
    }

    cout << "CP3" << endl;

    Mat weights = flattened_face * eigenFaces;

    cout << "CP4" << endl;

    cout << weights.col(0).rows << " " << weights.col(0).cols << endl;
    cout << eigenFaces.col(0).rows << " " << eigenFaces.col(0).cols << endl;

    Mat reconstruction = Mat::zeros(Size(1, faces[0].cols * faces[0].rows), CV_64FC1);
    for (int i = 0; i < eigenFaces.cols; i++)
    {
        reconstruction += weights.col(i).at<double>(0,0) * eigenFaces.col(i);
    }



//    cout << sum(reconstruction) << endl;

    cout << "CP5" << endl;

    Mat unflattened = Mat(faces[0].size(), CV_64FC1);
    for (int i = 0; i < faces[0].rows; i++)
    {
        for (int j = 0; j < faces[0].cols; j++)
        {
            unflattened.at<double>(i, j) = reconstruction.at<double>(j + i * faces[0].cols, 0);
        }
    }
    Mat output = unflattened + mean;
    cout << "ERROR: " << norm(faces[0],output) << endl;


    imshow("Display window", output);
    waitKey(0);


//    imshow("Display window", testImg); //visualization of eigenface
//    waitKey(0);
    return 0;
}