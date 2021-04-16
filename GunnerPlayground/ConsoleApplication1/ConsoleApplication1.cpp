// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;

// http://www.cplusplus.com/forum/windows/189681/
//
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

    /// Calculate the mean face of the dataset
    Mat mean = Mat::zeros(Size(faces[0].cols, faces[0].rows), CV_64FC1);
    for (Mat face : faces)
    {
        //store running total of face values
        mean += face;
    }
    // divide by total number of faces to get mean
    mean /= double(faces.size());

    
    //imshow("windowname", mean);
    //waitKey(0);

    cout << "CP2" << endl;

    /// Subtract mean from every face to normalize them
    for (int i = 0; i < faces.size(); i++)
    {
        faces.at(i) -= mean;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Create a matrix of flattened faces; call this matrix A
    // faces.size() is the number of samples
    // faces[0].cols * facces[0].rows gets the size of the flattened image vector
    // Size() opencv datatype takes in (width,height) of matrix
    ////////////////////////////////////////////////////////////////////////////////

    Mat A = Mat::zeros(Size(faces.size(), faces[0].cols * faces[0].rows), CV_64FC1);
    //cout << "A rows: " << A.rows << " A cols:" << A.cols << endl;

    // for every face
    for (int i = 0; i < faces.size(); i++)
    {
        Mat face = faces[i];
        face = face.t();

        // for every pixel
        for (int j = 0; j < face.cols; j++)
        {
            for (int k = 0; k < face.rows; k++)
            {
                // Mat.at() takes in (row,col) indexing
                A.at<double>(k + j * face.rows, i) = (double)face.at<double>(k, j);
            }
        }
    }

    // Sanity check to make sure faces weres flattened/stored in A correctly
    // Unflatten and display the first face at column 0
    /*
    Mat testImg = Mat(faces[0].size(), CV_64FC1);
    for (int i = 0; i < faces[0].rows; i++)
    {
        for (int j = 0; j < faces[0].cols; j++)
        {
            testImg.at<double>(i, j) = A.at<double>(j + i * faces[0].cols, 0);
        }
    }
    imshow("Display window", testImg+mean);
    waitKey(0);
    */

    // .t() doesnt transpose the object that calls it, returns a transposed matrix
    // https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html#aaa428c60ccb6d8ea5de18f63dfac8e11
    Mat t = A.t();


    Mat cov = t*A;


    // Size of covarianace matrix should be MxM where M is number of training samples
    cout << "Cov rows:" <<cov.rows << " Cov Cols:" << cov.cols << endl;

    // Compute eigen values & eigen vectors of covariance matrix and stores in descending order
    // https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=eigen#bool%20eigen%28InputArray%20src,%20OutputArray%20eigenvalues,%20OutputArray%20eigenvectors,%20int%20lowindex,int%20highindex%29
    Mat eigenValues, eigenVectors;
    eigen(cov, eigenValues, eigenVectors);
    cout << "Reconstructing Eigen Faces.." << endl;
    Mat eigenFaces = A * eigenVectors;


    // Sanity check to make sure eigenFaces arent garbage
    // Unflatten and display the first eigenFace at column 0
    /*
    Mat testImg = Mat(faces[0].size(), CV_64FC1);
    for (int i = 0; i < faces[0].rows; i++)
    {
        for (int j = 0; j < faces[0].cols; j++)
        {
            testImg.at<double>(i, j) = eigenFaces.at<double>(j + i * faces[0].cols, 0);
        }
    }
    imshow("Display window", testImg + mean);
    waitKey(0);
    */

    // Convert data back to scale of 0 - 255 so it can be read by an image editor like GIMP
    cout << "Normalizing mean back to 0 - 255" << endl;
    normalize(mean, mean, 0, 255, NORM_MINMAX, CV_8UC1);
    cout << "Normalizing eigenfaces 0-255" << endl;
    normalize(eigenFaces, eigenFaces, 0, 255, NORM_MINMAX, CV_8UC1);
    // store as PGMs
    imwrite("mean.pgm", mean);
    imwrite("eigenfaces.pgm", eigenFaces);

    return 0;
}