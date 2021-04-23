#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;

//todo: write: save_vector_as_image
//write: load_images_from_path and load_identities

double mahalanobis(Eigen::VectorXd y1, Eigen::VectorXd y2, Eigen::VectorXd lambda)
{
    double sum = 0;
    for (int i = 0; i < y1.size(); i++)
    {
        sum += (1/lambda(i)) * pow(y1(i) -y2(i),2);
    }
    return sum;
}

void view_vector_as_image(Eigen::VectorXd image_vec, int rows, int cols)
{
    Eigen::MatrixXd mat2 = image_vec.reshaped(rows, cols);
    cv::Mat image2;
    cv::eigen2cv(mat2, image2);
//    image2.convertTo(image2, CV_32FC1, 1.f/255);
    cv::imshow("pls", image2);
    cv::waitKey(0);
}

void write_eigen_matrix_to_file(Eigen::MatrixXd mat, string out_path, string name)
{
    cv::Mat cvMat;
    cv::eigen2cv(mat, cvMat);
    cv::FileStorage fs(out_path, cv::FileStorage::WRITE);
    fs << name << cvMat;
    fs.release();
}

Eigen::MatrixXd read_matrix_into_eigen(string in_path, string name)
{
    cv::FileStorage fs(in_path, cv::FileStorage::READ);
    Eigen::MatrixXd eigenMat;
    cv::Mat cvMat;
    fs[name] >> cvMat;
    cv::cv2eigen(cvMat, eigenMat);
    return eigenMat;
}

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
    bool A = 0, B = 0, C = 0, D = 1, E = 0;
    //Debug/Test Block
    if (A)
    {
        cv::Mat image = cv::imread( "Faces_FA_FB/fa_H/00001_930831_fa_a.pgm", cv::IMREAD_GRAYSCALE);
        Eigen::MatrixXd mat;
        cv::cv2eigen(image, mat);
        std::cout << "Original Image Dim" << mat.rows() << " " << mat.cols() << std::endl;
        Eigen::VectorXd vec = mat.reshaped();
        std::cout << "Flattened vector size " << vec.size() << std::endl;
        view_vector_as_image(vec, mat.rows(), mat.cols());
    }
    //compute eigens
    if (B)
    {
        auto file_name_list = get_filenames("Faces_FA_FB/fa_H");
        cv::Mat first_sample = cv::imread("Faces_FA_FB/fa_H/00001_930831_fa_a.pgm", cv::IMREAD_GRAYSCALE);
        int num_rows = first_sample.rows;
        int num_cols = first_sample.cols;
        int num_features = num_rows * num_cols;
        int num_samples = file_name_list.size();
        // Reads in all hd faces and stores them unflattened in a vector
        //num_features x num_samples (NxM) matrix
        Eigen::MatrixXd original_data = Eigen::MatrixXd(num_features, num_samples);
        for (int i = 0; i < num_samples; i++)
        {
            string name = file_name_list[i];
            cv::Mat im = imread(name,cv::IMREAD_GRAYSCALE);
            Eigen::MatrixXd mat;
            cv::cv2eigen(im, mat);
            mat /= 255.;
            original_data.col(i) = mat.reshaped();
        }
        cout << "Original Data matrix: " << original_data.rows() << " " << original_data.cols() << endl;
        view_vector_as_image(original_data.col(0), num_rows, num_cols);

        //Sample Mean
        //step 1
        Eigen::VectorXd x_bar = Eigen::VectorXd::Zero(num_features);
        for (int i = 0; i < num_samples; i++)
        {
            x_bar += original_data.col(i);
        }
        x_bar /= num_samples;
        view_vector_as_image(x_bar, num_rows, num_cols);

        //step 2
        Eigen::MatrixXd centered_data = Eigen::MatrixXd(num_features, num_samples);
        for (int i = 0; i < num_samples; i++)
        {
            centered_data.col(i) = original_data.col(i) - x_bar;
        }
//        view_vector_as_image(centered_data.col(0), num_rows, num_cols);

        //step 3 (MxM)
        //cov matrix: check symmetric
        auto A_transpose_A = centered_data.transpose() * centered_data;
        cout << "A Transpose A matrix: " << A_transpose_A.rows() << " " << A_transpose_A.cols() << endl;

        //step 4 (takes like 5 min or so)
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s(A_transpose_A);
        //they are sorted from SMALLEST TO LARGEST.ffs.
        cout << "Eigenvals " << s.eigenvalues().reverse() << endl;
        //4b
        Eigen::MatrixXd reversed_eigenvectors = s.eigenvectors().rowwise().reverse();

        write_eigen_matrix_to_file(s.eigenvalues().reverse(), "eigenvalues.mat", "eigenvalues");

        cout << "Checkpoint!" << endl;

        //num_features x num_samples
        //this multiplication also takes a hot minute
        Eigen::MatrixXd transformed_eigens = centered_data * reversed_eigenvectors;
        cout << "transformed eigenfaces dims" << transformed_eigens.rows() << " " << transformed_eigens.cols() << endl;
        //normalize to unit length

        //View top 10 eigenfaces
        //try outputting them BEFORE normalization
        //slide 30
        for (int i = 0; i < 10; i++)
        {
            double f_min = transformed_eigens.col(i).minCoeff();
            double f_max = transformed_eigens.col(i).maxCoeff();
            Eigen::VectorXd col = transformed_eigens.col(i);
            Eigen::VectorXd y = (col - f_min *Eigen::VectorXd::Ones(col.size())) / (f_max - f_min);
            view_vector_as_image(y, num_rows, num_cols);
        }


        for (int i = 0; i < transformed_eigens.cols(); i++)
        {
            transformed_eigens.col(i) = transformed_eigens.col(i) / transformed_eigens.col(i).norm();
        }

        write_eigen_matrix_to_file(transformed_eigens, "transformed_eigens.mat", "transformed_eigens");

        //trying it on image1
        //1 x num_samples
        Eigen::MatrixXd y = centered_data.col(0).transpose() * transformed_eigens;
        cout << "y dims" << y.rows() << " " << y.cols() << endl;

        cout << y << endl;

        Eigen::VectorXd x_hat = Eigen::VectorXd::Zero(num_features);
        for (int k = 0; k < 1000; k++)
        {
            x_hat += y(0, k) * transformed_eigens.col(k);
        }

        view_vector_as_image(x_hat + x_bar, num_rows, num_cols);
    }
    //testing on eigens already computed
    if (C)
    {
        Eigen::MatrixXd transformed_eigens = read_matrix_into_eigen("transformed_eigens.mat", "transformed_eigens");

        auto file_name_list = get_filenames("Faces_FA_FB/fa_H");
        cv::Mat first_sample = cv::imread("Faces_FA_FB/fa_H/00001_930831_fa_a.pgm", cv::IMREAD_GRAYSCALE);
        int num_rows = first_sample.rows;
        int num_cols = first_sample.cols;
        int num_features = num_rows * num_cols;
        int num_samples = file_name_list.size();
        // Reads in all hd faces and stores them unflattened in a vector
        //num_features x num_samples (NxM) matrix
        Eigen::MatrixXd original_data = Eigen::MatrixXd(num_features, num_samples);
        for (int i = 0; i < num_samples; i++)
        {
            string name = file_name_list[i];
            cv::Mat im = imread(name,cv::IMREAD_GRAYSCALE);
            Eigen::MatrixXd mat;
            cv::cv2eigen(im, mat);
            original_data.col(i) = mat.reshaped();
        }
        cout << "Original Data matrix: " << original_data.rows() << " " << original_data.cols() << endl;
//        view_vector_as_image(original_data.col(0), num_rows, num_cols);

        //Sample Mean
        //step 1
        Eigen::VectorXd x_bar = Eigen::VectorXd::Zero(num_features);
        for (int i = 0; i < num_samples; i++)
        {
            x_bar += original_data.col(i);
        }
        x_bar /= num_samples;
//        view_vector_as_image(x_bar, num_rows, num_cols);

        //step 2
        Eigen::MatrixXd centered_data = Eigen::MatrixXd(num_features, num_samples);
        for (int i = 0; i < num_samples; i++)
        {
            centered_data.col(i) = original_data.col(i) - x_bar;
        }

        int sample_to_reconstruct = 1;
        int num_eigenvectors_to_use = transformed_eigens.cols();

        Eigen::MatrixXd y = centered_data.col(sample_to_reconstruct).transpose() * transformed_eigens;
        cout << "y dims" << y.rows() << " " << y.cols() << endl;

//        cout << y << endl;

        Eigen::VectorXd x_hat = Eigen::VectorXd::Zero(num_features);
        for (int k = 0; k < num_eigenvectors_to_use; k++)
        {
            x_hat += y(0, k) * transformed_eigens.col(k);
        }

        //make sure to divide by 255 so they are between 0 and 1
        Eigen::VectorXd diff = (original_data.col(sample_to_reconstruct) - (x_hat + x_bar)) / 255.;

//        cout << original_data.col(sample_to_reconstruct).normalized() << endl;

//        cout << diff << endl;

        cout << "ERROR: " << diff.norm() << endl;

        view_vector_as_image(x_hat + x_bar, num_rows, num_cols);
    }
    //experiment A
    if (D)
    {
        Eigen::VectorXd eigenvalues = read_matrix_into_eigen("eigenvalues.mat", "eigenvalues");
//        cout << eigenvalues << endl;
        cout << eigenvalues.sum() << endl;
//        cout << eigenvalues(0) / eigenvalues.sum() << endl;
        //variance explained
        Eigen::VectorXd varExp = eigenvalues / eigenvalues.sum();
        cout << varExp(0) << endl;
        cout << varExp(1) << endl;
        //cumulative variance explained
        Eigen::VectorXd cumVar(varExp.size());
        double val = 0;
        for (int i = 0; i < varExp.size(); i++)
        {
            val += varExp(i);
            cumVar(i) = val;
        }
        cout << cumVar(varExp.size() - 2) << endl;
        cout << cumVar(varExp.size() - 1) << endl;

        int num_cmp_needed = -1;
        for (int i = 0; i < cumVar.size(); i++)
        {
            if (cumVar(i) > 0.8)
            {
                num_cmp_needed = i + 1;
                break;
            }
        }
        cout << "To explain 0.8, we need " << num_cmp_needed << " of " << cumVar.size() << " components.";

        auto file_name_list = get_filenames("Faces_FA_FB/fa_H");
        cv::Mat first_sample = cv::imread("Faces_FA_FB/fa_H/00001_930831_fa_a.pgm", cv::IMREAD_GRAYSCALE);
        int num_rows = first_sample.rows;
        int num_cols = first_sample.cols;
        int num_features = num_rows * num_cols;
        int num_samples = file_name_list.size();
        // Reads in all hd faces and stores them unflattened in a vector
        //num_features x num_samples (NxM) matrix
        Eigen::MatrixXd original_data = Eigen::MatrixXd(num_features, num_samples);
        Eigen::VectorXd identities = Eigen::VectorXd(num_samples);
        for (int i = 0; i < num_samples; i++)
        {
            string name = file_name_list[i];
            cv::Mat im = imread(name,cv::IMREAD_GRAYSCALE);
            Eigen::MatrixXd mat;
            cv::cv2eigen(im, mat);
            mat /= 255.;
            original_data.col(i) = mat.reshaped();
//            cout << filesystem::path(name).filename().generic_string().substr(0,5) << endl;
            identities(i) = stoi(filesystem::path(name).filename().generic_string().substr(0,5));
        }
        cout << "Original Data matrix: " << original_data.rows() << " " << original_data.cols() << endl;
//        view_vector_as_image(original_data.col(0), num_rows, num_cols);

        //Sample Mean
        //step 1
        Eigen::VectorXd x_bar = Eigen::VectorXd::Zero(num_features);
        for (int i = 0; i < num_samples; i++)
        {
            x_bar += original_data.col(i);
        }
        x_bar /= num_samples;

        //View average face
//        view_vector_as_image(x_bar, num_rows, num_cols);


        Eigen::MatrixXd transformed_eigens = read_matrix_into_eigen("transformed_eigens.mat", "transformed_eigens");

        //step 2
        Eigen::MatrixXd centered_data = Eigen::MatrixXd(num_features, num_samples);
        for (int i = 0; i < num_samples; i++)
        {
            centered_data.col(i) = original_data.col(i) - x_bar;
        }

        //project data onto the top k eigenvectors
        Eigen::MatrixXd top_k_eigens = transformed_eigens(Eigen::all, Eigen::seq(0, num_cmp_needed-1));
        cout << "TOP K SIZE" << top_k_eigens.rows() << " " << top_k_eigens.cols() << endl;

        Eigen::MatrixXd training_data_projected = centered_data.transpose() * top_k_eigens;
        cout << "Projected " << training_data_projected.rows() << " " << training_data_projected.cols() << endl;

        Eigen::VectorXd top_k_eigenvals = eigenvalues(Eigen::seq(0, num_cmp_needed-1));

        double dist = mahalanobis(training_data_projected.row(0),
                                  training_data_projected.row(0),
                                  top_k_eigenvals);
        cout << "DIST: (to self)" << dist << endl;
        double dist2 = mahalanobis(training_data_projected.row(0),
                                   training_data_projected.row(2),
                                   top_k_eigenvals);
        cout << "DIST2: (diff pics)" << dist2 << endl;
        cout << "IDENTITIES: " << identities(0) << ", " << identities(2) << endl;
        double dist3 = mahalanobis(training_data_projected.row(2),
                                   training_data_projected.row(3),
                                   top_k_eigenvals);
        cout << "DIST3: (same identity)" << dist3 << endl;
        cout << "IDENTITIES: " << identities(2) << ", " << identities(3) << endl;

        //Recognition: load in test imgs

    }

    return 0;
}