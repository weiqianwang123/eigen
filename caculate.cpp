#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace Eigen;
using namespace std;
using namespace cv;



#define MATRIX_SIZE 50
int main()
{
//向量
Vector3d v_3d;
//一般矩阵
Matrix<float,2,3> matrix_23;
//方阵,并且初始化为0
Matrix3d matrix_33 = Matrix3d::Zero();
//不知道大小的矩阵
Matrix<double,Dynamic,Dynamic> matrix_dynamic;
//矩阵输入
matrix_23 << 1,2,3,4,5,6;
//矩阵输出
cout<<matrix_23<<endl;
//访问矩阵元素
matrix_23(1,1) = 0;
//矩阵和向量相乘，二者内部数据类型必须一致
Matrix<double,2,1> result = matrix_23.cast<double>()*v_3d;
//随机数矩阵
matrix_33 = Matrix3d ::Random();
//转置
//matrix_33 << matrix_33.transpose();
//求矩阵各个元素和
cout<<matrix_33.sum()<<endl;
//求迹
cout<<matrix_33.trace()<<endl;
//求逆
cout<<matrix_33.inverse()<<endl;
//求行列式
cout<<matrix_33.determinant()<<endl;
//求特征值和特征向量，此处输入参数为一个矩阵乘以自身转置得到的实对称矩阵
SelfAdjointEigenSolver<Matrix3d> eigen_solver (matrix_33*matrix_33.transpose());
cout<<eigen_solver.eigenvalues()<<endl;
cout<<eigen_solver.eigenvectors()<<endl;
//解方程
Matrix<double,MATRIX_SIZE,MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
matrix_NN = matrix_NN*matrix_NN.transpose();
Matrix<double,MATRIX_SIZE,1> v_Nd = MatrixXd::Random(MATRIX_SIZE,1);
clock_t time_stt = clock();
//直接求逆法
Matrix<double,MATRIX_SIZE,1> x = matrix_NN.inverse()*v_Nd;
cout<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
cout<<x.transpose()<<endl;
//矩阵分解法中的QR分解
time_stt= clock();
x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
cout<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
cout<<x.transpose()<<endl;
//针对正定矩阵的cholesky分解
time_stt = clock();
x = matrix_NN.ldlt().solve(v_Nd);
cout<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
cout<<x.transpose()<<endl;
}
