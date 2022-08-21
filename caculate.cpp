#include <iostream>
#include <cmath>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include<Eigen/Geometry>
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
//cout<<matrix_23<<endl;
//访问矩阵元素
matrix_23(1,1) = 0;
//矩阵和向量相乘，二者内部数据类型必须一致
Matrix<double,2,1> result = matrix_23.cast<double>()*v_3d;
//随机数矩阵
matrix_33 = Matrix3d ::Random();
//转置
//matrix_33 << matrix_33.transpose();
//求矩阵各个元素和
//cout<<matrix_33.sum()<<endl;
//求迹
//cout<<matrix_33.trace()<<endl;
//求逆
//cout<<matrix_33.inverse()<<endl;
//求行列式
//cout<<matrix_33.de，但是terminant()<<endl;
//求特征值和特征向量，此处输入参数为一个矩阵乘以自身转置得到的实对称矩阵
SelfAdjointEigenSolver<Matrix3d> eigen_solver (matrix_33*matrix_33.transpose());
//cout<<eigen_solver.eigenvalues()<<endl;
//cout<<eigen_solver.eigenvectors()<<endl;
//解方程
Matrix<double,MATRIX_SIZE,MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
matrix_NN = matrix_NN*matrix_NN.transpose();
Matrix<double,MATRIX_SIZE,1> v_Nd = MatrixXd::Random(MATRIX_SIZE,1);
clock_t time_stt = clock();
//直接求逆法
Matrix<double,MATRIX_SIZE,1> x = matrix_NN.inverse()*v_Nd;
//cout<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
//cout<<x.transpose()<<endl;
//矩阵分解法中的QR分解
time_stt= clock();
x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
//cout<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
//cout<<x.transpose()<<endl;
//针对正定矩阵的cholesky分解
time_stt = clock();
x = matrix_NN.ldlt().solve(v_Nd);
//cout<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
//cout<<x.transpose()<<endl;



//旋转矩阵和普通矩阵的表示方法相同
Matrix3d rotation_matrix = Matrix3d::Identity();
//旋转向量使用AngleAxis,运算可以当作矩阵
AngleAxisd rotation_vector(M_PI/4,Vector3d(0,0,1));   //代表着沿z轴旋转45度
//可以将旋转向量转化成矩阵,方法有以下两种
//cout<<rotation_vector.matrix()<<endl;
rotation_matrix = rotation_vector.toRotationMatrix();
//利用旋转向量进行变换
Vector3d v(1,0,0);
Vector3d v_rotated = rotation_vector*v;
//cout<<v_rotated.transpose()<<endl;
//利用旋转矩阵进行变换
v_rotated = rotation_matrix*v;
//cout<<v_rotated.transpose()<<endl;
//欧拉角,顺序为z,y,x，即roll,pitch,yaw，可以将旋转矩阵直接转换成欧拉角
Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0);
//cout<< euler_angles.transpose();
//进行一次欧氏变换（旋转加平移）,这里写的是3d,这里的T的matrix为变换矩阵省略下面那一行
Isometry3d T = Isometry3d ::Identity();
T.rotate(rotation_vector);//旋转按照之前的rotation_vector
T.pretranslate(Vector3d(1,3,4));//设定平移向量(1,3,4)
//cout<<T.matrix()<<endl;
//利用变换矩阵进行变换
Vector3d v_transformed = T*v; //这里的*经过了重载，可以理解为先通过之前设定的旋转方式（可能是四元数等，再加上平移向量）
//cout<<v_transformed.transpose()<<endl;
//四元数
//可以直接将旋转向量赋值给四元数,顺序为x,y,z,w
Quaterniond q = Quaterniond (rotation_vector);
//cout<<q.coeffs().transpose();
//也可以用旋转矩阵赋值
q = Quaterniond (rotation_matrix);
q.normalize();//对四元数的归一化，未归一化的四元数对应的旋转矩阵可能不是正交矩阵
cout<<q.coeffs().transpose();
//使用四元数旋转一个向量，注意这里的乘法是重载的，真实的乘法应该是q乘v再乘q的逆。
v_rotated = q*v;
//cout<<v_rotated.transpose()<<endl;















}
