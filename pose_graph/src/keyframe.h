#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;

// 构建Brief产生器，用于通过Brief模板文件对图像特征点计算Brief描述子
class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};


class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	bool findConnection(KeyFrame* old_kf);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();


	double time_stamp;    			// 关键帧时间戳
	int index;						// 关键帧索引
	int local_index;  				// 四自由度优化中的索引
	Eigen::Vector3d vio_T_w_i; 		// 里程计位置
	Eigen::Matrix3d vio_R_w_i; 		// 里程计姿态
	Eigen::Vector3d T_w_i;			// 回环后的位置
	Eigen::Matrix3d R_w_i;			// 回环后的姿态
	Eigen::Vector3d origin_vio_T;	// 原始VIO结果的位置，后端发送过来的关键帧位置
	Eigen::Matrix3d origin_vio_R;	// 原始VIO结果的姿态，后端发送过来的关键帧姿态
	cv::Mat image;					// 原始图像
	cv::Mat thumbnail;				// 缩小尺寸后的图像
	vector<cv::Point3f> point_3d;  					// 关键帧对应的3D点
	vector<cv::Point2f> point_2d_uv;  				// 特征点的像素坐标
	vector<cv::Point2f> point_2d_norm;  			// 特征点的归一化坐标
	vector<double> point_id;						// 特征点的id
	vector<cv::KeyPoint> keypoints;					// fast角点的像素坐标
	vector<cv::KeyPoint> keypoints_norm;			// fast角点对应的归一化相机系坐标
	vector<cv::KeyPoint> window_keypoints;   		// 原来光流追踪的特征点的像素坐标
	vector<BRIEF::bitset> brief_descriptors;		// 额外提取的fast特征点的描述子
	vector<BRIEF::bitset> window_brief_descriptors;	// 原来光流追踪的特征点的描述子
	bool has_fast_point;  							// 是否提取了fast角点
	int sequence;									// 关键帧所在序列号
	bool has_loop;  								//是否检测到回环
	int loop_index;  								//初始化为-1
	Eigen::Matrix<double, 8, 1 > loop_info;			// 回环帧和当前帧的相对位置信息
};

