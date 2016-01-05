#include <thread>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <ros/package.h>

void cloudToMat(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, cv::Mat &mat) {
#pragma omp parallel for
  for (int row = 0; row < cloud->height; row++) {
    for (int col = 0; col < cloud->width; col++) {
      pcl::PointXYZ point = cloud->at(col, row);
      mat.at<cv::Vec3f>(row, col)[0] = point.x;
      mat.at<cv::Vec3f>(row, col)[1] = point.y;
      mat.at<cv::Vec3f>(row, col)[2] = point.z;
    }
  }
}

void matToCloud(cv::Mat &mat, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  cloud->height = mat.rows;
  cloud->width = mat.cols;
  cloud->is_dense = false;
  cloud->points.resize(cloud->height * cloud->width);

#pragma omp parallel for
  for (int row = 0; row < mat.rows; ++row) {
    pcl::PointXYZ *pointXYZ = &cloud->points[row * mat.cols];
    const cv::Vec3f *vec = mat.ptr<cv::Vec3f>(row);
    for (size_t col = 0; col < (size_t) mat.cols; ++pointXYZ, ++vec) {
      pointXYZ->x = vec->val[0];
      pointXYZ->y = vec->val[1];
      pointXYZ->z = vec->val[2];
    }
  }
}

void visualize(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &filteredCloud) {
  const std::string cloudName = "unfiltered";
  const std::string filteredCloudName = "filtered";

  pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
  visualizer->initCameraParameters();
  visualizer->setBackgroundColor(0, 0, 0);
  visualizer->setPosition(cloud->height, 0);
  visualizer->setSize(cloud->height, cloud->width);
  visualizer->setShowFPS(true);
  visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
  visualizer->addPointCloud(cloud, cloudName);
  visualizer->addPointCloud(filteredCloud, filteredCloudName);
  visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);

  while (!visualizer->wasStopped()) {
    visualizer->spinOnce(100);
  }

  visualizer->close();
}

void bilateralFilter(const cv::Mat& src_host, cv::Mat& dst_host);

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile(ros::package::getPath("point_cloud_test") + "/data/0000_cloud.pcd", *cloud) == -1) {
    PCL_ERROR("Couldn't read file data/0000_cloud.pcd\n");
    return (-1);
  }

  cv::Mat bilateralFilterMatSrc, bilateralFilterMatDst;
  bilateralFilterMatSrc = cv::Mat::zeros(cloud->height, cloud-> width, CV_32FC3);
  bilateralFilterMatDst = cv::Mat::zeros(cloud->height, cloud-> width, CV_32FC3);
  cloudToMat(cloud, bilateralFilterMatSrc);
  bilateralFilter(bilateralFilterMatSrc, bilateralFilterMatSrc);

  pcl::PointCloud<pcl::PointXYZ>::Ptr bilateralFilterCloud(new pcl::PointCloud<pcl::PointXYZ>);
  matToCloud(bilateralFilterMatDst, bilateralFilterCloud);

  visualize(cloud, bilateralFilterCloud);

  return (0);
}