#include <thread>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <stdlib.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <vector_types.h>

#include "node.h"

void cloudToMat(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, cv::Mat &mat) {
#pragma omp parallel for
  for (int row = 0; row < cloud->height; row++) {
    for (int col = 0; col < cloud->width; col++) {
      pcl::PointXYZ point = cloud->at(col, row);
      mat.at<float3>(row, col).x = point.x;
      mat.at<float3>(row, col).y = point.y;
      mat.at<float3>(row, col).z = point.z;
    }
  }
}

void matToCloud(cv::Mat &mat, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  cloud->height = mat.rows;
  cloud->width = mat.cols;
  cloud->is_dense = false;
  cloud->points.resize(cloud->height * cloud->width);

#pragma omp parallel for
  for (int row = 0; row < mat.rows; row++) {
    for (size_t col = 0; col < (size_t) mat.cols; col++) {
      const size_t index = mat.cols * row + col;
      cloud->points[index].x = mat.at<float3>(row, col).x;
      cloud->points[index].y = mat.at<float3>(row, col).y;
      cloud->points[index].z = mat.at<float3>(row, col).z;
    }
  }
}

void visualize(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &filteredCloud) {
  const std::string cloudName = "unfiltered";
  const std::string filteredCloudName = "filtered";
  int viewport0 = 0;
  int viewport1 = 1;

  pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));

  visualizer->createViewPort(0.5, 0.0, 1.0, 1.0, viewport0);
  visualizer->setBackgroundColor(0, 0, 0, viewport0);
  visualizer->addText(cloudName, 10, 10, "right", viewport0);
  visualizer->addPointCloud(cloud, cloudName, viewport0);
  visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);

  visualizer->createViewPort(0.0, 0.0, 0.5, 1.0, viewport1);
  visualizer->setBackgroundColor(0.1, 0.1, 0.1, viewport1);
  visualizer->addText(filteredCloudName, 10, 10, "left", viewport1);
  visualizer->addPointCloud(filteredCloud, filteredCloudName, viewport1);
  visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, filteredCloudName);

  visualizer->initCameraParameters();

  while (!visualizer->wasStopped()) {
    visualizer->spinOnce(100);
  }

  visualizer->close();
}

int compare(const void* a, const void* b) {
  return (0);
}

void createKdTree(unsigned char* pointList, int numPoints, int depth=0) {
  int axis = depth % 3;

  int median = numPoints / 2;

  Node node;
  //node.
}

void bilateralFilter(const cv::Mat& src_host, cv::Mat& dst_host);

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile(ros::package::getPath("point_cloud_test") + "/data/0000_cloud.pcd", *cloud) == -1) {
    PCL_ERROR("Couldn't read file data/0000_cloud.pcd\n");
    return (-1);
  }

  cv::Mat bilateralFilterMatSrc, bilateralFilterMatDst;
  bilateralFilterMatSrc = cv::Mat::zeros(cloud->height, cloud->width, CV_32FC3);
  bilateralFilterMatDst = cv::Mat::zeros(cloud->height, cloud->width, CV_32FC3);
  cloudToMat(cloud, bilateralFilterMatSrc);
  bilateralFilter(bilateralFilterMatSrc, bilateralFilterMatDst);

  unsigned char* pointList = bilateralFilterMatSrc.ptr(0);
  qsort(pointList, bilateralFilterMatSrc.rows * bilateralFilterMatSrc.cols, sizeof(float3), compare);
  createKdTree(pointList);

  pcl::PointCloud<pcl::PointXYZ>::Ptr bilateralFilterCloud(new pcl::PointCloud<pcl::PointXYZ>);
  matToCloud(bilateralFilterMatDst, bilateralFilterCloud);

  ROS_INFO("Bilateral filter cloud point count: %zu", bilateralFilterCloud->points.size());
  visualize(cloud, bilateralFilterCloud);

  return (0);
}