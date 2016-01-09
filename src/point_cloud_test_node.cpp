#include <thread>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_representation.h>

#include <pcl/filters/voxel_grid.h>

#include <pcl/features/normal_3d.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <ros/package.h>
#include <ros/ros.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
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

int compareX(const void* a, const void* b) {
  float arg1 = ((float3*) a)->x;
  float arg2 = ((float3*) b)->x;
  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;
}

int compareY(const void* a, const void* b) {
  float arg1 = ((float3*) a)->y;
  float arg2 = ((float3*) b)->y;
  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;
}

int compareZ(const void* a, const void* b) {
  float arg1 = ((float3*) a)->z;
  float arg2 = ((float3*) b)->z;
  if (arg1 < arg2) return -1;
  if (arg1 > arg2) return 1;
  return 0;
}

const int float3size = sizeof(float3);

Node* createKdTree(float3* pointList, int numPoints, int depth=0) {
  if (numPoints == 0) return NULL;

  int axis = depth % 3;

  if (pointList == NULL) {
    printf("Null point list");
  }

  switch (axis) {
    case 0:
      std::qsort(pointList, numPoints, sizeof(float3*), compareX);
      break;
    case 1:
      std::qsort(pointList, numPoints, sizeof(float3*), compareY);
      break;
    case 2:
      std::qsort(pointList, numPoints, sizeof(float3*), compareZ);
      break;
  }

  int median = numPoints / 2;

  Node* node = new Node;
  node->location = &pointList[median];
  node->left = createKdTree(pointList, median, depth + 1);
  node->right = createKdTree(pointList + (median + 1), numPoints - median - 1, depth + 1);
  return node;
}

void printNodes(Node* node) {
  if (node == NULL) return;
  float3 location = *node->location;
  printf("(%f, %f, %f)\n", location.x, location.y, location.z);
  printNodes(node->left);
  printNodes(node->right);
}

void bilateralFilter(const cv::Mat& src_host, cv::Mat& dst_host, Node* hostKdTree);
void bilateralFilter2(const float3* pointList, int n);

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile(ros::package::getPath("point_cloud_test") + "/data/0000_cloud.pcd", *cloud) == -1) {
    PCL_ERROR("Couldn't read file data/0000_cloud.pcd\n");
    return (-1);
  }

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

  float3* pts = new float3[cloud->points.size()];
#pragma omp parallel for
  for (int i = 0; i < cloud->points.size(); ++i, ++pts) {
    pcl::PointXYZ p = cloud->points.at(0);
    float3 newP;
    newP.x = p.x;
    newP.y = p.y;
    newP.z = p.z;
    pts[i] = newP;
  }

  bilateralFilter2(pts, cloud->points.size());

  /*
  cv::Mat bilateralFilterMatSrc, bilateralFilterMatDst;
  bilateralFilterMatSrc = cv::Mat::zeros(cloud->height, cloud->width, CV_32FC3);
  bilateralFilterMatDst = cv::Mat::zeros(cloud->height, cloud->width, CV_32FC3);
  cloudToMat(cloud, bilateralFilterMatSrc);

  int numPoints = bilateralFilterMatSrc.rows * bilateralFilterMatSrc.cols;
  float3** pointList = new float3*[numPoints];
  for (int row = 0; row < bilateralFilterMatSrc.rows; ++row) {
    int base = row * bilateralFilterMatSrc.cols;
    float3* rowPtr = bilateralFilterMatSrc.ptr<float3>(row);
    for (int col = 0; col < bilateralFilterMatSrc.cols; ++col, ++rowPtr) {
      pointList[base + col] = rowPtr;
    }
  }
  Node* root = createKdTree(pointList, numPoints);

  printNodes(root);
  bilateralFilter(bilateralFilterMatSrc, bilateralFilterMatDst, root);

  pcl::PointCloud<pcl::PointXYZ>::Ptr bilateralFilterCloud(new pcl::PointCloud<pcl::PointXYZ>);
  matToCloud(bilateralFilterMatDst, bilateralFilterCloud);

  ROS_INFO("Bilateral filter cloud point count: %zu", bilateralFilterCloud->points.size());
  visualize(cloud, bilateralFilterCloud);
  */

  return (0);
}