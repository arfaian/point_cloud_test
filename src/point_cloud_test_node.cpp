#include <thread>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <ros/package.h>


void visualize(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  const std::string cloudName = "rendered";

  pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
  visualizer->initCameraParameters();
  visualizer->setBackgroundColor(0, 0, 0);
  visualizer->setPosition(cloud->height, 0);
  visualizer->setSize(cloud->height, cloud->width);
  visualizer->setShowFPS(true);
  visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);

  while (!visualizer->wasStopped()) {
    if (!visualizer->updatePointCloud(cloud, cloudName)) {
      visualizer->addPointCloud(cloud, cloudName);
      visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
    }
    visualizer->spinOnce(100);
  }
  visualizer->close();
}

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile(ros::package::getPath("point_cloud_test") + "/data/wolf.pcd", *cloud) == -1) {
    PCL_ERROR("Couldn't read file data/wolf.pcd\n");
    return (-1);
  }

  visualize(cloud);

  return (0);
}