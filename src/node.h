//
// Created by arian on 1/6/16.
//

#ifndef POINT_CLOUD_TEST_NODE_H
#define POINT_CLOUD_TEST_NODE_H

#include <vector_types.h>

struct Node {
  const int step = sizeof(float3);
  int pointOffset;
  int numberOfPoints;
  int leftOffset;
  int rightOffset;
};

#endif //POINT_CLOUD_TEST_NODE_H
