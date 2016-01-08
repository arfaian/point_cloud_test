//
// Created by arian on 1/6/16.
//

#ifndef POINT_CLOUD_TEST_NODE_H
#define POINT_CLOUD_TEST_NODE_H

#include <vector_types.h>

struct Node {
  float3* location;
  Node* left;
  Node* right;
};

#endif //POINT_CLOUD_TEST_NODE_H
