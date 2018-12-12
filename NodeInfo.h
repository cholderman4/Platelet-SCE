#ifndef NODE_INFO_H
#define NODE_INFO_H

#include <iostream>
#include <cstdio>
#include "thrust/device_vector.h"

class Edge {
    public:

    // Store ID of each node connected to edge.
    thrust::device_vector<unsigned> node_L;
    thrust::device_vector<unsigned> node_R;

       
    Edge(unsigned _E) :
        node_L(_E),
        node_R(_E)
        {
            // Set edge values (connections, length) via circular connection
            for(int i = 0; i < node_L.size(); ++i) {
                node_L[i] = i;
                node_R[i] = (i+1 >= node_L.size()) ? 0 : i+1;
            }
        }

    void printConnections() {
        std::cout << "Testing edge connections:" << std::endl;
        for(auto i = 0; i < node_L.size(); ++i) {
            std::cout << "Edge " << i << ": "
                << node_L[i] << ", "
                << node_R[i] << std::endl;
        }
    }
};


class Node {
    public:
    // Holds a set of xyz coordinates for a single node.
    thrust::device_vector<double> pos_x;
    thrust::device_vector<double> pos_y;
    thrust::device_vector<double> pos_z;
    
    thrust::device_vector<double> vel_x;
    thrust::device_vector<double> vel_y;
    thrust::device_vector<double> vel_z;

    thrust::device_vector<double> force_x;
    thrust::device_vector<double> force_y;
    thrust::device_vector<double> force_z;

    
    // Number of connections at a single node.

    Node(unsigned _N) :
        pos_x(_N),
        pos_y(_N),
        pos_z(_N),
        vel_x(_N),
        vel_y(_N),
        vel_z(_N),
        force_x(_N),
        force_y(_N),
        force_z(_N)
        {
            //Fill device vectors with test values.
            pos_x[0] = -1.0f;
            pos_y[0] = 0.0f;
            pos_z[0] = 0.0f;

            vel_x[0] = 0.0;
            vel_y[0] = 0.0;
            vel_z[0] = 0.0;

            pos_x[1] = 1.0f;
            pos_y[1] = 0.0f;
            pos_z[1] = 0.0f;

            vel_x[1] = 0.0;
            vel_y[1] = 0.0;
            vel_z[1] = 0.0;

            pos_x[2] = 0.0f;
            pos_y[2] = 1.0f;
            pos_z[2] = 0.0f;

            vel_x[2] = 0.0;
            vel_y[2] = 0.0;
            vel_z[2] = 0.0;

            /* pos_x[3] = 0.0f;
            pos_y[3] = 3.0f;
            pos_z[3] = 0.0f;

            vel_x[3] = 0.0;
            vel_y[3] = 0.0;
            vel_z[3] = 0.0; */
        }

    void printPoints() {
        std::cout << "Testing initialization of vector position:" << std::endl;
        for(auto i = 0; i < pos_x.size(); ++i) {
            std::cout << "Node " << i << ": ("
                << pos_x[i] << ", "
                << pos_y[i] << ", "
                << pos_z[i] << ")" << std::endl;
        }
    }
};

#endif