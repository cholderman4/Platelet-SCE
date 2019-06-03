#ifndef BUCKET_SORT_H_
#define BUCKET_SORT_H_

#include "SystemStructures.h"

void initialize_bucket_dimensions(
    MembraneNode& memNode,
    Node& intNode,
    DomainParams& domainParams);

void set_bucket_grids(
    MembraneNode& memNode,
    Node& intNode,
    DomainParams& domainParams,
    BucketScheme& bucketScheme);

void assign_nodes_to_buckets(
    MembraneNode& memNode,
    Node& intNode,
    DomainParams& domainParams,
    BucketScheme& bucketScheme);

void extend_to_bucket_neighbors(
    MembraneNode& memNode,
    Node& intNode,
    DomainParams& domainParams,
    BucketScheme& bucketScheme);

#endif
