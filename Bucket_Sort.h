#ifndef BUCKET_SORT_H_
#define BUCKET_SORT_H_

#include "SystemStructures.h"

void initialize_bucket_dimensions(
    Node& node,
    DomainParams& domainParams);

void set_bucket_grids(
    Node& node,
    DomainParams& domainParams,
    BucketScheme& bucketScheme);

void assign_nodes_to_buckets(
    Node& node,
    DomainParams& domainParams,
    BucketScheme& bucketScheme);

void extend_to_bucket_neighbors(
    Node& node,
    DomainParams& domainParams,
    BucketScheme& bucketScheme);

#endif
