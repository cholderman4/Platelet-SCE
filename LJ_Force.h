#ifndef LJ_FORCE_H_
#define LJ_FORCE_H_

#include "SystemStructures.h"

void LJ_Force(
    MembraneNode& memNode,
    Node& intNode,
    BucketScheme& bucketScheme, 
    GeneralParams& generalParams);


#endif