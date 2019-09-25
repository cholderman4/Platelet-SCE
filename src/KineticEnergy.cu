#include "KineticEnergy.h"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include "NodeData.h"
#include "functor_kinetic_energy.h"


KineticEnergy::KineticEnergy(NodeData& _nodeData) :
    nodeData(_nodeData) {};


void KineticEnergy::calculate() {

    energy = thrust::transform_reduce(
        nodeData.velocity.begin(),
        nodeData.velocity.end(),
        functor_kinetic_energy(),
        0.0,
        thrust::plus< double >());
}

double KineticEnergy::getParameter() {
    return energy;
}