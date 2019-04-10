#include <cuda.h>
#include <cstdlib>
#include <random>
#include <set>
#include <list>
#include <vector>
#include <memory>
#include "PlatletSystemBuilder.h"
#include "PlatletSystem.h"
# define M_PI 3.14159265358979323846  /* pi */


PlatletSystemBuilder::SystemBuilder(double _epsilon, double _dt):
	epsilon(_epsilon), dt(_dt) {}

PlatletSystemBuilder::~SystemBuilder() {
}

