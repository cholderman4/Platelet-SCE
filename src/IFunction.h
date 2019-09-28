#ifndef IFUNCTION_H_
#define IFUNCTION_H_

// *******************************************
/* 
Used to call various functions that act on the system by calling functors on the device. Most calls to execute() will simply contain a for_each or transform that invokes a device functor.

Example: each type of force is calculated by a different device functor, called from a single IFunction object.
*/
// *******************************************


class IFunction {
    public:
    // virtual ~IFunction() {}
    virtual void execute() = 0;

    virtual ~IFunction() {};
};

#endif // IFUNCTION_H_