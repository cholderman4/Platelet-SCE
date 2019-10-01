

struct functor_slice_predicate {
    double threshold;
    bool isTop;

    __host__ __device__
    functor_slice_predicate(
        double& _threshold,
        bool _isTop) :

        threshold(_threshold),
        isTop(_isTop) {}

    __host__ __device__
    bool operator() (const double& value) {
        if (isTop == true) {
            return (value >= threshold);
        } else {
            return (value <= threshold);
        }
    }
};