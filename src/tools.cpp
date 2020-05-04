#include "tools.h"

using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd err = VectorXd::Zero(4);
    if (estimations.size() < 1 || estimations.size() != ground_truth.size()) {
        return err;
    }
    else {
        for (int i = 0;i < estimations.size();i++)
        {
            VectorXd temp = estimations[i] - ground_truth[i];
            temp = temp.array() * temp.array();
            err += temp;
        }
    }
    err /= estimations.size();
    err = err.array().sqrt();
    
    return err;
}