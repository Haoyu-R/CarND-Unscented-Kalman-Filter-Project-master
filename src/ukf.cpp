#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Zero(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  weights_ = VectorXd::Zero(2*n_aug_+1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1;i < 2 * n_aug_ + 1;i++)
  {
      weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2*n_aug_+1);
  std::cout << "Constrcutor" << std::endl;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
    if (!is_initialized_) 
    {
        x_.fill(0);
        
        P_ = MatrixXd::Identity(n_x_, n_x_);

        if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            x_(0) = meas_package.raw_measurements_(0);
            x_(1) = meas_package.raw_measurements_(1);
        }
        else
        {
            double ro = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            x_(0) = ro * cos(phi);
            x_(1) = ro * sin(phi);
        }
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;

        return;
    }

    if (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) 
    {
        return;
    }
    else if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        return;
    }

    Prediction(double(meas_package.timestamp_ - time_us_)/1000000.0);

    time_us_ = meas_package.timestamp_;
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        std::cout << "Laser" << std::endl;
        UpdateLidar(meas_package);
        std::cout << "Updated" << std::endl;
    }
    else
    {
        std::cout << "Radar" << std::endl;
        UpdateRadar(meas_package);
        std::cout << "Updated" << std::endl;
    }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_aug_ - 2, n_aug_ - 2) = std_a_ * std_a_;
    P_aug(n_aug_ - 1, n_aug_ - 1) = std_yawdd_ * std_yawdd_;

    MatrixXd A_temp = P_aug.llt().matrixL();

    VectorXd x_aug = VectorXd::Zero(n_aug_);
    x_aug.head(n_x_) = x_;
    
    MatrixXd X_sig = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    X_sig << x_aug,
         (sqrt(lambda_ + n_aug_)*A_temp).colwise() + x_aug,
         (- sqrt(lambda_ + n_aug_) * A_temp).colwise() + x_aug;
   
    for (int i = 0;i < 2 * n_aug_ + 1; i++) 
    {
        double px = X_sig(0, i);
        double py = X_sig(1, i);
        double v = X_sig(2, i);
        double phi = X_sig(3, i);
        double phi_dot = X_sig(4, i);
        double v_a = X_sig(5, i);
        double v_phidd = X_sig(6, i);

        if (phi_dot < 0.0001) {
            Xsig_pred_(0, i) = px + v*cos(phi)*delta_t + 0.5*delta_t*delta_t*cos(phi)*v_a;
            Xsig_pred_(1, i) = py + v * sin(phi) * delta_t + 0.5 * delta_t * delta_t * sin(phi) * v_a;
        }
        else 
        {
            Xsig_pred_(0, i) = px + v/ phi_dot *(sin(phi+ phi_dot *delta_t)-sin(phi)) +  0.5 * delta_t * delta_t * cos(phi) * v_a;
            Xsig_pred_(1, i) = py + v / phi_dot * (-cos(phi + phi_dot * delta_t) + cos(phi)) + 0.5 * delta_t * delta_t * sin(phi) * v_a;
        }
        Xsig_pred_(2, i) = v + delta_t*v_a;
        Xsig_pred_(3, i) = phi + phi_dot * delta_t + 0.5 * delta_t * delta_t * v_phidd;
        Xsig_pred_(4, i) = phi_dot + delta_t * v_phidd;
    }
    //std::cout << "Xsig_pred_: " << Xsig_pred_ << "\n" << std::endl;


    x_.fill(0);
    P_.fill(0);

    for (int i = 0;i < 2*n_aug_+1;i++)
    {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    for (int i = 0;i < 2 * n_aug_ + 1;i++)
    {
        VectorXd temp = Xsig_pred_.col(i) - x_;
        if (temp(3) > M_PI)
        {
            temp(3) -= 2 * M_PI;
        }
        else if(temp(3) < -M_PI)
        {
            temp(3) += 2 * M_PI;
        }
        P_ += weights_(i) * temp * temp.transpose();
    }
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    MatrixXd Z_sig = MatrixXd::Zero(3, 2 * n_aug_ + 1);
    VectorXd z_mean = VectorXd::Zero(3);

    //std::cout << "Xsig_pred_: " << Xsig_pred_ << "\n" << std::endl;
    for (int i = 0;i < 2 * n_aug_ + 1;i++)
    {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        Z_sig(0, i) = sqrt(px*px+py*py);
        Z_sig(1, i) = atan2(py, px);
        Z_sig(2, i) = (px * cos(yaw)*v+py*sin(yaw)*v)/ Z_sig(0, i);

        z_mean += Z_sig.col(i) * weights_(i);
    }
    //std::cout << "Z_sig: " << Z_sig << "\n" << std::endl;

    MatrixXd S = MatrixXd::Zero(3, 3);

    for (int i = 0;i < 2 * n_aug_ + 1;i++) 
    {
        VectorXd temp = Z_sig.col(i) - z_mean;

        if (temp(1) > M_PI) {
            temp(1) -= 2 * M_PI;
        }
        else if (temp(1) < -M_PI)
        {
            temp(1) += 2 * M_PI;
        }

        S += weights_(i) * temp * temp.transpose();
    }

    //std::cout << "S: " << S << "\n" << std::endl;
    MatrixXd R = MatrixXd::Zero(3, 3);
    R(0, 0) = std_radr_ * std_radr_;
    R(1, 1) = std_radphi_ * std_radphi_;
    R(2, 2) = std_radrd_ * std_radrd_;

    S += R;

    MatrixXd T = MatrixXd::Zero(5, 3);
    VectorXd temp_x = VectorXd(5);
    VectorXd temp_z = VectorXd(3);

    for (int i = 0;i < 2 * n_aug_ + 1;i++) 
    {

        temp_x = Xsig_pred_.col(i) - x_;
        temp_z = Z_sig.col(i) - z_mean;
        if (temp_x(3) > M_PI) 
        {
            temp_x(3) -= 2 * M_PI;
        }
        else if (temp_x(3) < -M_PI)
        {
            temp_x(3) += 2 * M_PI;
        }

        if (temp_z(1) > M_PI)
        {
            temp_z(1) -= 2 * M_PI;
        }
        else if (temp_z(1) < -M_PI)
        {
            temp_z(1) += 2 * M_PI;
        }
        T += weights_(i) * temp_x * temp_z.transpose();
    }
  
    //std::cout << "T: " << T << "\n" << std::endl;
    MatrixXd K = T * S.inverse();

    VectorXd delta_z = meas_package.raw_measurements_ - z_mean;

    if (delta_z(1) > M_PI)
    {
        delta_z(1) -= 2 * M_PI;
    }
    else if (delta_z(1) < -M_PI)
    {
        delta_z(1) += 2 * M_PI;
    }

    x_ = x_ + K * delta_z;
    P_ = P_ - K * S * K.transpose();
    std::cout << "x_: " << x_ << "\n" << std::endl;

    double NIS_radar = delta_z.transpose() * S.inverse() * delta_z;
    std::cout << "NIS_radar: " << NIS_radar << std::endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

    MatrixXd H = MatrixXd::Zero(2, 5);
    H(0, 0) = 1;
    H(1, 1) = 1;

    VectorXd y = meas_package.raw_measurements_ - H * x_;

    MatrixXd R = MatrixXd::Zero(2, 2);

    R(0, 0) = std_laspx_ * std_laspx_;
    R(1, 1) = std_laspy_ * std_laspy_;

    MatrixXd S = H * P_ * H.transpose() + R;
    MatrixXd K = P_ * H.transpose() * S.inverse();
    MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
    x_ = x_ + K * y;
    P_ = (I - K * H) * P_;


    //MatrixXd Z_sig = MatrixXd::Zero(2, 2 * n_aug_ + 1);
    //VectorXd z_mean = VectorXd::Zero(2);

    ////std::cout << "Xsig_pred_: " << Xsig_pred_ << "\n" << std::endl;
    //for (int i = 0;i < 2 * n_aug_ + 1;i++)
    //{
    //    Z_sig(0, i) = Xsig_pred_(0, i);
    //    Z_sig(1, i) = Xsig_pred_(1, i);

    //    z_mean += Z_sig.col(i) * weights_(i);
    //}
    ////std::cout << "Z_sig: " << Z_sig << "\n" << std::endl;

    //MatrixXd S = MatrixXd::Zero(2, 2);

    //for (int i = 0;i < 2 * n_aug_ + 1;i++)
    //{
    //    VectorXd temp = Z_sig.col(i) - z_mean;
    //    S += weights_(i) * temp * temp.transpose();
    //}

    ////std::cout << "S: " << S << "\n" << std::endl;
    //MatrixXd R = MatrixXd::Zero(2, 2);

    //R(0, 0) = std_laspx_ * std_laspx_;
    //R(1, 1) = std_laspy_ * std_laspy_;

    //S += R;

    //MatrixXd T = MatrixXd::Zero(5, 2);
    //VectorXd temp_x = VectorXd(5);
    //VectorXd temp_z = VectorXd(2);

    //for (int i = 0;i < 2 * n_aug_ + 1;i++)
    //{

    //    temp_x = Xsig_pred_.col(i) - x_;
    //    temp_z = Z_sig.col(i) - z_mean;
    //    if (temp_x(3) > M_PI)
    //    {
    //        temp_x(3) -= 2 * M_PI;
    //    }
    //    else if (temp_x(3) < -M_PI)
    //    {
    //        temp_x(3) += 2 * M_PI;
    //    }

    //    T += weights_(i) * temp_x * temp_z.transpose();
    //}

    ////std::cout << "T: " << T << "\n" << std::endl;
    //MatrixXd K = T * S.inverse();

    //VectorXd delta_z = meas_package.raw_measurements_ - z_mean;

    //x_ = x_ + K * delta_z;
    //P_ = P_ - K * S * K.transpose();
    ////std::cout << "x_: " << x_ << "\n" << std::endl;

    //double NIS_lidar = delta_z.transpose() * S.inverse() * delta_z;
    //std::cout << "NIS_lidar: " << NIS_lidar << std::endl;
}