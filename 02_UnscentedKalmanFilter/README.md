# ESE 650: Learning in Robotics - Homework 2[^1^][1]
This project implements an Unscented Kalman Filter (UKF) to track the orientation of a drone in three dimensions using data from an inertial measurement unit (IMU) and a motion capture system (Vicon).

## Data
The data consists of IMU readings (accelerometer and gyroscope) and Vicon ground-truth orientation for different datasets. The data files are stored in the `imu` and `vicon` folders respectively. The IMU readings are raw values that need to be calibrated using the bias and sensitivity parameters provided in the `imu_reference.pdf` file. The Vicon data contains rotation matrices that can be converted to quaternions or Euler angles[^3^][3].

## Dependencies
The project requires the following Python packages:
- numpy
- scipy
- matplotlib
- quaternion

## Usage
To run the UKF on a specific dataset, use the following command:

`python estimate_rot.py --data_num <dataset number>`

For example, to run the UKF on dataset 1, use:

`python estimate_rot.py --data_num 1`

The script will load the IMU and Vicon data, calibrate the sensors, generate sigma points, propagate the dynamics, update the measurements, and plot the results. The output plots will show the estimated quaternion, angular velocity, gyroscope readings, and Euler angles as a function of time, along with the Vicon ground-truth orientation.

## Parameters
The project uses several parameters that can be tuned for better performance. These include:
- The initial covariance of the state
- The dynamics noise covariance R
- The measurement noise covariance Q
- The threshold for quaternion averaging

The values of these parameters are defined in the `estimate_rot.py` file and can be modified as needed.
