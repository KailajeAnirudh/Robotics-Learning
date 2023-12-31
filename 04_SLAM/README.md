# SLAM with IMU and LiDAR

This project implements Simultaneous Localization and Mapping (SLAM) in an indoor environment using information from an IMU and a LiDAR sensor. The data is collected from a humanoid robot named THOR built at Penn and UCLA. Find a video about the robot [here](https://youtu.be/JhWYYuba1nE). The goal is to estimate the robot's pose and build an occupancy grid map of the surroundings.

## Requirements

- Python 3.8 or higher
- Numpy
- Matplotlib
- Scipy

## Usage

- To run the SLAM algorithm, execute `python main.py.` This will run SLAM on four datasets corresponding to four different robot trajectories in the Towne Building at Penn. The datasets are stored in the `data/train` folder as `.mat` files.
- To change the parameters of the SLAM algorithm, such as the number of particles, the sensor model, the dynamics noise, and the log-odds threshold, modify the `slam.py` file.
- To visualize the results, use the `plot.py` file. This will plot each dataset's final binarized map, particle trajectory, and odometry trajectory. It will also animate the motion of the robot and the LiDAR scans.

## Results

The SLAM algorithm uses a particle filter to estimate the pose of the robot and an occupancy grid to update the map. The algorithm performs one dynamics step and one observation step at each iteration. The dynamics step propagates the particles using the IMU data and adds some noise. The observation step updates the weights of the particles using the LiDAR data and resamples them if needed. It also updates the log odds of each cell in the map based on whether it is occupied or free according to the LiDAR scan.

These are the results for all of the maps put together.

![CombinedMaps](./01_Report/combined.gif)


## Approach

The main steps of the project were:

- Reading and processing the data: I used the functions provided in load_data.py to read the LiDAR and joint data from four different datasets. The LiDAR data contained the time stamps, the poses, and the scans of the robot. The joint data contained the angles of the head and neck of the robot. I used these angles to calculate the transformation from the body to LiDAR frames.
- Initializing the SLAM system: I used the classes map_t and slam_t defined in slam.py to initialize the SLAM system. I set up the parameters of the sensor model, such as the log-odds values for occupied and free cells and the number of particles for the particle filter. I initialized the particles with random poses around the first LiDAR pose.
- Performing the dynamics step: I implemented the function slam_t.dynamics_step to propagate the particles using the IMU data and adding some noise. I used smart_plus_2d and smart_minus_2d functions from utils.py to handle the wrap-around of angles. I checked my dynamics step by plotting the odometry trajectory and three particle trajectories, which matched well.

- Performing the observation step: I implemented the function slam_t.observation_step to update the weights, resample the particles using the LiDAR data, and update the occupancy grid map using ray casting. I used slam_t.rays2world to project the LiDAR scans into world coordinates for each particle, and calculated the log-probability of each particle based on how well it matched with the binarized map. I used scipy.stats.logsumexp to normalize the log weights and scipy.stats.resample to perform stratified resampling. I updated the log-odds values of each cell in the map according to whether it was occupied or free by a LiDAR scan and clipped them to a maximum value. I checked my observation step by plotting a single particle trajectory and its corresponding map.
- Running SLAM on all datasets: I ran SLAM on all four datasets by performing one dynamics step and one observation step at each iteration. I started SLAM only after both LiDAR and joint data were available. I plotted the final binarized map for each dataset, the particle trajectory with the largest weight, and the odometry trajectory in different colors.

The results showed that SLAM could estimate the robot's pose and build a map of its environment with reasonable accuracy. The particle filter corrected some errors in odometry by using LiDAR observations. The occupancy grid map captured some features of different rooms' walls, doors, and furniture. However, SLAM also had limitations and challenges, such as dealing with nonlinear dynamics, noisy sensors, sparse observations, computational complexity, and loop closure detection.
