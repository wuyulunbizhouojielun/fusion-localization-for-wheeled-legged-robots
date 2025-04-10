# Set ROS melodic
source /opt/ros/melodic/setup.bash

# Start Livox ROS Driver
source  /home/helloworld/xdl/software/ws_livox/devel/setup.sh

#roslaunch lidar_ros_area area.launch 
roslaunch livox_ros_driver livox_lidar.launch

