#!/bin/bash
root_path=/home/helloworld/Desktop/radar_2024_developing_version/RadarStation2024_two_layers_three_cam
livox_path=/home/helloworld/Desktop/radar_2024_developing_version/RadarStation2024_two_layers_three_cam
cd $root_path 
gnome-terminal --geometry 60x20+10+10 -- bash $livox_path/start.sh
gnome-terminal --geometry 60x20+10+10 -- bash main.sh
