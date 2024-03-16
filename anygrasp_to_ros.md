## Rosbridge
https://github.com/RobotWebTools/rosbridge_suite?tab=readme-ov-file

https://wiki.ros.org/rosbridge_suite

1. install ```ros-humble-rosbridge-suite```
2. ```ros2 launch /opt/ros/humble/share/rosbridge_server/launch/rosbridge_websocket_launch.xml```
3. write a python websocket to send message to ```ws://127.0.0.1:9090```