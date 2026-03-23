

# new terminal
conda activate evo-rl
#     bash ~/cobot_magic/Piper_ros_private-ros-noetic/find_all_can_port.sh
bash can_config.sh
lerobot-setup-can --mode=setup --interfaces=can_left,can_back_left,can_right,can_back_right
  
  # eval 
  lerobot-setup-can --mode=test --interfaces can_left,can_back_left,can_right,can_back_right > output.txt
