
#1. start roscore
roscore

# new terminal
conda activate evo-rl
#     bash ~/cobot_magic/Piper_ros_private-ros-noetic/find_all_can_port.sh
bash can_config.sh
lerobot-setup-can --mode=setup --interfaces=can_left,can_back_left,can_right,can_back_right
  
  # eval 
  lerobot-setup-can --mode=test --interfaces can_left,can_back_left,can_right,can_back_right > output.txt
  

#2.  test 遥操
lerobot-teleoperate \
  --robot.type=bi_piper_follower \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.port=can_left \
  --robot.right_arm_config.port=can_right \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --teleop.type=bi_piper_leader \
  --teleop.id=my_bi_piper_leader \
  --teleop.left_arm_config.port=can_back_left \
  --teleop.right_arm_config.port=can_back_right \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false


#3. test 数采




lerobot-human-inloop-record \
  --robot.type=bi_piper_follower \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.port=can_left \
  --robot.right_arm_config.port=can_right \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243322070942", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243722071316", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "239622301704", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_piper_leader \
  --teleop.id=my_bi_piper_leader \
  --teleop.left_arm_config.port=can_back_left \
  --teleop.right_arm_config.port=can_back_right \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false \
  --policy.type=remote_client \
  --policy.policy_name=ace_policy \
  --policy.host= \
  --policy.port= \
  --policy.chunk_size=50 \
  --policy.n_action_steps=20 \
  --dataset.repo_id="ACE_ROBOTICS/bi-piper-fold-clothes-v0" \
  --dataset.single_task="fold_clothes" \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=100 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=true \
  --display_data=true \
  --play_sounds=false

lerobot-human-inloop-record \
  --robot.type=bi_piper_follower \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.port=can_left \
  --robot.right_arm_config.port=can_right \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243322070942", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243722071316", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "239622301704", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_piper_leader \
  --teleop.id=my_bi_piper_leader \
  --teleop.left_arm_config.port=can_back_left \
  --teleop.right_arm_config.port=can_back_right \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false \
  --policy.type=remote_client \
  --dataset.repo_id="ACE_ROBOTICS/eval_pizero-bi-piper-fold-clothes-v0" \
  --dataset.single_task="fold_clothes" \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=100 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --policy.policy_name=pi0 \
  --policy.host=103.237.28.254 \
  --policy.port=7380 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=20

lerobot-dataset-report --dataset "ACE_ROBOTICS/bi-piper-fold-clothes-v0"