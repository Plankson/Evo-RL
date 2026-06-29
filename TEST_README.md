cd Evo-RL
conda activate evo-rl

然后先配置can口：
bash can_config.sh

## 目前是3个can口了。用这个。

bash can_config_3.sh

打开test_policy_record.sh
PROMPT: 改为当前需要的
POLICY_NAME: pi05就用pi0, 我们自己模型就用ace_policy
PORT: 对应eip端口

args参数：
--policy.host: 默认是cci部署103.237.28.254，如果是本地部署改为169.254.118.66
--policy.chunk_size: 默认是50
--policy.n_action_steps: 默认是24

改好之后bash test_policy_record.sh即可开始测试，ctrl+c结束测试

bash reset_to_default.sh 进行复位

