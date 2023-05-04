# python trainer.py --num_env_steps 1000000 --pop_size 1 --xp_weight 0.5 --mp_weight 0.0 --lr 2.0e-4 --critic_lr 2.0e-4 --episode_length 200 --env_length 200 --use_linear_lr_decay --entropy_coef 0.0 --env_name overcooked --seed 1 --over_layout simple --run_dir simple_sp --restored 0 --n_rollout_threads 50 --ppo_epoch 10 --hidden_size 512
# python torch_to_tfjs.py --env_name overcooked --seed 1 --over_layout simple --run_dir simple_sp --n_rollout_threads 1
# cp simple/results/simple_sp/1/models/AI_S_cramped_room_agent-tfjs-fp32/* ../PantheonRL/overcookedgym/overcooked-flask/static/assets/ppo_bc_cramped_room_agent/


from MAPPO.main_player import MainPlayer

from config import get_config
import os
from pathlib import Path

from env_utils import generate_env

import torch
import torch.nn as nn

from torch.distributions import Categorical

import onnx
from onnx_tf.backend import prepare


class Policy(nn.Module):

    def __init__(self, actor):
        super(Policy, self).__init__()

        self.base = actor.base.cnn.cnn
        self.act_layer = actor.act.action_out.linear

    def forward(self, x: torch.Tensor):
        x = self.base(x.permute((0, 3, 1, 2)))
        x = self.act_layer(x)
        return nn.functional.softmax(x, dim=1)


args = get_config().parse_args()

envs = generate_env(args.env_name, args.n_rollout_threads, args.over_layout)

args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name

run_dir = (
        os.path.dirname(os.path.abspath(__file__))
        + "/"
        + args.hanabi_name
        + "/results/"
        + (args.run_dir)
        + "/"
        + str(args.seed)
    )
os.makedirs(run_dir, exist_ok=True)
with open(run_dir + "/" + "args.txt", "w", encoding="UTF-8") as file:
    file.write(str(args))

run_dir = Path(run_dir)

args.model_dir = str(run_dir / 'models')

config = {
    'all_args': args,
    'envs': envs,
    'device': 'cpu',
    'num_agents': 2,
    'run_dir': run_dir
}
ego = MainPlayer(config)
ego.restore()

# print(ego.policy.actor)

torch_network = Policy(ego.policy.actor)

vobs, _ = envs.n_reset()
obs = vobs.obs.to(dtype=torch.float)

print("*" * 20, " TORCH ", "*" * 20)

print(torch_network)

print(obs.shape)

print(torch_network(obs))

print("*" * 20, " ONNX ", "*" * 20)
onnx_model_path = str(run_dir / "models" / "AI_S_cramped_room_agent.onnx")

input_name = 'input'  # 'ppo_agent/ppo2_model/Ob'

torch.onnx.export(torch_network,
                  obs,
                  onnx_model_path,
                  export_params=True,
                  input_names=[input_name],
                  output_names=['output'],
                  opset_version=10)

onnx_model = onnx.load(onnx_model_path)

print(onnx_model.graph.input[0])

onnx.checker.check_model(onnx_model)

print("*" * 20, " TF ", "*" * 20)
tf_rep = prepare(onnx_model)
tf_model_dir = str(run_dir / 'models' / 'AI_S_cramped_room_agent')
tf_rep.export_graph(tf_model_dir)

tfjs_model_dir = f"{tf_model_dir}-tfjs-fp32"
tfjs_convert_command = f"""tensorflowjs_converter
                 --input_format=tf_saved_model 
                 --output_format=tfjs_graph_model 
                 --signature_name=serving_default 
                 --saved_model_tags=serve 
                 "{tf_model_dir}" 
                 "{tfjs_model_dir}"
                 """
tfjs_convert_command = " ".join(tfjs_convert_command.split())

os.system(tfjs_convert_command)


# Save to AI_S_cramped_room_agent
# Inputs are ppo_agent/ppo2_model/Ob
