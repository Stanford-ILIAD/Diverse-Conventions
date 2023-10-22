# cp ../overcooked_flask/static/assets/pbt_cramped_room_agent/* tf_results/pbt_cramped_room_agent/
# MADRONA_MWGPU_KERNEL_CACHE=/tmp/simplecookedcache python tfjs_to_torch.py --env_name overcooked --seed 1 --over_layout simple --run_dir tf_pbt --n_rollout_threads 1 --layer_N 2 --hidden_size 64 --full_dir torch_results

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

import tf2onnx


class Policy(nn.Module):

    def __init__(self, actor):
        super(Policy, self).__init__()

        self.base = actor.base.cnn.cnn
        self.act_layer = actor.act.action_out.linear

    def forward(self, x: torch.Tensor):
        x = self.base(x.permute((0, 3, 1, 2)))
        x = self.act_layer(x)
        return nn.functional.softmax(x, dim=1)

config = get_config()
config.add_argument("--ai_name", type=str)
config.add_argument("--full_dir", type=str)
args = config.parse_args()

envs = generate_env(args.env_name, args.n_rollout_threads, args.over_layout, use_env_cpu=True)

args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name

run_dir = args.full_dir

flask_dir = Path(
        os.path.dirname(os.path.abspath(__file__))
        # + "/../overcooked_flask/static/assets/"
        + "/tf_results/"
    )
os.makedirs(run_dir, exist_ok=True)
run_dir = Path(run_dir)

args.model_dir = str(run_dir / 'models')

config = {
    'all_args': args,
    'envs': envs,
    'device': 'cpu',
    'num_agents': 2,
    'run_dir': run_dir
}
# ego = MainPlayer(config)
# ego.restore()

### TFJS to TF
exported_name = args.ai_name
torch_model_path = str(run_dir / "models" / f"{exported_name}.pt")
onnx_model_path = str(run_dir / "models" / f"{exported_name}.onnx")
tf_model_dir = str(run_dir / 'models' / f'{exported_name}')
# os.system(tfjs_convert_command)

import tensorflow as tf
print(tf_model_dir)
tf_model = tf.saved_model.load(tf_model_dir)
print(tf_model)




vobs, _ = envs.n_reset()
obs = vobs.obs.to(dtype=torch.float)
print(obs.shape)

onnx_convert_command = f"""python -m tf2onnx.convert
                 --saved-model "{tf_model_dir}" 
                 --output "{onnx_model_path}"
                 --opset 13
                 """
onnx_convert_command = " ".join(onnx_convert_command.split())
print(onnx_convert_command)
os.system(onnx_convert_command)

onnx_model = onnx.load(onnx_model_path)

import torch
from onnx2torch import convert

obs = obs.expand(30, -1, -1, -1)
print(obs.shape)
torch_model = convert(onnx_model_path)
print(torch_model(obs))

torch.save(torch_model, torch_model_path)
# print(onnx_model)

# pbt_cramped_room_agent
# tensorflowjs_converter --input_format=tfjs_graph_model --output_format=tf_saved_model --signature_name=serving_default --saved_model_tags=serve "/iliad/u/bidiptas/CoMeDi/train/tf_results/pbt_cramped_room_agent" "torch_results/pbt_cramped_room_agent/models/pbt_cramped_room_agent_tf"
