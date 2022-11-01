import os
import io
import json
import copy
import argparse
import torch
import numpy as np
import gym

from flask import Flask, jsonify, request
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

from overcooked_utils import NAME_TRANSLATION
from pantheonrl.common.trajsaver import SimultaneousTransitions

from overcooked_env import DecentralizedOvercooked

from partner_agents import DecentralizedAgent
from MAPPO.r_actor_critic import R_Actor
from config import get_config

app = Flask(__name__)

MLPs = {}
MDPs = {}
POLICIES = {}


def get_prediction(s, policy):
    s = torch.tensor(s).unsqueeze(0).float()
    actions = policy.predict(observation=s)
    return int(actions[0])


def process_state(state_dict, layout_name):
    def object_from_dict(object_dict):
        return ObjectState(**object_dict)

    def player_from_dict(player_dict):
        held_obj = player_dict.get("held_object")
        if held_obj is not None:
            player_dict["held_object"] = object_from_dict(held_obj)
        return PlayerState(**player_dict)

    def state_from_dict(state_dict):
        state_dict["players"] = [player_from_dict(
            p) for p in state_dict["players"]]
        object_list = [object_from_dict(o)
                       for _, o in state_dict["objects"].items()]
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        return OvercookedState(**state_dict)

    state = state_from_dict(copy.deepcopy(state_dict))
    MDP = MDPs[layout_name]
    MLP = MLPs[layout_name]
    return MDP.featurize_state(state, MLP)


def convert_traj_to_simultaneous_transitions(traj_dict, layout_name):

    ego_obs = []
    alt_obs = []
    ego_act = []
    alt_act = []
    flags = []

    for state_list in traj_dict['ep_states']:  # loop over episodes
        ego_obs.append([process_state(state, layout_name)[0]
                        for state in state_list])
        alt_obs.append([process_state(state, layout_name)[1]
                        for state in state_list])

        # check pantheonrl/common/wrappers.py for flag values
        flag = [0 for state in state_list]
        flag[-1] = 1
        flags.append(flag)

    for action_list in traj_dict['ep_actions']:  # loop over episodes
        ego_act.append([joint_action[0] for joint_action in action_list])
        alt_act.append([joint_action[1] for joint_action in action_list])

    ego_obs = np.concatenate(ego_obs, axis=-1)
    alt_obs = np.concatenate(alt_obs, axis=-1)
    ego_act = np.concatenate(ego_act, axis=-1)
    alt_act = np.concatenate(alt_act, axis=-1)
    flags = np.concatenate(flags, axis=-1)

    return SimultaneousTransitions(
            ego_obs,
            ego_act,
            alt_obs,
            alt_act,
            flags,
        )


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_json = json.loads(request.data)
        state_dict, player_id_dict, server_layout_name, algo, timestep = data_json["state"], data_json[
            "npc_index"], data_json["layout_name"], data_json["algo"], data_json["timestep"]
        player_id = int(player_id_dict)
        s_all = process_state(state_dict, server_layout_name)
        policy = POLICIES[server_layout_name][algo]
        a = get_prediction(s_all[player_id], policy)
        return jsonify({'action': a})


@app.route('/updatemodel', methods=['POST'])
def updatemodel():
    if request.method == 'POST':
        data_json = json.loads(request.data)
        traj_dict, traj_id, layout_name, algo = data_json["traj"], data_json[
            "traj_id"], data_json["layout_name"], data_json["algo"]
        print(traj_id)

        if ARGS.trajs_savepath:
            # Save trajectory (save this to keep reward information)
            folder = f"{ARGS.trajs_savepath}/{layout_name}/{algo}"
            os.makedirs(folder, exist_ok=True)

            cur_entries = os.listdir(folder)
            idnum = -1
            for entry in cur_entries:
                splitentry = entry.split('.')
                if splitentry[-1] == 'json':
                    newestid = int(splitentry[0])
                    idnum = max(idnum, newestid)

            idnum += 1

            savepath = folder + "/" + str(idnum)

            filename = "%s.json" % (savepath)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(traj_dict, f)

            # Save transitions minimal (only state/action/done, no reward)
            simultaneous_transitions = convert_traj_to_simultaneous_transitions(
                traj_dict, layout_name)
            simultaneous_transitions.write_transition(savepath)

        # Finetune model: todo

        return jsonify({'status': True})


@app.route('/')
def root():
    return app.send_static_file('index.html')


def load_models(path: str, ARGS):
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }

    for layout in os.listdir(path):
        layoutdir = path + "/" + layout
        if not os.path.isdir(layoutdir):
            continue
        translated_name = NAME_TRANSLATION[layout]

        mdp = OvercookedGridworld.from_layout_name(
            layout_name=translated_name, rew_shaping_params=rew_shaping_params
        )

        mlp = MediumLevelPlanner.from_pickle_or_compute(
            mdp, NO_COUNTERS_PARAMS, force_compute=False
        )

        dummy_env = DecentralizedOvercooked(translated_name)

        layout_algos = {}
        for algo in os.listdir(layoutdir):
            splitalgo = algo.split('.')
            if splitalgo[-1] != 'pt':
                break

            algopath = layoutdir + "/" + algo
            actor = R_Actor(ARGS, dummy_env.observation_space, dummy_env.action_space)
            state_dict = torch.load(algopath)
            actor.load_state_dict(state_dict)
            layout_algos[splitalgo[0]] = DecentralizedAgent(actor)

        MLPs[layout] = mlp
        MDPs[layout] = mdp
        POLICIES[layout] = layout_algos


if __name__ == '__main__':
    parser = get_config()
    parser.add_argument('--modelpath', type=str,
                        help = "folder to load AI player")
    parser.add_argument('--trajs_savepath', type=str,
                        help="folder to save trajectories")
    ARGS = parser.parse_args()
    load_models(ARGS.modelpath, ARGS)

    app.run(debug=True, host='0.0.0.0')
