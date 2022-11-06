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

LAYOUTS = ["cramped_room", "coordination_ring"]
ALGOS = ["BLANK", "SP", "MP", "ADAP", "XP"]
ALGO_NAMES = ["No AI", "AI S", "AI M", "AI D", "AI X"]
op_to_ind = {
    "No AI": 0,
    "AI S": 1,
    "AI M": 2,
    "AI D": 3,
    "AI X": 4
}
NUM_EXP_PER = 2
CODE = 0


def get_user_status(prolific_id, ip_addr, options, curalgo=None, curlayout=None):
    stringtoret = ""
    done = True
    bolded = False
    next_algo = options[0]
    next_layout = LAYOUTS[0]
    for layout in LAYOUTS:
        for algname in options:
            i = op_to_ind[algname]
            algo = ALGOS[i]
            folder = f"{ARGS.trajs_savepath}/{layout}/{algo}/{prolific_id}/{ip_addr}"
            os.makedirs(folder, exist_ok=True)

            cur_entries = os.listdir(folder)
            num_games = 0
            for entry in cur_entries:
                splitentry = entry.split('.')
                if splitentry[-1] == 'json':
                    num_games += 1

            out_string = ""
            if num_games < NUM_EXP_PER:
                done = False
                out_string = f"Please play {NUM_EXP_PER - num_games} more games with \"{ALGO_NAMES[i]}\" in {layout}.<br />"
                if not bolded:
                    out_string = "<b>" + out_string + "</b>"
                    next_algo = algo
                    next_layout = layout
                    bolded = True
            else:
                if curalgo is not None and curlayout is not None and i != 0 and algo == curalgo and layout == curlayout:
                    out_string = f"All done for \"{ALGO_NAMES[i]}\" in {layout}! <b>Please fill out <a href=\"https://forms.gle/9jXt8zacHVsjjwsu7\" target=\"_blank\" rel=\"noopener noreferrer\">this Google form</a></b><br />"
                else:
                    # print(curalgo, curlayout, i, layout)
                    out_string = f"All done for \"{ALGO_NAMES[i]}\" in {layout}!<br />"
            stringtoret += out_string
    return {'status': done, 'code': CODE, 'record': stringtoret, 'nextalgo': next_algo, 'nextlayout': next_layout}


def get_prediction(s, policy):
    s = torch.tensor(s).unsqueeze(0).float()
    actions = policy.predict(observation=s, record=False, deterministic=False)
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


@app.route('/initrecord', methods=['POST'])
def initrecord():
    ip_addr = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    if request.method == 'POST':
        data_json = json.loads(request.data)
        prolific_id = data_json["prolific_id"]
        algo_options = data_json["algo_options"]
        # print(prolific_id)

        if ARGS.trajs_savepath:
            return jsonify(get_user_status(prolific_id, ip_addr, algo_options))

        return jsonify({'status': True})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_json = json.loads(request.data)
        state_dict, player_id_dict, server_layout_name, algo, timestep = data_json["state"], data_json[
            "npc_index"], data_json["layout_name"], data_json["algo"], data_json["timestep"]
        if algo == "BLANK":
            return jsonify({'action': 4})
        player_id = int(player_id_dict)
        s_all = process_state(state_dict, server_layout_name)
        policy = POLICIES[server_layout_name][algo]
        a = get_prediction(s_all[player_id], policy)
        return jsonify({'action': a})


@app.route('/updatemodel', methods=['POST'])
def updatemodel():
    ip_addr = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    if request.method == 'POST':
        data_json = json.loads(request.data)
        prolific_id = data_json["prolific_id"]
        traj_dict, traj_id, layout_name, algo, algo_options = data_json["traj"], data_json[
            "traj_id"], data_json["layout_name"], data_json["algo"], data_json["algo_options"]
        print(traj_id, prolific_id)

        if ARGS.trajs_savepath:
            # Save trajectory (save this to keep reward information)
            folder = f"{ARGS.trajs_savepath}/{layout_name}/{algo}/{prolific_id}/{ip_addr}"
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
            return jsonify(get_user_status(prolific_id, ip_addr, algo_options, algo, layout_name))

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
                        help="folder to load AI player")
    parser.add_argument('--trajs_savepath', type=str,
                        help="folder to save trajectories")
    ARGS = parser.parse_args()
    load_models(ARGS.modelpath, ARGS)

    app.run(debug=True, host='0.0.0.0')
