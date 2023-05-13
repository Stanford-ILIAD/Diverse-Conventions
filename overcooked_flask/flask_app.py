import os
import io
import json
import copy
import argparse
import torch
import numpy as np
import gym

from flask import Flask, jsonify, request
from pyngrok import ngrok
# from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
# from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

# from overcooked_utils import NAME_TRANSLATION

# from overcooked_env import DecentralizedOvercooked

# from partner_agents import DecentralizedAgent
# from MAPPO.r_actor_critic import R_Actor
# from config import get_config

app = Flask(__name__)

MLPs = {}
MDPs = {}
POLICIES = {}


LAYOUTS = ["simple", "random1"]
# LAYOUTS = ["five_by_five", "random1", "random3", "scenario1_s", "simple"]
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
                    out_string = f"All done for \"{ALGO_NAMES[i]}\" in {layout}! <b>Please fill out <a href=\"https://forms.gle/DqrE75gJPyjR1y3D6\" target=\"_blank\" rel=\"noopener noreferrer\">this Google form</a></b><br />"
                else:
                    # print(curalgo, curlayout, i, layout)
                    out_string = f"All done for \"{ALGO_NAMES[i]}\" in {layout}!<br />"
            stringtoret += out_string

    if done:
        stringtoret += f"Congratulations! You finished all experiments. Please fill out <a href=\"https://forms.gle/EqCMvtrJbs5RCTFy7\" target=\"_blank\" rel=\"noopener noreferrer\">this last form</a></b> (in addition to the one above) to get the prolific completion code."
    return {'status': done, 'code': CODE, 'record': stringtoret, 'nextalgo': next_algo, 'nextlayout': next_layout}


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


@app.route('/updatemodel', methods=['POST'])
def updatemodel():
    ip_addr = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    if request.method == 'POST':
        data_json = json.loads(request.data)
        print(data_json.keys())
        prolific_id = data_json["prolific_id"]
        traj_dict, traj_id, layout_name, algo, algo_options = data_json["traj"], data_json[
            "traj_id"], data_json["layout_name"], data_json["algo"], data_json["algo_options"]
        print(traj_id, prolific_id)

        print(layout_name)

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
            print("adding entry", idnum)

            idnum += 1

            savepath = folder + "/" + str(idnum)

            filename = "%s.json" % (savepath)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(traj_dict, f)

            return jsonify(get_user_status(prolific_id, ip_addr, algo_options, algo, layout_name))

        return jsonify({'status': True})


@app.route('/')
def root():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajs_savepath', type=str,
                        help="folder to save trajectories")
    ARGS = parser.parse_args()

    public_url = ngrok.connect(5000).public_url
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, 5000))
    app.run(debug=False, host="0.0.0.0")
