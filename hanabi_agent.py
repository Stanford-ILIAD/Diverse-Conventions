from pantheonrl.common.agents import Agent

from MAPPO.utils.util import _t2n


class CentralizedAgent(Agent):

    def __init__(self, cent_player, player_id: int, policy=None):
        self.cent_player = cent_player
        self.player_id = player_id
        if policy is None:
            self.policy = self.cent_player.trainer.policy
        else:
            self.policy = policy

    def get_action(self, obs, record=True):
        obs, share_obs, available_actions = obs
        self.cent_player.trainer.prep_rollout()

        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.policy.get_actions(
                                        share_obs,
                                        obs,
                                        self.cent_player.turn_rnn_states[0, self.player_id],
                                        self.cent_player.turn_rnn_states_critic[0, self.player_id],
                                        self.cent_player.turn_masks[0, self.player_id],
                                        available_actions)
        if record:
            self.cent_player.turn_obs[0, self.player_id] = obs.copy()
            self.cent_player.turn_share_obs[0, self.player_id] = share_obs.copy()
            self.cent_player.turn_available_actions[0, self.player_id] = available_actions.copy()
            self.cent_player.turn_values[0, self.player_id] = _t2n(value)
            self.cent_player.turn_actions[0, self.player_id] = _t2n(action)
            self.cent_player.turn_action_log_probs[0, self.player_id] = _t2n(action_log_prob)
            self.cent_player.turn_rnn_states[0, self.player_id] = _t2n(rnn_state)
            self.cent_player.turn_rnn_states_critic[0, self.player_id] = _t2n(rnn_state_critic)
            self.cent_player.turn_rewards[0, self.player_id] = 0
            self.cent_player.turn_active_masks[0, self.player_id] = 1

        return _t2n(action)

    def update(self, reward, done):
        self.cent_player.turn_rewards[0, self.player_id] += reward

        if done:
            self.cent_player.turn_masks[0, self.player_id] = 0
            self.cent_player.turn_rnn_states[0, self.player_id] = 0
            self.cent_player.turn_rnn_states_critic[0, self.player_id] = 0
        else:
            self.cent_player.turn_masks[0, self.player_id] = 1
