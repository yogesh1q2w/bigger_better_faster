import jax
import numpy as np

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

from bbf.eval.atari_eval import AtariEval
from bbf.eval.utils import select_action_eval
from bbf.eval.bbf import BBF


def run(game, key, bbf_wts):

    for i in range(3):
        bbf_wts["params"]["encoder"][f"Stack_{i}"] = bbf_wts["params"]["encoder"].pop(f"ResidualStage_{i}")
    bbf_wts["params"]["projector"] = bbf_wts["params"]["projection"].pop("net")
    bbf_wts["params"]["a_logits_head"] = bbf_wts["params"]["head"]["advantage"].pop("net")
    bbf_wts["params"]["v_logits_head"] = bbf_wts["params"]["head"]["value"].pop("net")
    del (
        bbf_wts["params"]["head"],
        bbf_wts["params"]["predictor"],
        bbf_wts["params"]["transition_model"],
        bbf_wts["params"]["projection"],
    )

    q_key, key = jax.random.split(key)
    env = AtariEval(game, False, 1)
    agent = BBF(q_key, (84, 84, 4), env.n_actions, 51, [64, 128, 128, 2048])
    agent.target_params = bbf_wts
    del env

    env_eval = lambda x: AtariEval(game, False, x)
    episode_returns, episode_lengths = evaluate(key, {"horizon": 27_000}, agent, env_eval(10))  # run for 10 envs
    return episode_returns, episode_lengths


def evaluate(key: jax.Array, p: dict, agent, env):
    key, reset_key = jax.random.split(key)
    env.reset_with_noop(reset_key)
    episode_termination = env.game_over_mask  # needed for considering rewards,length until env.game_over_mask
    episode_returns = np.zeros(env.n_envs)
    episode_lengths = np.zeros(env.n_envs)
    epsilon_fn = lambda _: 0.001

    while not episode_termination.all() and env.n_steps < p["horizon"]:
        key, actions_key = jax.random.split(key)

        actions = select_action_eval(
            agent.best_action, agent.target_params, env.states, actions_key, env.n_actions, epsilon_fn
        )
        rewards = env.step(np.array(actions))

        # episode.termination changes here, so we use episode_termination
        episode_returns += rewards * (1 - episode_termination)
        episode_lengths += 1 - episode_termination
        episode_termination = env.game_over_mask

    return episode_returns.tolist(), episode_lengths.tolist()
