import jax
import numpy as np

from bbf.eval.atari_eval import AtariEnv
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
    env = AtariEnv(game, False)
    agent = BBF(q_key, (84, 84, 4), env.n_actions, 51, [64, 128, 128, 2048])
    agent.target_params = bbf_wts
    del env

    episode_returns, episode_lengths = evaluate(key, agent, game)  # run for 10 envs
    return episode_returns, episode_lengths


def evaluate(key: jax.Array, agent, game_name):
    epsilon_fn = lambda _: 0.001

    episode_returns = []
    episode_lengths = []

    for _ in range(10):
        episode_returns.append(0)
        episode_lengths.append(0)

        env = AtariEnv(game_name, False)
        reset_key, key = jax.random.split(key)
        env.reset_with_noop(reset_key)
        game_over = False

        while not game_over and env.n_steps < 27_000:
            key, action_key = jax.random.split(key)

            action = select_action_eval(
                agent.best_action, agent.target_params, np.array([env.state]), action_key, env.n_actions, epsilon_fn
            )
            reward, _, game_over = env.step(action[0])

            # episode.termination changes here, so we use episode_termination
            episode_returns[-1] += reward
            episode_lengths[-1] += 1 - game_over
        del env

    return episode_returns, episode_lengths
