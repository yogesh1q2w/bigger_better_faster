import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_fn"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_fn, n_sampling_steps):
    uniform_key, action_key = jax.random.split(key)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_fn(n_sampling_steps),  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        best_action_fn(params, state),  # otherwise, take a greedy action
    )


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_fn"))
def select_action_eval(best_action_fn, params, states, key, n_actions, epsilon_fn):
    selection_keys = jax.random.split(key, states.shape[0])
    return jax.vmap(select_action, in_axes=(None, None, 0, 0, None, None, None))(
        best_action_fn, params, states, selection_keys, n_actions, epsilon_fn, 0
    )
