from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from bbf.eval.bbf_arch import BBFNet


class BBF:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bins: int,
        features: list,
    ):
        self.observation_dim = observation_dim
        self.key, init_key = jax.random.split(key)

        self.network = BBFNet(features, n_actions, n_bins)
        self.params = self.network.init(
            init_key, jnp.zeros(observation_dim, dtype=jnp.float32), jnp.zeros(5, dtype=int)
        )
        self.bins = np.linspace(start=-10, stop=10, num=n_bins)
        self.target_params = self.params.copy()

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        normalized_state = state.astype(jnp.float32) / 255.0
        return jnp.argmax(jax.nn.softmax(self.network.apply(params, normalized_state)) @ self.bins)

    def get_model(self):
        return {"params": self.target_params}
