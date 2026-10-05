"""Experiment-local deterministic hard-reset fix for hf-libero0.1.4."""

import hashlib

from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv as OfficialLiberoEnv


class LiberoEnv(OfficialLiberoEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        domain = self._env.env
        original = domain._load_model

        def reload_model():
            # _load_model appends open/close samplers but the upstream hard reset
            # never clears the old list. Duplicates consume extra RNG draws and
            # change fixture body poses, which are absent from the saved qpos state.
            domain.object_property_initializers.clear()
            return original()

        domain._load_model = reload_model

    def initial_fingerprints(self, observation):
        sim = self._env.env.sim
        return {
            "fixture_model_sha256": hashlib.sha256(
                sim.model.body_pos.tobytes() + sim.model.body_quat.tobytes()
            ).hexdigest(),
            "settled_sim_state_sha256": hashlib.sha256(
                sim.get_state().flatten().tobytes()
            ).hexdigest(),
            "initial_rgb_sha256": {
                k: hashlib.sha256(v.tobytes()).hexdigest()
                for k, v in observation.items()
                if k.startswith("video.")
            },
            "property_sampler_count": len(self._env.env.object_property_initializers),
        }
