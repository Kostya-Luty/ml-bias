from stable_baselines3.common.callbacks import BaseCallback


class CurriculumStepCallback(BaseCallback):
    """Pushes SB3's global timestep into the bias wrapper so eps(t) can advance."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # self.num_timesteps is SB3's authoritative global step count.
        # self.training_env is the (possibly vectorized) env SB3 is training on.
        self.training_env.env_method("set_global_step", self.num_timesteps)
        eps = self.training_env.env_method("get_current_epsilon")[0]
        self.logger.record("curriculum/epsilon", eps)
        return True   # returning False would abort training