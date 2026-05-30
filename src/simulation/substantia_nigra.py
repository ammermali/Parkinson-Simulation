from dataclasses import dataclass
from src.simulation.utils import clamp, GridHabitatMixin, LocalGrid


@dataclass(frozen=True)
class SNEnvironmentConfig:
    """Parameters controlling shared extracellular scalar dynamics."""

    initial_debris: float
    initial_inflammation: float
    initial_dopamine: float
    debris_decay: float
    inflammation_decay: float
    dopamine_smoothing: float


@dataclass
class SNScalars:
    """Current extracellular state of the Substantia Nigra."""

    extracellular_debris: float
    inflammation_level: float
    dopamine_output: float


@dataclass
class SNEffects:
    """Buffered extracellular effects accumulated during one tick."""
    debris_added: float = 0.0
    debris_removed: float = 0.0
    inflammation_added: float = 0.0
    inflammation_removed: float = 0.0
    dopamine_released: float = 0.0


class SubstantiaNigra(GridHabitatMixin):
    """Shared extracellular habitat and scalar buffer for the simulation."""
    def __init__(self, grid, config: SNEnvironmentConfig):
        self.grid = LocalGrid(repast_grid=grid)
        self.config = config
        self.scalars = SNScalars(
            extracellular_debris=config.initial_debris,
            inflammation_level=config.initial_inflammation,
            dopamine_output=config.initial_dopamine,
        )

        self.effects = SNEffects()

    # TICK CYCLE

    def begin_tick(self):
        """Reset extracellular effect buffers for a new tick."""
        self.effects = SNEffects()

    def commit_effects(self, max_possible_dopamine: float):
        """Apply buffered effects, decay and dopamine smoothing to scalars."""
        cfg = self.config
        old = self.scalars
        eff = self.effects

        new_debris = (old.extracellular_debris + eff.debris_added - eff.debris_removed - cfg.debris_decay * old.extracellular_debris)
        new_inflammation = (old.inflammation_level + eff.inflammation_added - eff.inflammation_removed - cfg.inflammation_decay * old.inflammation_level)

        if max_possible_dopamine > 0:
            dopamine_raw = eff.dopamine_released / max_possible_dopamine
        else:
            dopamine_raw = 0.0

        new_dopamine = (
            (1.0 - cfg.dopamine_smoothing) * old.dopamine_output
            + cfg.dopamine_smoothing * dopamine_raw
        )

        self.scalars.extracellular_debris = clamp(new_debris)
        self.scalars.inflammation_level = clamp(new_inflammation)
        self.scalars.dopamine_output = clamp(new_dopamine)

    # AGENTS EFFECTS

    def add_debris(self, amount: float):
        """Buffer debris released into the extracellular environment."""
        self.effects.debris_added += amount

    def remove_debris(self, amount: float):
        """Buffer extracellular debris removal."""
        self.effects.debris_removed += amount

    def add_inflammation(self, amount: float):
        """Buffer inflammatory signal added to the environment."""
        self.effects.inflammation_added += amount

    def remove_inflammation(self, amount: float):
        """Buffer inflammatory signal removal from the environment."""
        self.effects.inflammation_removed += amount

    def release_dopamine(self, amount: float):
        """Buffer dopamine released by neurons this tick."""
        self.effects.dopamine_released += amount
