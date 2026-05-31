import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


def install_runtime_stubs(monkeypatch):
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda stream: {}
    monkeypatch.setitem(sys.modules, "yaml", yaml_module)

    class TestComm:
        def Get_rank(self):
            return 0

    mpi_module = types.ModuleType("mpi4py")
    mpi_module.MPI = types.SimpleNamespace(COMM_WORLD=TestComm(), Intracomm=TestComm)
    monkeypatch.setitem(sys.modules, "mpi4py", mpi_module)

    repast_module = types.ModuleType("repast4py")
    core_module = types.ModuleType("repast4py.core")

    class TestCoreAgent:
        def __init__(self, local_id, type_id, rank):
            self.uid = (local_id, type_id, rank)
            self.local_id = local_id
            self.type_id = type_id
            self.rank = rank

    core_module.Agent = TestCoreAgent

    space_module = types.ModuleType("repast4py.space")

    @dataclass(frozen=True)
    class TestDiscretePoint:
        x: int
        y: int
        z: int = 0

    class TestBoundingBox:
        def __init__(self, xmin, xextent, ymin, yextent, zmin=0, zextent=0):
            self.xmin = xmin
            self.xextent = xextent
            self.ymin = ymin
            self.yextent = yextent
            self.zmin = zmin
            self.zextent = zextent

    class TestSharedGrid:
        def __init__(self, name, bounds, borders, occupancy, buffer_size, comm):
            self.name = name
            self.bounds = bounds
            self.borders = borders
            self.occupancy = occupancy
            self.buffer_size = buffer_size
            self.comm = comm
            self.locations = {}
            self.cells = {}
            self.next_point_index = 0

        def get_local_bounds(self):
            return self.bounds

        def get_random_local_pt(self, rng):
            x = self.bounds.xmin + self.next_point_index % self.bounds.xextent
            y = self.bounds.ymin + self.next_point_index // self.bounds.xextent
            self.next_point_index += 1
            return TestDiscretePoint(x, y)

        def add(self, agent):
            return True

        def move(self, agent, point):
            old_point = self.locations.get(agent)
            if old_point is not None and agent in self.cells.get((old_point.x, old_point.y), []):
                self.cells[(old_point.x, old_point.y)].remove(agent)
            self.locations[agent] = point
            self.cells.setdefault((point.x, point.y), []).append(agent)
            return point

        def remove(self, agent):
            old_point = self.locations.pop(agent, None)
            if old_point is not None and agent in self.cells.get((old_point.x, old_point.y), []):
                self.cells[(old_point.x, old_point.y)].remove(agent)

        def get_location(self, agent):
            return self.locations.get(agent)

        def get_agents(self, point):
            return list(self.cells.get((point.x, point.y), []))

        def get_num_agents(self, point, agent_type):
            return sum(
                1
                for agent in self.get_agents(point)
                if getattr(agent, "ptype", None) == agent_type
            )

    space_module.DiscretePoint = TestDiscretePoint
    space_module.BoundingBox = TestBoundingBox
    space_module.SharedGrid = TestSharedGrid
    space_module.BorderType = types.SimpleNamespace(Sticky="Sticky")
    space_module.OccupancyType = types.SimpleNamespace(Multiple="Multiple")

    context_module = types.ModuleType("repast4py.context")

    class TestSharedContext:
        def __init__(self, comm):
            self.comm = comm
            self.projections = []
            self._agents = []

        def add_projection(self, projection):
            self.projections.append(projection)

        def add(self, agent):
            if agent not in self._agents:
                self._agents.append(agent)

        def agents(self):
            return list(self._agents)

    context_module.SharedContext = TestSharedContext

    schedule_module = types.ModuleType("repast4py.schedule")

    class TestScheduleRunner:
        def __init__(self):
            self.repeating_events = []
            self.stop_at = None
            self.executed = False

        def schedule_repeating_event(self, start, interval, callback):
            self.repeating_events.append((start, interval, callback))

        def schedule_stop(self, stop_at):
            self.stop_at = stop_at

        def execute(self):
            self.executed = True

    schedule_module.init_schedule_runner = lambda comm: TestScheduleRunner()

    random_module = types.ModuleType("repast4py.random")
    random_module.default_rng = object()
    random_module.last_seed = None

    def init(seed):
        random_module.last_seed = seed

    random_module.init = init

    repast_module.core = core_module
    repast_module.space = space_module
    repast_module.context = context_module
    repast_module.schedule = schedule_module
    repast_module.random = random_module

    monkeypatch.setitem(sys.modules, "repast4py", repast_module)
    monkeypatch.setitem(sys.modules, "repast4py.core", core_module)
    monkeypatch.setitem(sys.modules, "repast4py.space", space_module)
    monkeypatch.setitem(sys.modules, "repast4py.context", context_module)
    monkeypatch.setitem(sys.modules, "repast4py.schedule", schedule_module)
    monkeypatch.setitem(sys.modules, "repast4py.random", random_module)


@pytest.fixture()
def engine_module(monkeypatch):
    install_runtime_stubs(monkeypatch)
    sys.modules.pop("src.simulation.engine", None)
    engine = importlib.import_module("src.simulation.engine")
    install_engine_config_stubs(engine, monkeypatch)
    return engine


def install_engine_config_stubs(engine, monkeypatch):
    neuron_module = importlib.import_module("src.simulation.agents.neuron")
    alpha_module = importlib.import_module("src.simulation.agents.alphasynuclein")
    microglia_module = importlib.import_module("src.simulation.agents.microglia")
    astrocyte_module = importlib.import_module("src.simulation.agents.astrocyte")
    mitochondrion_module = importlib.import_module("src.simulation.agents.mitochondrion")
    lysosome_module = importlib.import_module("src.simulation.agents.lysosome")
    sn_module = importlib.import_module("src.simulation.substantia_nigra")

    neuron_params = {
        "intracellular": {
            "grid": {"width": 3, "height": 4},
            "population": {"alpha": 2, "mitochondria": 2, "lysosomes": 1},
            "scalars": {"energy_demand_baseline": 0.55},
        }
    }
    system_params = {
        "stop.at": 9,
        "random.seed": 13,
        "world": {"width": 5, "height": 5, "buffer_size": 1},
        "external.population": {"neurons": 1, "microglia": 0, "astrocytes": 0, "alpha": 0},
    }

    class TestParams:
        def __init__(self, name):
            self.name = str(name)

        def as_dict(self):
            if "neuron" in self.name:
                return dict(neuron_params)
            if "system" in self.name:
                return dict(system_params)
            return {}

    monkeypatch.setattr(engine, "Params", TestParams)
    monkeypatch.setattr(
        engine.ConfigFactory,
        "build_substantia_nigra_config",
        lambda *args, **kwargs: sn_module.SNEnvironmentConfig(0.0, 0.0, 0.0, 0.01, 0.01, 0.5),
    )
    monkeypatch.setattr(
        engine.ConfigFactory,
        "build_neuron_config",
        lambda *args, **kwargs: neuron_module.NeuronConfig(
            per_radius=1,
            nearby_alpha_high_threshold=0.7,
            inflammation_high_threshold=0.7,
            debris_high_threshold=0.7,
            alpha_load_release_threshold=0.8,
            damage_accumulation_rate=0.1,
            damage_recovery_rate=0.1,
            low_stress_threshold=0.1,
            inflammation_damage_weight=0.3,
            debris_damage_weight=0.3,
            alpha_damage_weight=0.4,
            compromised_threshold=0.4,
            apoptotic_threshold=0.7,
            ruptured_threshold=0.9,
            dopamine_release_rate=0.25,
            stress_inflammation_release_rate=0.1,
            debris_release_rate=0.1,
            alpha_absorption_rate=0.5,
            alpha_release_amount=0.1,
        ),
    )
    monkeypatch.setattr(
        engine.ConfigFactory,
        "build_neuron_internal_config",
        lambda *args, **kwargs: neuron_module.NeuronInternalConfig(
            width=3,
            height=4,
            energy_demand_baseline=0.55,
        ),
    )
    monkeypatch.setattr(
        engine.ConfigFactory,
        "build_microglia_config",
        lambda *args, **kwargs: microglia_module.MicrogliaConfig(1, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2, 0.1, 0.1, 0.5),
    )
    monkeypatch.setattr(
        engine.ConfigFactory,
        "build_astrocyte_config",
        lambda *args, **kwargs: astrocyte_module.AstrocyteConfig(0.7, 0.2, 0.7, 0.2, 0.05, 0.05),
    )
    monkeypatch.setattr(
        engine.ConfigFactory,
        "build_alpha_synuclein_config",
        lambda *args, **kwargs: alpha_module.AlphaSynucleinConfig(1, 1, 0.5, 0.6),
    )
    monkeypatch.setattr(
        engine.ConfigFactory,
        "build_mitochondrion_config",
        lambda *args, **kwargs: mitochondrion_module.MitochondrionConfig(
            1, 0.7, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2, 0.8, 0.02, 0.04, 0.02, 0.02, 0.02, 0.03, 0.01, 1.5
        ),
    )
    monkeypatch.setattr(
        engine.ConfigFactory,
        "build_lysosome_config",
        lambda *args, **kwargs: lysosome_module.LysosomeConfig(1, 1, 0.8, 1, 1, 0.8, 1, 1, 0.4, 0.05, 0.02, 0.01),
    )


class TestEngineParams:
    def test_params_loader_resolves_default_params_directory(self, engine_module):
        paramsloader_module = importlib.import_module("src.simulation.utils.paramsloader")

        params = paramsloader_module.Params("system")

        assert params.path.parent.name == "params"
        assert params.path.name == "system.yaml"

    def test_params_loader_reads_yaml_from_default_params_directory(self, engine_module):
        paramsloader_module = importlib.import_module("src.simulation.utils.paramsloader")
        captured = {}

        def safe_load(stream):
            path = Path(stream.name)
            captured["path"] = path
            return {
                "stop.at": 11,
                "world": {"width": 12, "height": 13, "buffer_size": 3},
            }

        paramsloader_module.yaml.safe_load = safe_load

        params = paramsloader_module.Params("system")

        assert captured["path"].parent.name == "params"
        assert captured["path"].name == "system.yaml"
        assert params.as_dict()["stop.at"] == 11
        assert params.as_dict()["world"]["width"] == 12

    def test_param_reads_direct_nested_and_dotted_section_values(self, engine_module):
        params = {
            "stop.at": 7,
            "world": {"width": 10},
            "external.population": {"neurons": 3},
        }

        assert engine_module._param(params, "stop.at") == 7
        assert engine_module._param(params, "world.width") == 10
        assert engine_module._param(params, "external.population.neurons") == 3
        assert engine_module._param(params, "missing.value", 99) == 99

    def test_project_root_bootstrap_is_idempotent(self, engine_module, monkeypatch):
        project_root = str(Path(engine_module.__file__).resolve().parents[2])
        monkeypatch.setattr(sys, "path", [path for path in sys.path if path != project_root])

        engine_module._ensure_project_root_on_path()
        engine_module._ensure_project_root_on_path()

        assert sys.path[0] == project_root
        assert sys.path.count(project_root) == 1


class TestParkinsonModel:
    def test_initializes_grid_context_schedule_and_agent_populations(self, engine_module):
        params = {
            "stop.at": 5,
            "random.seed": 21,
            "world": {"width": 6, "height": 7, "buffer_size": 2},
            "external.population": {"neurons": 2, "microglia": 1, "astrocytes": 1, "alpha": 1},
        }

        model = engine_module.ParkinsonModel(engine_module.MPI.COMM_WORLD, params)
        neurons = [agent for agent in model.context.agents() if isinstance(agent, engine_module.Neuron)]

        assert model.grid.bounds.xextent == 6
        assert model.grid.bounds.yextent == 7
        assert model.grid.buffer_size == 2
        assert model.runner.stop_at == 5
        assert len(model.context.agents()) == 5
        assert len(neurons) == 2
        assert len(neurons[0].grid.agent_registry) == 5
        assert neurons[0].internal_cfg.width == 3
        assert neurons[0].internal_cfg.height == 4
        assert neurons[0].internal_scalars.energy_demand == 0.55

    def test_model_passes_loaded_neuron_params_to_neuron_factories(self, engine_module, monkeypatch):
        captured = {}
        original_neuron_config_builder = engine_module.ConfigFactory.build_neuron_config
        original_internal_config_builder = engine_module.ConfigFactory.build_neuron_internal_config

        def build_neuron_config(params, rng=None):
            captured["neuron_config_params"] = params.name
            captured["neuron_config_rng"] = rng
            return original_neuron_config_builder(params, rng=rng)

        def build_neuron_internal_config(params):
            captured["internal_config_params"] = params.name
            return original_internal_config_builder(params)

        monkeypatch.setattr(engine_module.ConfigFactory, "build_neuron_config", build_neuron_config)
        monkeypatch.setattr(engine_module.ConfigFactory, "build_neuron_internal_config", build_neuron_internal_config)

        params = {
            "world": {"width": 4, "height": 4, "buffer_size": 1},
            "external.population": {"neurons": 1, "microglia": 0, "astrocytes": 0, "alpha": 0},
        }

        model = engine_module.ParkinsonModel(engine_module.MPI.COMM_WORLD, params)

        assert captured["neuron_config_params"] == "neuron"
        assert captured["internal_config_params"] == "neuron"
        assert captured["neuron_config_rng"] is model.config_rng
        assert model.neuron_param_values["intracellular"]["population"]["alpha"] == 2

    def test_intracellular_population_prefers_neuron_params_over_system_params(self, engine_module):
        params = {
            "world": {"width": 4, "height": 4, "buffer_size": 1},
            "external.population": {"neurons": 1, "microglia": 0, "astrocytes": 0, "alpha": 0},
            "intracellular.population": {"alpha": 99, "mitochondria": 99, "lysosomes": 99},
        }

        model = engine_module.ParkinsonModel(engine_module.MPI.COMM_WORLD, params)
        neuron = next(agent for agent in model.context.agents() if isinstance(agent, engine_module.Neuron))

        assert len(neuron.grid.agent_registry) == 5

    def test_step_uses_agent_snapshot_and_commits_environment(self, engine_module):
        class TestEnvironment:
            def __init__(self):
                self.begin_calls = 0
                self.commit_values = []

            def begin_tick(self):
                self.begin_calls += 1

            def commit_effects(self, max_possible_dopamine):
                self.commit_values.append(max_possible_dopamine)

        class TestContext:
            def __init__(self):
                self._agents = []

            def agents(self):
                return list(self._agents)

        class StepAgent:
            def __init__(self):
                self.calls = 0

            def step(self, model):
                self.calls += 1
                model.context._agents.append(LateAgent())

        class LateAgent:
            def __init__(self):
                self.calls = 0

            def step(self, model):
                self.calls += 1

        model = object.__new__(engine_module.ParkinsonModel)
        model.rank = 1
        model.environment = TestEnvironment()
        model.context = TestContext()
        step_agent = StepAgent()
        model.context._agents.append(step_agent)
        model._max_possible_dopamine = lambda: 1.25

        model.step()

        assert model.environment.begin_calls == 1
        assert model.environment.commit_values == [1.25]
        assert step_agent.calls == 1
        assert model.context._agents[1].calls == 0

    def test_loggers_are_configured_from_system_params(self, engine_module, tmp_path):
        params = {
            "stop.at": 2,
            "random.seed": 21,
            "world": {"width": 4, "height": 4, "buffer_size": 1},
            "external.population": {"neurons": 0, "microglia": 0, "astrocytes": 0, "alpha": 0},
            "logging": {
                "enabled": True,
                "output_dir": str(tmp_path),
                "scalar_stdout": False
            }}
        model = engine_module.ParkinsonModel(engine_module.MPI.COMM_WORLD, params)
        assert model.causal_logger.enabled is True
        assert model.initialization_logger.enabled is True
        assert model.causal_logger.output_dir == tmp_path
        assert model.initialization_logger.output_dir == tmp_path

    def test_step_records_g0_field_nodes_when_causal_logging_is_enabled(self, engine_module, tmp_path):
        params = {
            "stop.at": 2,
            "random.seed": 21,
            "world": {"width": 4, "height": 4, "buffer_size": 1},
            "external.population": {"neurons": 0, "microglia": 0, "astrocytes": 0, "alpha": 0},
            "logging": {
                "enabled": True,
                "output_dir": str(tmp_path),
                "scalar_stdout": False}}
        model = engine_module.ParkinsonModel(engine_module.MPI.COMM_WORLD, params)
        model.step()
        rows = (tmp_path / "g0_nodes.jsonl").read_text(encoding="utf-8").splitlines()
        assert any('"field": "extracellular_debris"' in row for row in rows)
        assert (tmp_path / "run_metadata.json").exists()

    def test_max_possible_dopamine_excludes_apoptotic_and_ruptured_neurons(self, engine_module):
        neuron_module = importlib.import_module("src.simulation.agents.neuron")
        def make_neuron(local_id, state, dopamine_release_rate):
            config = neuron_module.NeuronConfig(
                per_radius=1,
                nearby_alpha_high_threshold=0.7,
                inflammation_high_threshold=0.7,
                debris_high_threshold=0.7,
                alpha_load_release_threshold=0.8,
                damage_accumulation_rate=0.1,
                damage_recovery_rate=0.1,
                low_stress_threshold=0.1,
                inflammation_damage_weight=0.3,
                debris_damage_weight=0.3,
                alpha_damage_weight=0.4,
                compromised_threshold=0.4,
                apoptotic_threshold=0.7,
                ruptured_threshold=0.9,
                dopamine_release_rate=dopamine_release_rate,
                stress_inflammation_release_rate=0.1,
                debris_release_rate=0.1,
                alpha_absorption_rate=0.5,
                alpha_release_amount=0.1,
            )
            neuron = engine_module.Neuron(local_id, 0, engine_module.AgentType.NEURON, config, alpha_type_id=3)
            neuron.state = state
            return neuron

        class TestContext:
            def __init__(self, agents):
                self._agents = agents
            def agents(self):
                return list(self._agents)
        model = object.__new__(engine_module.ParkinsonModel)
        model.context = TestContext([
            make_neuron(1, engine_module.NeuronState.HEALTHY, 0.2),
            make_neuron(2, engine_module.NeuronState.COMPROMISED, 0.3),
            make_neuron(3, engine_module.NeuronState.APOPTOTIC, 0.4),
            make_neuron(4, engine_module.NeuronState.RUPTURED, 0.5),
        ])
        assert model._max_possible_dopamine() == pytest.approx(0.5)

    def test_run_loads_system_params_and_starts_model(self, engine_module, monkeypatch):
        captured = {}
        class TestModel:
            def __init__(self, comm, params):
                captured["comm"] = comm
                captured["params"] = params
            def start(self):
                captured["started"] = True
        monkeypatch.setattr(engine_module, "ParkinsonModel", TestModel)
        engine_module.run()
        assert captured["comm"] is engine_module.MPI.COMM_WORLD
        assert captured["params"]["stop.at"] == 9
        assert captured["started"] is True