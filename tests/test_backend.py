from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend import main
from backend import ReplayExportSpec
from backend import export_replay_library
from backend import run_backend
from domain import DEFAULT_PROBLEM
from domain import SnakeProblem


SMALL_PROBLEM = SnakeProblem(
    grid_size=8,
    start=(0, 0),
    apples=((1, 1), (3, 5), (6, 2)),
)


class BackendTests(unittest.TestCase):
    def assert_replay_payload(self, payload: dict[str, object], apple_count: int, grid_size: int) -> None:
        self.assertEqual(
            set(payload),
            {"title", "gridSize", "frameDurationMs", "wallMode", "apples", "frames"},
        )
        self.assertEqual(payload["gridSize"], grid_size)
        self.assertEqual(payload["wallMode"], "wrap")
        self.assertEqual(len(payload["apples"]), apple_count)
        self.assertTrue(payload["frames"])
        self.assertTrue(payload["frames"][0]["snake"])

    def test_astar_payload_matches_frontend_contract(self) -> None:
        payload = run_backend(SMALL_PROBLEM, approach="astar")

        self.assert_replay_payload(payload, len(SMALL_PROBLEM.apples), SMALL_PROBLEM.grid_size)

    def test_route_approaches_export_through_same_interface(self) -> None:
        for approach in ("genetic", "simulated_annealing", "metropolis"):
            with self.subTest(approach=approach):
                payload = run_backend(SMALL_PROBLEM, approach=approach)
                self.assert_replay_payload(payload, len(SMALL_PROBLEM.apples), SMALL_PROBLEM.grid_size)

    def test_learned_approach_exports_through_same_interface(self) -> None:
        kohonen_payload = run_backend(SMALL_PROBLEM, approach="kohonen")

        self.assert_replay_payload(kohonen_payload, len(SMALL_PROBLEM.apples), SMALL_PROBLEM.grid_size)

    def test_cli_writes_replay_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "trajectory.json"
            with redirect_stdout(StringIO()):
                main(["--approach", "astar", "--output", str(output)])

            payload = json.loads(output.read_text())
            self.assert_replay_payload(payload, len(DEFAULT_PROBLEM.apples), DEFAULT_PROBLEM.grid_size)

    def test_export_replay_library_writes_manifest_and_replays(self) -> None:
        specs = (
            ReplayExportSpec(
                approach="astar",
                label="A*",
                file_name="trajectory.json",
                group="Search",
            ),
            ReplayExportSpec(
                approach="dqn",
                label="DQN (Adam)",
                file_name="dqn_adam.json",
                group="Learned",
                dqn_optimizer_name="adam",
                dqn_sample_count=32,
                dqn_train_steps=4,
            ),
        )
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            manifest = export_replay_library(output_dir, problem=SMALL_PROBLEM, specs=specs)

            self.assertEqual(
                manifest,
                [
                    {"label": "A*", "file": "trajectory.json", "group": "Search"},
                    {"label": "DQN (Adam)", "file": "dqn_adam.json", "group": "Learned"},
                ],
            )
            self.assertEqual(json.loads((output_dir / "trajectories.json").read_text()), manifest)

            for entry in manifest:
                payload = json.loads((output_dir / entry["file"]).read_text())
                self.assert_replay_payload(payload, len(SMALL_PROBLEM.apples), SMALL_PROBLEM.grid_size)


if __name__ == "__main__":
    unittest.main()
