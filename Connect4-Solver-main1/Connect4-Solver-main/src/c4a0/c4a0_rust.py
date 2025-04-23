# src/c4a0_rust.py (giả lập)
import random
from typing import List
from dataclasses import dataclass, field

N_COLS = 7
N_ROWS = 6
BUF_N_CHANNELS = 2

@dataclass
class Sample:
    def to_numpy(self):
        import numpy as np
        return (
            np.random.rand(BUF_N_CHANNELS, N_ROWS, N_COLS).astype('float32'),
            np.random.dirichlet(np.ones(N_COLS)).astype('float32'),
            np.array(0.0, dtype='float32'),
            np.array(0.0, dtype='float32')
        )

    def flip_h(self):
        return self

@dataclass
class GameMetadata:
    id: int
    player0_id: int
    player1_id: int

@dataclass
class PlayGamesResult:
    results: List

    def split_train_test(self, split_ratio, seed):
        return self, self

    def score_policies(self, *args, **kwargs):
        return round(random.uniform(0.4, 0.7), 4)

@dataclass
class MockResult:
    def player0_score(self):
        return random.choice([0, 0.5, 1])

    metadata: GameMetadata = field(default_factory=lambda: GameMetadata(0, 0, 1))
    samples: List[Sample] = field(default_factory=lambda: [Sample(), Sample()])

def play_games(reqs, batch_size, mcts_iters, c_exploration, callback_fn, *args):
    results = [MockResult() for _ in reqs]
    return PlayGamesResult(results=results)

def run_tui(fn, max_mcts_iters, c_exploration, c_ply_penalty):
    print(">>> Mocked Connect 4 TUI started (no real gameplay)")
