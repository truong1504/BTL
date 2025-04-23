#!/usr/bin/env python
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import sys
from typing import List, Optional
import warnings
from loguru import logger
import optuna
import torch
import typer
import math
import random

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from c4a0.pos import Pos
from c4a0.nn import ModelConfig
from c4a0.sweep import perform_hparam_sweep
from c4a0.tournament import ModelID, RandomPlayer, UniformPlayer
from c4a0.training import SolverConfig, TrainingGen, parse_lr_schedule, training_loop
from c4a0.utils import get_torch_device

app = typer.Typer()

@app.command()
def train(
    base_dir: str = "training",
    device: str = str(get_torch_device()),
    n_self_play_games: int = 3000,
    n_mcts_iterations: int = 2000,
    c_exploration: float = 5.0,
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 3,
    conv_filter_size: int = 64,
    n_policy_layers: int = 3,
    n_value_layers: int = 3,
    lr_schedule: List[float] = [0, 1e-3, 20, 5e-4, 40, 1e-4],
    l2_reg: float = 1e-4,
    label_smoothing: float = 0.1,
    max_gens: Optional[int] = None,
    solver_path: Optional[str] = None,
    book_path: Optional[str] = None,
    solutions_path: str = "./solutions.db",
):
    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=parse_lr_schedule(lr_schedule),
        l2_reg=l2_reg,
        label_smoothing=label_smoothing,
    )

    solver_config = None
    if solver_path and book_path:
        logger.info("Using solver")
        solver_config = SolverConfig(
            solver_path=solver_path,
            book_path=book_path,
            solutions_path=solutions_path,
        )

    training_loop(
        base_dir=base_dir,
        device=torch.device(device),
        n_self_play_games=n_self_play_games,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        model_config=model_config,
        max_gens=max_gens,
        solver_config=solver_config,
    )

class Node:
    EPS = 1e-8

    def __init__(self, pos, parent=None, initial_policy_value=0):
        self.pos = pos
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.q_sum_penalty = 0.0
        self.q_sum_no_penalty = 0.0
        self.initial_policy_value = initial_policy_value

    def q_with_penalty(self):
        return self.q_sum_penalty / (self.visit_count + self.EPS)

    def q_no_penalty(self):
        return self.q_sum_no_penalty / (self.visit_count + self.EPS)

    def exploration_value(self):
        parent_visits = self.parent.visit_count if self.parent else self.visit_count
        exploration = math.sqrt(math.log(parent_visits + 1) / (self.visit_count + self.EPS))
        return exploration * (self.initial_policy_value + self.EPS)

    def uct_value(self, c_exploration):
        return -self.q_with_penalty() + c_exploration * self.exploration_value()

    def is_terminal(self):
        return self.pos.is_terminal_state() is not None

    def policy(self):
        if self.is_terminal():
            return [0.0] * 7
        if not self.children:
            return [1.0 / 7] * 7
        child_visits = [self.children.get(a, Node(self.pos)).visit_count for a in range(7)]
        total_visits = sum(child_visits)
        if total_visits == 0:
            return [1.0 / 7] * 7
        return [v / total_visits for v in child_visits]

class MCTS:
    def __init__(self, root_pos, model, c_exploration=5.0, c_ply_penalty=0.01):
        self.root = Node(root_pos)
        self.leaf = self.root
        self.model = model
        self.c_exploration = c_exploration
        self.c_ply_penalty = c_ply_penalty

    def select_leaf(self):
        current = self.root
        while current.children and not current.is_terminal():
            current = max(current.children.values(), key=lambda child: child.uct_value(self.c_exploration))
        self.leaf = current

    def expand_leaf(self, policy_probs):
        if self.leaf.is_terminal():
            return
        legal_actions = [a for a, is_legal in enumerate(self.leaf.pos.legal_moves()) if is_legal]
        for action in legal_actions:
            new_pos = self.leaf.pos.make_move(action)
            if new_pos is not None:
                child = Node(new_pos, parent=self.leaf, initial_policy_value=policy_probs[action])
                self.leaf.children[action] = child

    def backpropagate(self, q_penalty, q_no_penalty):
        current = self.leaf
        while current:
            current.visit_count += 1
            current.q_sum_penalty += q_penalty
            current.q_sum_no_penalty += q_no_penalty
            q_penalty = -q_penalty
            q_no_penalty = -q_no_penalty
            current = current.parent

    def on_received_policy(self, policy_logits, q_penalty, q_no_penalty):
        if self.leaf.is_terminal():
            terminal_values = self.leaf.pos.terminal_value_with_ply_penalty(self.c_ply_penalty)
            if terminal_values is not None:
                q_penalty, q_no_penalty = terminal_values
                self.backpropagate(q_penalty, q_no_penalty)
        else:
            policy_probs = self.softmax(policy_logits)
            self.expand_leaf(policy_probs)
            self.backpropagate(q_penalty, q_no_penalty)

    def run(self, num_iterations):
        for _ in range(num_iterations):
            self.select_leaf()
            input_tensor = self.leaf.pos.write_numpy_buffer()
            policy_logits, q_penalty, q_no_penalty = self.model.forward_numpy(input_tensor)
            policy_logits = policy_logits[0]
            self.on_received_policy(policy_logits, q_penalty, q_no_penalty)

    def get_root_policy(self):
        return self.root.policy()
        
    @staticmethod
    def softmax(logits):
        logits = logits - np.max(logits)
        e = np.exp(logits)
        return e / e.sum()

@app.command()
def debug2(model: str = "best", base_dir: str = "training"):
    gen = TrainingGen.load_latest(base_dir)
    print("gen", gen)
    if model == "best":
        nn = gen.get_model(base_dir)
    elif model == "random":
        nn = RandomPlayer(ModelID(0))
    elif model == "uniform":
        nn = UniformPlayer(ModelID(0))
    else:
        raise ValueError(f"unrecognized model: {model}")

    board4 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    print("Start")
    initial_pos = Pos.from_2d_board(board4)
    mcts = MCTS(initial_pos, nn, c_exploration=5.0, c_ply_penalty=0.01)
    mcts.run(1300)
    policy = mcts.get_root_policy()
    print("Policy:", policy)
    best_move = np.argmax(policy)
    print("BestMove", best_move)

@app.command()
def nn_sweep(base_dir: str = "training"):
    perform_hparam_sweep(base_dir)

@app.command()
def mcts_sweep(
    device: str = str(get_torch_device()),
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 3,
    conv_filter_size: int = 64,
    n_policy_layers: int = 3,
    n_value_layers: int = 3,
    lr_schedule: List[float] = [0, 1e-3, 20, 5e-4],
    l2_reg: float = 1e-4,
    base_training_dir: str = "training-sweeps",
    optuna_db_path: str = "optuna.db",
    n_trials: int = 100,
    max_gens_per_trial: int = 10,
    solver_path: str = "/home/advait/connect4/c4solver",
    book_path: str = "/home/advait/connect4/7x6.book",
    solutions_path: str = "./solutions.db",
):
    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=parse_lr_schedule(lr_schedule),
        l2_reg=l2_reg,
    )

    def objective(trial: optuna.Trial):
        trial_path = Path(base_training_dir) / f"trial_{trial.number}"
        trial_path.mkdir(exist_ok=False)
        
        return training_loop(
            base_dir=str(trial_path),
            device=torch.device(device),
            n_self_play_games=trial.suggest_int("n_self_play_games", 2000, 5000),
            n_mcts_iterations=trial.suggest_int("n_mcts_iterations", 1500, 2500),
            c_exploration=trial.suggest_float("c_exploration", 3.0, 8.0),
            c_ply_penalty=c_ply_penalty,
            self_play_batch_size=self_play_batch_size,
            training_batch_size=training_batch_size,
            model_config=model_config,
            max_gens=max_gens_per_trial,
            solver_config=SolverConfig(
                solver_path=solver_path,
                book_path=book_path,
                solutions_path=solutions_path,
            ),
        ).solver_score

    study = optuna.create_study(
        storage=f"sqlite:///{optuna_db_path}",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)

@app.command()
def score(
    solver_path: str,
    book_path: str,
    base_dir: str = "training",
    solutions_path: str = "./solutions.db",
):
    gens = TrainingGen.load_all(base_dir)
    for gen in gens:
        if games := gen.get_games(base_dir):
            if gen.solver_score is None:
                gen.solver_score = games.score_policies(solver_path, book_path, solutions_path)
                gen.save_metadata(base_dir)
                logger.info("Gen {} scored: {}", gen.gen_n, gen.solver_score)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    app()