"""
Cleaning Bot in an Unknown Environment

Required packages:
    pip install numpy matplotlib pandas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import csv
import random

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd


# Basic types and helpers

Coord = Tuple[int, int]

DIRS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}

DIR_ORDER = ["N", "E", "S", "W"]


def add_coords(a: Coord, b: Coord) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Environment

class GridWorld:
    """
    Simulator-side world.
    The agent does not know this full map initially.
    """

    def __init__(
        self,
        rows: int = 18,
        cols: int = 24,
        obstacle_density: float = 0.20,
        dirt_density: float = 0.30,
        seed: int = 42,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.rng = random.Random(seed)

        # 1 = obstacle, 0 = free
        self.grid = np.ones((rows, cols), dtype=int)
        self.dirt = np.zeros((rows, cols), dtype=int)

        self.start = self._generate_connected_free_shape(obstacle_density)
        self._scatter_dirt(dirt_density)

    def _generate_connected_free_shape(self, obstacle_density: float) -> Coord:
        
        # Create a connected free region with irregular shape
        
        target_free = max(25, int(self.rows * self.cols * (1 - obstacle_density)))

        start = (self.rows // 2, self.cols // 2)
        self.grid[start] = 0

        frontier = [start]
        free_cells: Set[Coord] = {start}

        while frontier and len(free_cells) < target_free:
            cell = frontier[self.rng.randrange(len(frontier))]
            directions = DIR_ORDER[:]
            self.rng.shuffle(directions)

            expanded = False
            for d in directions:
                nxt = add_coords(cell, DIRS[d])

                if not self.in_bounds(nxt):
                    continue
                if nxt in free_cells:
                    continue

                free_neighbors = 0
                for dd in DIR_ORDER:
                    nn = add_coords(nxt, DIRS[dd])
                    if nn in free_cells:
                        free_neighbors += 1

                # Bias toward irregular but connected shapes
                prob = 0.80 if free_neighbors >= 1 else 0.35

                if self.rng.random() < prob:
                    free_cells.add(nxt)
                    self.grid[nxt] = 0
                    frontier.append(nxt)
                    expanded = True
                    break

            if not expanded:
                frontier.remove(cell)

        # Carve a few extra free patches to create richer shapes
        free_list = list(free_cells)
        extra_patches = max(2, (self.rows * self.cols) // 150)
        for _ in range(extra_patches):
            cx, cy = free_list[self.rng.randrange(len(free_list))]
            h = self.rng.randint(1, 2)
            w = self.rng.randint(1, 2)
            for r in range(max(0, cx - h), min(self.rows, cx + h + 1)):
                for c in range(max(0, cy - w), min(self.cols, cy + w + 1)):
                    self.grid[r, c] = 0
                    free_cells.add((r, c))

        return start

    def _scatter_dirt(self, dirt_density: float) -> None:
        free_cells = list(zip(*np.where(self.grid == 0)))
        for cell in free_cells:
            if cell == self.start:
                continue
            self.dirt[cell] = 1 if self.rng.random() < dirt_density else 0

    def in_bounds(self, cell: Coord) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, cell: Coord) -> bool:
        return self.in_bounds(cell) and self.grid[cell] == 0

    def is_dirty(self, cell: Coord) -> bool:
        return self.is_free(cell) and self.dirt[cell] == 1

    def clean(self, cell: Coord) -> None:
        if self.is_free(cell):
            self.dirt[cell] = 0

    def neighbors(self, cell: Coord) -> List[Coord]:
        result = []
        for d in DIR_ORDER:
            nxt = add_coords(cell, DIRS[d])
            if self.is_free(nxt):
                result.append(nxt)
        return result

    def reachable_free_cells(self) -> Set[Coord]:
        q = deque([self.start])
        seen = {self.start}

        while q:
            cur = q.popleft()
            for nxt in self.neighbors(cur):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return seen

    def remaining_dirt_count(self) -> int:
        return int(self.dirt.sum())

# Agent memory and statistics

@dataclass
class CellBelief:
    discovered: bool = False
    free: Optional[bool] = None
    dirty: Optional[bool] = None
    cleaned: bool = False
    visited_count: int = 0


@dataclass
class RunStats:
    steps: int = 0
    moves: int = 0
    cleans: int = 0
    revisits: int = 0
    discovered_cells: int = 0
    completion_ratio: float = 0.0
    path: List[Coord] = field(default_factory=list)


# Frontier-guided cleaning agent

class FrontierCleaningAgent:
    
    # Cleaning agent with local perception, internal memory, frontier-guided exploration & optional pheromone-inspired memory

    def __init__(self, world: GridWorld, seed: int = 0) -> None:
        self.world = world
        self.rng = random.Random(seed)

        self.pos = world.start
        self.beliefs: Dict[Coord, CellBelief] = {}
        self.blocked_edges: Set[Tuple[Coord, Coord]] = set()
        self.frontier: Set[Coord] = set()
        self.stats = RunStats(path=[self.pos])

        # Pheromone-inspired memory
        self.pheromone: Dict[Coord, float] = {}
        self.evaporation_rate = 0.05
        self.deposit_amount = 1.0

        self._ensure_cell(self.pos)
        self._perceive_local()

    def _ensure_cell(self, cell: Coord) -> None:
        if cell not in self.beliefs:
            self.beliefs[cell] = CellBelief()
        if cell not in self.pheromone:
            self.pheromone[cell] = 0.0

    def _mark_blocked(self, a: Coord, b: Coord) -> None:
        self.blocked_edges.add((a, b))
        self.blocked_edges.add((b, a))

    def _perceive_local(self) -> None:
        
        # Agent perceives current cell and immediate neighbors

        self._ensure_cell(self.pos)

        cur = self.beliefs[self.pos]
        cur.discovered = True
        cur.free = True
        cur.dirty = self.world.is_dirty(self.pos)
        cur.visited_count += 1

        if cur.visited_count > 1:
            self.stats.revisits += 1

        for d in DIR_ORDER:
            nxt = add_coords(self.pos, DIRS[d])
            self._ensure_cell(nxt)

            if self.world.in_bounds(nxt):
                if self.world.is_free(nxt):
                    self.beliefs[nxt].discovered = True
                    self.beliefs[nxt].free = True
                    self.beliefs[nxt].dirty = self.world.is_dirty(nxt)
                else:
                    self.beliefs[nxt].discovered = True
                    self.beliefs[nxt].free = False
                    self._mark_blocked(self.pos, nxt)
            else:
                self.beliefs[nxt].discovered = True
                self.beliefs[nxt].free = False
                self._mark_blocked(self.pos, nxt)

        self._refresh_frontier()

    def _refresh_frontier(self) -> None:
        
        # Frontier = known free cell that borders unknown space
       
        self.frontier.clear()

        for cell, belief in self.beliefs.items():
            if belief.free is True:
                for d in DIR_ORDER:
                    nxt = add_coords(cell, DIRS[d])
                    nb = self.beliefs.get(nxt)
                    if nb is None or nb.free is None:
                        self.frontier.add(cell)
                        break

    def evaporate_pheromone(self) -> None:
        for cell in list(self.pheromone.keys()):
            self.pheromone[cell] *= (1 - self.evaporation_rate)

    def clean_if_needed(self) -> None:
        if self.world.is_dirty(self.pos):
            self.world.clean(self.pos)
            belief = self.beliefs[self.pos]
            belief.dirty = False
            belief.cleaned = True
            self.stats.cleans += 1
            self.stats.steps += 1

    def reachable_known_neighbors(self, cell: Coord) -> List[Coord]:
        result = []
        for d in DIR_ORDER:
            nxt = add_coords(cell, DIRS[d])
            b = self.beliefs.get(nxt)
            if b and b.free is True and (cell, nxt) not in self.blocked_edges:
                result.append(nxt)
        return result

    def unvisited_neighbors(self) -> List[Coord]:
        candidates = []
        for nxt in self.reachable_known_neighbors(self.pos):
            if self.beliefs[nxt].visited_count == 0:
                candidates.append(nxt)
        return candidates

    def heuristic_score(self, nxt: Coord) -> Tuple[int, float, int]:
        """
        Lower score is better:
        1) dirty cells first
        2) lower pheromone + fewer visits
        3) smaller coordinate distance as tie-break
        """
        b = self.beliefs[nxt]
        dirty_priority = 0 if (b.dirty is True) else 1
        pher = self.pheromone.get(nxt, 0.0)
        return (dirty_priority, pher + b.visited_count, abs(nxt[0]) + abs(nxt[1]))

    def move_to(self, nxt: Coord) -> bool:
        self.stats.steps += 1

        if self.world.is_free(nxt):
            self.pos = nxt
            self.stats.moves += 1
            self.stats.path.append(self.pos)
            self.pheromone[self.pos] += self.deposit_amount
            self._perceive_local()
            return True

        self._mark_blocked(self.pos, nxt)
        self._ensure_cell(nxt)
        self.beliefs[nxt].discovered = True
        self.beliefs[nxt].free = False
        return False

    def shortest_known_path(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        q = deque([start])
        prev: Dict[Coord, Optional[Coord]] = {start: None}

        while q:
            cur = q.popleft()
            if cur == goal:
                break

            for nxt in self.reachable_known_neighbors(cur):
                if nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)

        if goal not in prev:
            return None

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def nearest_frontier(self) -> Optional[Coord]:
        if not self.frontier:
            return None

        candidates = list(self.frontier)
        candidates.sort(
            key=lambda c: (manhattan(self.pos, c), self.beliefs[c].visited_count)
        )
        return candidates[0]

    def termination_condition(self) -> bool:
        if self.frontier:
            return False

        # All known reachable free cells must be clean
        for cell, belief in self.beliefs.items():
            if belief.free is True and self.world.is_dirty(cell):
                return False

        # No free cell should border unknown space
        for cell, belief in self.beliefs.items():
            if belief.free is True:
                for d in DIR_ORDER:
                    nxt = add_coords(cell, DIRS[d])
                    nb = self.beliefs.get(nxt)
                    if nb is None or nb.free is None:
                        return False

        return True

    def run(self, max_steps: int = 50000) -> RunStats:
        while self.stats.steps < max_steps:
            self.evaporate_pheromone()
            self.clean_if_needed()

            # Prefer unvisited neighbors
            unvisited = self.unvisited_neighbors()
            if unvisited:
                unvisited.sort(key=self.heuristic_score)
                self.move_to(unvisited[0])
                continue

            # Otherwise go to nearest frontier
            frontier_target = self.nearest_frontier()
            if frontier_target is not None and frontier_target != self.pos:
                path = self.shortest_known_path(self.pos, frontier_target)
                if path and len(path) > 1:
                    self.move_to(path[1])
                    continue

            # Stop if environment seems fully explored and cleaned
            if self.termination_condition():
                break

            # Fallback: move among known neighbors
            known_neighbors = self.reachable_known_neighbors(self.pos)
            if known_neighbors:
                known_neighbors.sort(key=self.heuristic_score)
                self.move_to(known_neighbors[0])
                continue

            # Dead-end safety
            self.stats.steps += 1

        reachable = self.world.reachable_free_cells()
        discovered_free = {c for c, b in self.beliefs.items() if b.free is True}

        self.stats.discovered_cells = len(discovered_free)

        cleaned_count = sum(1 for c in reachable if not self.world.is_dirty(c))
        self.stats.completion_ratio = cleaned_count / max(1, len(reachable))

        return self.stats


# Random baseline agent

class RandomCleaningAgent:
    def __init__(self, world: GridWorld, seed: int = 0) -> None:
        self.world = world
        self.rng = random.Random(seed)
        self.pos = world.start
        self.stats = RunStats(path=[self.pos])
        self.visits: Dict[Coord, int] = {self.pos: 1}

    def run(self, max_steps: int = 50000) -> RunStats:
        while self.stats.steps < max_steps and self.world.remaining_dirt_count() > 0:
            if self.world.is_dirty(self.pos):
                self.world.clean(self.pos)
                self.stats.cleans += 1
                self.stats.steps += 1

            neighbors = self.world.neighbors(self.pos)
            if not neighbors:
                break

            nxt = self.rng.choice(neighbors)
            self.pos = nxt
            self.stats.moves += 1
            self.stats.steps += 1
            self.stats.path.append(self.pos)

            self.visits[self.pos] = self.visits.get(self.pos, 0) + 1
            if self.visits[self.pos] > 1:
                self.stats.revisits += 1

        reachable = self.world.reachable_free_cells()
        cleaned_count = sum(1 for c in reachable if not self.world.is_dirty(c))
        self.stats.discovered_cells = len(reachable)
        self.stats.completion_ratio = cleaned_count / max(1, len(reachable))

        return self.stats

# Visualization

def render_world(world: GridWorld, path: Optional[List[Coord]] = None, title: str = ""):
    """
    Legend:
    0 obstacle
    1 free
    2 dirty
    3 start
    4 visited
    5 final agent position
    """
    arr = np.full(world.grid.shape, 0)
    arr[world.grid == 0] = 1
    arr[(world.grid == 0) & (world.dirt == 1)] = 2
    arr[world.start] = 3

    if path:
        for cell in path:
            if arr[cell] == 1:
                arr[cell] = 4
        arr[path[-1]] = 5

    cmap = mcolors.ListedColormap([
        "#202020",  # obstacle
        "#f4f4f4",  # free
        "#ffcc33",  # dirty
        "#66bbff",  # start
        "#99e699",  # visited
        "#ff6666",  # agent final position
    ])
    bounds = np.arange(-0.5, 6.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(arr, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return fig


def save_run_figure(world: GridWorld, stats: RunStats, output_path: Path, title: str) -> None:
    fig = render_world(world, stats.path, title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# Experiment helpers

def clone_world(world: GridWorld) -> GridWorld:
    
    # Create a copy so both agents run on the same environment

    clone = GridWorld(world.rows, world.cols, 0.2, 0.2, 1)
    clone.grid = world.grid.copy()
    clone.dirt = world.dirt.copy()
    clone.start = world.start
    return clone


def run_experiments(output_dir: str = "cleaning_bot_outputs", seed: int = 7) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs = [
        {"name": "small", "rows": 12, "cols": 16, "obstacle_density": 0.15, "dirt_density": 0.25},
        {"name": "medium", "rows": 18, "cols": 24, "obstacle_density": 0.20, "dirt_density": 0.30},
        {"name": "large", "rows": 28, "cols": 36, "obstacle_density": 0.24, "dirt_density": 0.32},
    ]

    records = []

    for i, cfg in enumerate(configs):
        world = GridWorld(
            rows=cfg["rows"],
            cols=cfg["cols"],
            obstacle_density=cfg["obstacle_density"],
            dirt_density=cfg["dirt_density"],
            seed=seed + i,
        )

        # Save initial environment
        fig = render_world(world, None, f"Initial environment: {cfg['name']}")
        fig.tight_layout()
        fig.savefig(out / f"{cfg['name']}_initial.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # Frontier-guided agent
        fw = clone_world(world)
        frontier_agent = FrontierCleaningAgent(fw, seed=seed + i)
        frontier_stats = frontier_agent.run()
        save_run_figure(
            fw,
            frontier_stats,
            out / f"{cfg['name']}_frontier.png",
            f"Frontier-guided result: {cfg['name']}",
        )

        # Random baseline
        rw = clone_world(world)
        random_agent = RandomCleaningAgent(rw, seed=seed + i)
        random_stats = random_agent.run()
        save_run_figure(
            rw,
            random_stats,
            out / f"{cfg['name']}_random.png",
            f"Random baseline result: {cfg['name']}",
        )

        records.append(
            {
                "environment": cfg["name"],
                "agent": "frontier_guided",
                "rows": cfg["rows"],
                "cols": cfg["cols"],
                "steps": frontier_stats.steps,
                "moves": frontier_stats.moves,
                "cleans": frontier_stats.cleans,
                "revisits": frontier_stats.revisits,
                "discovered_cells": frontier_stats.discovered_cells,
                "completion_ratio": round(frontier_stats.completion_ratio, 4),
            }
        )

        records.append(
            {
                "environment": cfg["name"],
                "agent": "random_baseline",
                "rows": cfg["rows"],
                "cols": cfg["cols"],
                "steps": random_stats.steps,
                "moves": random_stats.moves,
                "cleans": random_stats.cleans,
                "revisits": random_stats.revisits,
                "discovered_cells": random_stats.discovered_cells,
                "completion_ratio": round(random_stats.completion_ratio, 4),
            }
        )

    # Save CSV
    csv_path = out / "experiment_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Save comparison charts
    df = pd.DataFrame(records)

    for metric in ["steps", "revisits", "completion_ratio"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot = df.pivot(index="environment", columns="agent", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Comparison by {metric}")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"comparison_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("\nResults saved to:", out.resolve())
    print("\nExperiment Summary:\n")
    print(df.to_string(index=False))


# Main

if __name__ == "__main__":
    run_experiments()