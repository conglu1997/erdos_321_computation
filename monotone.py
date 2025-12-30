"""Monotone extension controller utilities."""

from __future__ import annotations

from typing import Dict, List, Optional


class MonotoneController:
    def __init__(self, max_window: int):
        self.max_window = max(0, max_window)
        self.current_window = min(2, self.max_window) if self.max_window > 0 else 0
        self.avg_solve: Optional[float] = None
        self.avg_oracle: Optional[float] = None

    def record_solve(self, runtime: Optional[float]) -> None:
        if runtime is None:
            return
        if self.avg_solve is None:
            self.avg_solve = runtime
        else:
            self.avg_solve = 0.5 * self.avg_solve + 0.5 * runtime

    def record_oracle(self, duration: float) -> None:
        if duration < 0:
            return
        if self.avg_oracle is None:
            self.avg_oracle = duration
        else:
            self.avg_oracle = 0.5 * self.avg_oracle + 0.5 * duration

    def planned_window(self, remaining: int) -> int:
        if self.current_window <= 0:
            return 0
        window = min(self.current_window, remaining)
        if window <= 0:
            return 0
        if self.avg_oracle is not None and self.avg_solve is not None:
            est_cost = window * self.avg_oracle
            if est_cost > self.avg_solve:
                # reduce window so expected oracle cost stays under solve time
                window_cap = max(1, int(self.avg_solve / max(self.avg_oracle, 1e-9)))
                window = min(window, window_cap)
        return max(0, window)

    def update_after_attempt(self, extend_by: int, collision_found: bool) -> None:
        if extend_by == 0 or collision_found:
            if self.current_window > 1:
                self.current_window = max(1, self.current_window // 2)
        elif extend_by >= self.current_window and self.current_window < self.max_window:
            self.current_window = min(self.max_window, self.current_window + 1)
        self.current_window = min(self.current_window, self.max_window)


def summarize_monotone_stats(stats: Dict[str, List[int]], window_cap: int) -> str:
    if not stats:
        return ""
    attempts = len(stats.get("attempt_windows", []))
    total_extended = sum(stats.get("extend_by", []))
    collision_hits = sum(stats.get("collision_found", []))
    successes = sum(1 for e in stats.get("extend_by", []) if e > 0)
    success_rate = successes / attempts if attempts else 0.0
    parts = [
        f"[monotone] attempts={attempts}",
        f"extended_steps={total_extended}",
        f"collisions={collision_hits}",
        f"success_rate={success_rate:.2f}",
        f"window_cap={window_cap}",
    ]
    recommendation = ""
    if attempts == 0:
        recommendation = "monotone disabled or never attempted."
    elif (
        success_rate > 0.8
        and collision_hits == 0
        and any(w >= window_cap for w in stats.get("attempt_windows", []))
    ):
        recommendation = (
            f"High success; consider raising --monotone-window above {window_cap}."
        )
    elif success_rate < 0.2 and collision_hits > 0:
        recommendation = "Low success with collisions; consider lowering --monotone-window or disabling monotone extension."
    elif total_extended == 0:
        recommendation = (
            "No successful extensions; monotone shortcut not helping—reduce the window."
        )
    else:
        recommendation = "Monotone helping intermittently; window cap looks reasonable."
    parts.append(f"recommendation={recommendation}")
    return " | ".join(parts)
