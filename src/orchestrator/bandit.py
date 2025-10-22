import random
from typing import Dict, List


class ThompsonBandit:
    def __init__(self, arms: List[str]):
        if not arms:
            raise ValueError("At least one arm is required")
        self.successes: Dict[str, int] = {arm: 1 for arm in arms}
        self.failures: Dict[str, int] = {arm: 1 for arm in arms}

    def choose(self) -> str:
        samples = {
            arm: random.betavariate(self.successes[arm], self.failures[arm])
            for arm in self.successes
        }
        return max(samples.items(), key=lambda item: item[1])[0]

    def update(self, arm: str, success: bool) -> None:
        if arm not in self.successes:
            raise KeyError(f"Unknown arm: {arm}")
        if success:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
