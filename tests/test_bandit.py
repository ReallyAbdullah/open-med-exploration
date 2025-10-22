from src.orchestrator.bandit import ThompsonBandit


def test_bandit_basic():
    bandit = ThompsonBandit(["a", "b", "c"])
    arm = bandit.choose()
    assert arm in {"a", "b", "c"}
    bandit.update(arm, True)
