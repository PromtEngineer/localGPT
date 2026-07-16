from rag_system.agent.loop import Agent


def test_explicit_standalone_subject_does_not_inherit_history() -> None:
    assert not Agent._query_needs_history(
        "What is the current calibration phrase for Zephyr?"
    )


def test_pronoun_follow_up_uses_history() -> None:
    assert Agent._query_needs_history("What is its recalibration interval?")


def test_what_about_follow_up_uses_history() -> None:
    assert Agent._query_needs_history("What about the maintenance schedule?")


def test_pronoun_resolved_by_explicit_subject_does_not_use_history() -> None:
    assert not Agent._query_needs_history(
        "Give the Borealis phrase, its custodian, and its interval."
    )
