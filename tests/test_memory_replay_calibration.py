from memory_replay_calibration.runner import _choose_setting


def _summary(*, replay_kl: float, replay_agreement: float):
    return {
        "full_context": {"answer_token_f1": 0.8},
        "uncompressed_full_replay": {
            "answer_token_f1": 0.8,
            "kl_from_full_context": replay_kl,
            "full_argmax_agreement": replay_agreement,
        },
        "thresholds": {
            "0.0001": {
                "retained_record_ratio": 0.6,
                "kl_from_uncompressed_replay": 0.002,
                "uncompressed_argmax_agreement": 0.99,
                "answer_token_f1_delta_from_full": 0.0,
            }
        },
    }


def _criteria():
    return {
        "max_mean_kl_from_uncompressed_replay": 0.02,
        "min_mean_uncompressed_argmax_agreement": 0.98,
        "max_mean_uncompressed_replay_kl_from_full_context": 0.02,
        "min_mean_uncompressed_replay_full_argmax_agreement": 0.98,
        "max_mean_answer_f1_drop_from_full": 0.02,
    }


def test_selection_rejects_non_equivalent_uncompressed_replay():
    selection = _choose_setting(
        _summary(replay_kl=0.04, replay_agreement=0.97), _criteria()
    )

    assert selection["status"] == "uncompressed_replay_not_equivalent"
    assert selection["uncompressed_replay_gate_passed"] is False
    assert selection["uncompressed_replay_gate_checks"] == {
        "answer_f1": True,
        "kl_from_full_context": False,
        "full_argmax_agreement": False,
    }
    assert selection["selected_usage_threshold"] is None


def test_selection_accepts_threshold_after_equivalent_replay_gate():
    selection = _choose_setting(
        _summary(replay_kl=0.004, replay_agreement=0.99), _criteria()
    )

    assert selection["status"] == "accepted"
    assert selection["uncompressed_replay_gate_passed"] is True
    assert selection["selected_usage_threshold"] == 0.0001
    assert selection["selected_retained_record_ratio"] == 0.6
