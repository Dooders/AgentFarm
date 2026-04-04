"""Light coverage for analysis module data processor entry points."""

import pandas as pd
import pytest

from farm.analysis.advantage.data import process_advantage_data
from farm.analysis.genesis.data import process_genesis_data


def test_process_advantage_data_dataframe():
    df = pd.DataFrame({"a": [1]})
    assert process_advantage_data(df) is df


def test_process_advantage_data_non_dataframe_returns_empty():
    out = process_advantage_data(object())
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_process_genesis_data_dataframe():
    df = pd.DataFrame({"a": [1]})
    assert process_genesis_data(df) is df


def test_process_genesis_data_session_like_not_implemented():
    class FakeSession:
        def execute(self, *_args, **_kwargs):
            return None

    with pytest.raises(NotImplementedError):
        process_genesis_data(FakeSession())


def test_process_genesis_data_unsupported_type():
    with pytest.raises(TypeError, match="Unsupported data type"):
        process_genesis_data(42)
