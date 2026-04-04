"""Tests for farm.analysis.null_module.null_module."""

import pandas as pd

from farm.analysis.null_module.null_module import (
    NullModule,
    null_analysis_function,
    null_data_processor,
    null_module,
)


def test_null_data_processor_passthrough():
    df = pd.DataFrame({"a": [1]})
    assert null_data_processor(df) is df


def test_null_analysis_function_returns_none():
    assert null_analysis_function(pd.DataFrame(), None) is None


def test_null_module_lazy_registration_and_processor():
    mod = NullModule()
    info = mod.get_info()
    assert info["name"] == "null"
    assert "null_function" in info["functions"]

    proc = mod.get_data_processor()
    df = pd.DataFrame({"x": [1]})
    assert proc.process(df) is df


def test_singleton_null_module_instance():
    assert isinstance(null_module, NullModule)
    assert null_module.get_info()["name"] == "null"
