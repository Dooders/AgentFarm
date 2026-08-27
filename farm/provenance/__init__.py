"""Optional FarmNotary adapter. Simulation stays in AgentFarm."""

from farm.provenance.notary import farm_notary_available, notarize_run_dir

__all__ = ["farm_notary_available", "notarize_run_dir"]
