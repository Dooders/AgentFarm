import unittest

from farm.utils.identity import Identity, IdentityConfig


class TestIdentity(unittest.TestCase):
    def test_simulation_id_default_prefix(self):
        identity = Identity()
        sim_id = str(identity.simulation_id())
        self.assertTrue(sim_id.startswith("sim_"))
        self.assertGreater(len(sim_id), 5)

    def test_simulation_id_custom_prefix(self):
        identity = Identity()
        sim_id = str(identity.simulation_id(prefix="test"))
        self.assertTrue(sim_id.startswith("test_"))

    def test_run_id_length(self):
        identity = Identity()
        run_id = str(identity.run_id(8))
        self.assertEqual(len(run_id), 8)

    def test_experiment_id_length(self):
        identity = Identity()
        exp_id = str(identity.experiment_id())
        self.assertGreaterEqual(len(exp_id), 10)

    def test_agent_state_round_trip(self):
        identity = Identity()
        aid = "agent_abcdef"
        sid = int(42)
        agent_state = str(identity.agent_state_id(aid, sid))
        parsed = identity.parse_agent_state_id(agent_state)
        self.assertEqual(parsed, (aid, sid))

    def test_agent_id_random(self):
        identity = Identity()
        a1 = str(identity.agent_id())
        a2 = str(identity.agent_id())
        self.assertNotEqual(a1, a2)
        self.assertTrue(a1.startswith("agent_"))
        self.assertTrue(a2.startswith("agent_"))

    def test_agent_id_deterministic(self):
        identity1 = Identity(IdentityConfig(deterministic_seed=123))
        identity2 = Identity(IdentityConfig(deterministic_seed=123))
        seq1 = [str(identity1.agent_id()) for _ in range(5)]
        seq2 = [str(identity2.agent_id()) for _ in range(5)]
        self.assertEqual(seq1, seq2)

        identity3 = Identity(IdentityConfig(deterministic_seed=456))
        seq3 = [str(identity3.agent_id()) for _ in range(5)]
        self.assertNotEqual(seq1, seq3)

    def test_genome_id_format_no_parents(self):
        """Test genome ID format for initial agents (no parents)."""
        identity = Identity()
        gid = str(identity.genome_id([]))
        self.assertEqual(gid, "::")

    def test_genome_id_format_single_parent(self):
        """Test genome ID format for cloning (single parent)."""
        identity = Identity()
        gid = str(identity.genome_id(["agent_p1"]))
        self.assertEqual(gid, "agent_p1:")

    def test_genome_id_format_two_parents(self):
        """Test genome ID format for sexual reproduction (two parents)."""
        identity = Identity()
        gid = str(identity.genome_id(["agent_p1", "agent_p2"]))
        self.assertEqual(gid, "agent_p1:agent_p2")

    def test_genome_id_counter_increments(self):
        """Test that genome ID counter increments for duplicate base IDs."""
        identity = Identity()
        # First occurrence - no counter
        gid1 = str(identity.genome_id(["agent_p1", "agent_p2"]))
        self.assertEqual(gid1, "agent_p1:agent_p2")
        
        # Simulate registry having the base - manually add it
        identity._genome_id_registry["agent_p1:agent_p2"] = -1
        
        # Second occurrence - should get counter 0
        gid2 = str(identity.genome_id(["agent_p1", "agent_p2"]))
        self.assertEqual(gid2, "agent_p1:agent_p2:0")
        
        # Third occurrence - should get counter 1
        gid3 = str(identity.genome_id(["agent_p1", "agent_p2"]))
        self.assertEqual(gid3, "agent_p1:agent_p2:1")

    def test_genome_id_registry_tracking(self):
        """Test that registry correctly tracks genome IDs."""
        identity = Identity()
        
        # Create first genome ID (no parents)
        gid1 = str(identity.genome_id([]))
        self.assertEqual(gid1, "::")
        self.assertIn("::", identity._genome_id_registry)
        
        # Create second with same base - should get counter
        gid2 = str(identity.genome_id([]))
        self.assertEqual(gid2, "::0")

    def test_genome_id_with_existing_checker(self):
        """Test genome ID generation with database checker callback."""
        identity = Identity()
        
        # Create a mock checker that says base exists
        def checker(genome_id: str) -> bool:
            return genome_id == "agent_p1:agent_p2"
        
        # First call - checker says base exists, so should get counter 0
        gid = str(identity.genome_id(["agent_p1", "agent_p2"], existing_genome_checker=checker))
        self.assertEqual(gid, "agent_p1:agent_p2:0")


if __name__ == "__main__":
    unittest.main()

