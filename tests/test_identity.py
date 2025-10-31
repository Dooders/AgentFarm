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

    def test_genome_id_format(self):
        identity = Identity()
        gid = str(identity.genome_id("SystemAgent", 2, ["p1", "p2"], 100))
        self.assertEqual(gid, "SystemAgent:2:p1_p2:100")
        gid_none = str(identity.genome_id("Sys", 0, [], 0))
        self.assertEqual(gid_none, "Sys:0:none:0")

    def test_genome_id_with_single_parent(self):
        identity = Identity()
        gid = str(identity.genome_id("system", 1, ["agent_abc123"], 50))
        self.assertEqual(gid, "system:1:agent_abc123:50")

    def test_genome_id_with_multiple_parents(self):
        identity = Identity()
        gid = str(identity.genome_id("independent", 3, ["parent1", "parent2", "parent3"], 200))
        self.assertEqual(gid, "independent:3:parent1_parent2_parent3:200")

    def test_genome_id_deterministic(self):
        identity1 = Identity(IdentityConfig(deterministic_seed=123))
        identity2 = Identity(IdentityConfig(deterministic_seed=123))
        
        gid1 = str(identity1.genome_id("system", 0, [], 0))
        gid2 = str(identity2.genome_id("system", 0, [], 0))
        # Note: genome_id format doesn't use deterministic generation, 
        # it's just string formatting, so they should match
        self.assertEqual(gid1, gid2)


if __name__ == "__main__":
    unittest.main()

