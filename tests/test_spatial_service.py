import os
import sys
import unittest

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List

from farm.core.spatial_index import SpatialIndex
from farm.core.resources import Resource
from farm.core.services import SpatialIndexAdapter


class MockAgent:
    def __init__(self, agent_id: str, position: tuple[float, float]) -> None:
        self.agent_id = agent_id
        self.position = list(position)
        self.alive = True


class TestSpatialServiceAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 100
        self.height = 100
        self.index = SpatialIndex(width=self.width, height=self.height)

        # Build small world
        self.agents: List[MockAgent] = [
            MockAgent("a0", (10, 10)),
            MockAgent("a1", (15, 10)),
            MockAgent("a2", (80, 80)),
        ]
        self.resources: List[Resource] = [
            Resource(resource_id=0, position=(12, 10), amount=5, max_amount=10),
            Resource(resource_id=1, position=(60, 60), amount=5, max_amount=10),
        ]

        self.index.set_references(self.agents, self.resources)
        # initial build
        self.index.force_rebuild()

        self.service = SpatialIndexAdapter(self.index, self.width, self.height)

    def test_get_dimensions_and_bounds(self) -> None:
        self.assertEqual(self.service.get_dimensions(), (self.width, self.height))
        self.assertTrue(self.service.is_valid_position((0, 0)))
        self.assertTrue(self.service.is_valid_position((self.width, self.height)))
        self.assertFalse(self.service.is_valid_position((-1, 0)))
        self.assertFalse(self.service.is_valid_position((0, self.height + 1)))

    def test_nearby_agents(self) -> None:
        # Around a0 at (10,10) within radius 6 should include a1 but not a2
        agents = self.service.get_nearby_agents((10, 10), radius=6)
        ids = {a.agent_id for a in agents}
        self.assertIn("a0", ids)
        self.assertIn("a1", ids)
        self.assertNotIn("a2", ids)

    def test_nearby_resources(self) -> None:
        # Around (10,10) within radius 5 should include resource at (12,10)
        res = self.service.get_nearby_resources((10, 10), radius=5)
        positions = {r.position for r in res}
        self.assertIn((12, 10), positions)
        self.assertNotIn((60, 60), positions)

    def test_get_nearest_resource(self) -> None:
        nearest = self.service.get_nearest_resource((9, 10))
        self.assertIsNotNone(nearest)
        if nearest is not None:
            self.assertEqual(nearest.position, (12, 10))

    def test_mark_positions_dirty_and_update(self) -> None:
        # Move a1 and mark dirty
        self.agents[1].position = [30, 30]
        self.service.mark_positions_dirty()
        # The index defers rebuild until update is called; ensure a rebuild occurs
        # by invoking update and validating spatial query reflects new position
        self.index.update()
        agents = self.service.get_nearby_agents((10, 10), radius=6)
        ids = {a.agent_id for a in agents}
        # a1 moved away, should no longer be in nearby set
        self.assertIn("a0", ids)
        self.assertNotIn("a1", ids)


if __name__ == "__main__":
    unittest.main()