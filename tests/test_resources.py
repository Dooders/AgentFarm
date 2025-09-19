from farm.core.resources import Resource


def test_resource_depletion_and_regeneration():
    r = Resource(resource_id=1, position=(0, 0), amount=5.0, max_amount=10.0, regeneration_rate=0.5)

    assert r.is_depleted() is False

    # Consume more than available should clamp at 0
    r.consume(7.5)
    assert r.amount == 0
    assert r.is_depleted() is True

    # Regenerate should not exceed max_amount
    r.regenerate(3.0)
    assert r.amount == 3.0
    r.regenerate(100.0)
    assert r.amount == 10.0

