def test_database_services_package_importable():
    import farm.database.services as svc  # noqa: PLC0415

    assert svc.__all__ == []
