def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: trains a tiny synthetic model end-to-end (seconds, not milliseconds)",
    )
