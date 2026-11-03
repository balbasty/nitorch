import pytest
import torch

# Disable TorchScript's CPU fuser (NNC) during tests, as torch >= 2 does
# by default. With it on (torch 1.x), the profiling executor runs the
# first call(s) of a scripted function unfused and later calls through a
# fused kernel, which rounds differently (~1 ulp). Results would then
# depend on how often a function was called before (test order), and
# bitwise comparisons (e.g. `test_shoot_default_is_unchanged`) fail.
# This must be set before any scripted function is optimised: plans
# that are already compiled keep their fused kernels.
if hasattr(torch._C, '_jit_override_can_fuse_on_cpu'):
    torch._C._jit_override_can_fuse_on_cpu(False)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="also run tests marked as slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test: use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
