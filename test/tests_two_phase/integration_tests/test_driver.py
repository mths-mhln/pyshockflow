import pytest

from pyshockflow.driver import Driver
import unittest.mock as mock

# Strategy
# create mock config object (passed to driver init)
# create mock fluid object import (imported in driver file)
# patch Driver methods
# Create driver test in which driver is instantiated
# assert that driver object has the expected state after initialization 

@mock.patch(Driver, "extractRestartData")
@mock.patch("pyshockflow.fluid.FluidReal")
def test_driver(mock_config, mock_extractRestartData, restart_file = "empty_restart_file"):
    """
    Test that the Driver can be instantiated with a mock config and that the expected state is reached after initialization.
    """
    mock_extractRestartData.return_value = None
    driver = Driver(config = mock_config, restartFilePath=restart_file)
    assert driver.config == mock_config
    assert driver.restartFilePath == restart_file