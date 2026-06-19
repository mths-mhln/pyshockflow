import configparser
import os
from typing import Literal


class ConfigError(ValueError):
    """Raised when a config value is missing or invalid."""


class Config:
    def __init__(self, config_file: str = "input.ini"):
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Config file '{config_file}' not found. "
                f"Check path relative to cwd: {os.getcwd()}"
            )
        self.config_file = config_file
        self._parser = configparser.ConfigParser()
        self._parser.read(config_file)

    # utils
    def _get_raw(self, section: str, key: str) -> str:
        try:
            return self._parser.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            raise ConfigError(
                f"{self.config_file} [{section}]: missing key '{key}'"
            ) from e

    def _get_int(self, section: str, key: str) -> int:
        raw = self._get_raw(section, key)
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must be an integer, got '{raw}'"
            )

    def _get_float(self, section: str, key: str, positive: bool = False) -> float:
        raw = self._get_raw(section, key)
        try:
            value = float(raw)
        except ValueError:
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must be a real number, got '{raw}'"
            )
        if positive and value <= 0:
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must be positive, got {value}"
            )
        return value

    def _get_bool(self, section: str, key: str, fallback: bool | None = None) -> bool:
        try:
            raw = self._get_raw(section, key).lower()
        except ConfigError:
            if fallback is not None:
                return fallback
            raise
        if raw in ("yes", "true"):
            return True
        if raw in ("no", "false"):
            return False
        raise ConfigError(
            f"{self.config_file} [{section}]: '{key}' must be yes/no or true/false, got '{raw}'"
        )

    def _get_str(self, section: str, key: str, fallback: str | None = None) -> str:
        try:
            return self._get_raw(section, key).lower()
        except ConfigError:
            if fallback is not None:
                return fallback
            raise

    # extraction tools
    def getNumberOfPoints(self) -> int:
        return self._get_int("SIMULATION", "NUMBER_POINTS")

    def getLength(self) -> float:
        return self._get_float("GEOMETRY", "LENGTH", positive=True)

    def getPressureLeft(self) -> float:
        return self._get_float("SIMULATION", "PRESSURE_LEFT", positive=True)

    def getPressureRight(self) -> float:
        return self._get_float("SIMULATION", "PRESSURE_RIGHT", positive=True)

    def getDensityLeft(self) -> float:
        return self._get_float("SIMULATION", "DENSITY_LEFT", positive=True)

    def getDensityRight(self) -> float:
        return self._get_float("SIMULATION", "DENSITY_RIGHT", positive=True)

    def getTemperatureLeft(self) -> float:
        return self._get_float("SIMULATION", "TEMPERATURE_LEFT", positive=True)

    def getTemperatureRight(self) -> float:
        return self._get_float("SIMULATION", "TEMPERATURE_RIGHT", positive=True)

    def getVelocityLeft(self) -> float:
        return self._get_float("SIMULATION", "VELOCITY_LEFT")

    def getVelocityRight(self) -> float:
        return self._get_float("SIMULATION", "VELOCITY_RIGHT")

    def getCFLMax(self) -> float:
        return self._get_float("SIMULATION", "CFL_MAX", positive=True)

    def getTimeMax(self) -> float:
        return self._get_float("SIMULATION", "TIME_MAX", positive=True)

    def getTimeStepMethod(self) -> str:
        return self._get_str("SIMULATION", "TIME_STEP_METHOD", fallback="constant")

    def getFluidName(self) -> str:
        return self._get_raw("FLUID", "FLUID_NAME")

    def getFluidModel(self) -> str:
        return self._get_str("FLUID", "FLUID_MODEL")

    def getFluidGamma(self) -> float:
        return self._get_float("FLUID", "FLUID_GAMMA", positive=True)

    def getGasRConstant(self) -> float:
        return self._get_float("FLUID", "GAS_R_CONSTANT", positive=True)

    def getInterfaceLocation(self) -> float:
        return self._get_float("GEOMETRY", "INTERFACE_LOCATION")

    def getBoundaryConditions(self) -> tuple[str, str]:
        left = self._get_str("SIMULATION", "BOUNDARY_CONDITION_LEFT")
        right = self._get_str("SIMULATION", "BOUNDARY_CONDITION_RIGHT")
        return left, right

    def getInletConditionsType(self) -> Literal["total", "static"]:
        value = self._get_str("SIMULATION", "INLET_CONDITIONS_TYPE", fallback="total")
        if value not in ("total", "static"):
            raise ConfigError(
                f"{self.config_file} [SIMULATION]: 'INLET_CONDITIONS_TYPE' "
                f"must be 'total' or 'static', got '{value}'"
            )
        return value  # type: ignore[return-value]

    def getInletConditions(self) -> list[float]:
        raw = self._get_raw("SIMULATION", "INLET_CONDITIONS")
        try:
            values = [float(v.strip()) for v in raw.split(",")]
        except ValueError:
            raise ConfigError(
                f"{self.config_file} [SIMULATION]: 'INLET_CONDITIONS' must be "
                f"comma-separated numbers, got '{raw}'"
            )
        expected = 3 if self.getInletConditionsType() == "total" else 2
        if len(values) != expected:
            raise ConfigError(
                f"{self.config_file} [SIMULATION]: 'INLET_CONDITIONS' expects "
                f"{expected} values for '{self.getInletConditionsType()}' conditions, "
                f"got {len(values)}"
            )
        return values

    def getOutletConditions(self) -> float:
        return self._get_float("SIMULATION", "OUTLET_CONDITIONS")

    def getNumericalScheme(self) -> str:
        return self._get_str("SIMULATION", "NUMERICAL_SCHEME")

    def getFluxLimiter(self) -> str:
        return self._get_str("SIMULATION", "FLUX_LIMITER", fallback="van albada")

    def getResultsDirectoryName(self) -> str:
        return self._get_raw("OUTPUT", "RESULTS_DIRECTORY_NAME")

    def showAnimation(self) -> bool:
        return self._get_bool("OUTPUT", "SHOW_ANIMATION", fallback=False)

    def isMusclActive(self) -> bool:
        return self._get_bool("SIMULATION", "MUSCL_RECONSTRUCTION", fallback=False)

    def isMeshRefined(self) -> bool:
        return self._get_bool("SIMULATION", "MESH_REFINEMENT", fallback=False)

    def isEntropyFixActive(self) -> bool:
        return self._get_bool("SIMULATION", "ENTROPY_FIX_ACTIVE", fallback=True)

    def getEntropyFixCoefficient(self) -> float:
        try:
            return self._get_float("SIMULATION", "ENTROPY_FIX_COEFFICIENT", positive=True)
        except ConfigError:
            return 0.2

    def getFluidLibrary(self) -> str:
        return self._get_raw("FLUID", "FLUID_LIBRARY")

    def getPropertyExtractionMethod(self) -> str:
        valid = {"fluid", "abstractstate", "abstractstate_v2"}
        value = self._get_str("FLUID", "PROPERTY_EXTRACTION_METHOD", fallback="fluid")
        if value not in valid:
            raise ConfigError(
                f"{self.config_file} [FLUID]: 'PROPERTY_EXTRACTION_METHOD' "
                f"must be one of {valid}, got '{value}'"
            )
        return value

    def adaptMeshRefinementExtremities(self) -> bool:
        return self._get_bool("SIMULATION", "ADAPT_MESH_REFINEMENT", fallback=False)

    def getRefinementBoundaries(self) -> tuple[float, float]:
        start = self._get_float("SIMULATION", "X_START_REFINEMENT")
        end = self._get_float("SIMULATION", "X_END_REFINEMENT")
        return start, end

    def getNumberPointsRefinement(self) -> int:
        return self._get_int("SIMULATION", "NUMBER_POINTS_REFINEMENT")

    def getTopology(self) -> str:
        return self._get_str("GEOMETRY", "TOPOLOGY", fallback="default")

    def getNozzleFilePath(self) -> str:
        return self._get_raw("GEOMETRY", "NOZZLE_FILEPATH")

    def getAreaReference(self) -> float:
        try:
            return self._get_float("GEOMETRY", "REFERENCE_AREA", positive=True)
        except ConfigError:
            return 1.0

    def getWriteInterval(self) -> int:
        try:
            return self._get_int("OUTPUT", "WRITE_INTERVAL")
        except ConfigError:
            return 250

    def getOverwriteResults(self) -> bool:
        return self._get_bool("OUTPUT", "OVERWRITE_RESULTS", fallback=False)

    def getPrintInfoResidualsBool(self) -> bool:
        return self._get_bool("OUTPUT", "PRINT_INFO_RESIDUALS", fallback=False)