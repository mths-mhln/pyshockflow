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

    def _get_int(self, section: str, key: str, default: int | None = None) -> int:
        raw = self._get_raw(section, key)
        try:
            return int(raw)
        except ValueError:
            if default is not None:
                return default
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must be an integer, got '{raw}'"
            )

    def _get_float(self, section: str, key: str, positive: bool = False, default: float | None = None) -> float:
        raw = self._get_raw(section, key)
        try:
            value = float(raw)
        except ValueError:
            if default is not None:
                return default
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must be a real number, got '{raw}'"
            )
        if positive and value <= 0:
            raise ConfigError(
                f"{self.config_file} [{section}]: '{key}' must be positive, got {value}"
            )
        return value

    def _get_bool(self, section: str, key: str, default: bool | None = None) -> bool:
        try:
            raw = self._get_raw(section, key).lower()
        except ConfigError:
            if default is not None:
                return default
            raise
        if raw in ("yes", "true"):
            return True
        if raw in ("no", "false"):
            return False
        raise ConfigError(
            f"{self.config_file} [{section}]: '{key}' must be yes/no or true/false, got '{raw}'"
        )

    def _get_str(self, section: str, key: str, default: str | None = None) -> str:
        try:
            return self._get_raw(section, key).lower()
        except ConfigError:
            if default is not None:
                return default
            raise


    # [GEOMETRY]
    # ==========
    def expansionDeviceType(self) -> str:
        return self._get_str("GEOMETRY", "EXPANSION_DEVICE_TYPE")

    def deviceGeometryFilePath(self) -> str:
        return self._get_raw("GEOMETRY", "DEVICE_GEOMETRY_FILE_PATH")


    # [MESH]
    # ======
    def numberOfMeshNodes(self) -> int:
        return self._get_int("MESH", "NUM_MESH_NODES")

    def meshRefinementBool(self) -> bool:
        return self._get_bool("MESH", "MESH_REFINEMENT_BOOL")

    def refinementBoundaries(self) -> tuple[float, float]:
        start = self._get_float("MESH", "X_START_REFINEMENT")
        end = self._get_float("MESH", "X_END_REFINEMENT")
        return start, end

    def numberOfRefMeshNodes(self) -> int:
        return self._get_int("MESH", "NUM_REFINEMENT_MESH_NODES")



    # [TIME]
    # ======
    def maxTime(self) -> float:
        return self._get_float("TIME", "MAX_TIME", positive=True)



    # [INITIAL CONDITIONS]
    # ====================
    # for shocktube simulations
    # -------------------------
    def initialDensityLeft(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "DENSITY_LEFT", positive=True)

    def initialDensityRight(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "DENSITY_RIGHT", positive=True)

    def initialTemperatureLeft(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "TEMPERATURE_LEFT", positive=True)

    def initialTemperatureRight(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "TEMPERATURE_RIGHT", positive=True)

    def initialVelocityLeft(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "VELOCITY_LEFT")

    def initialVelocityRight(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "VELOCITY_RIGHT")

    def initialPressureLeft(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "PRESSURE_LEFT", positive=True)

    def initialPressureRight(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "PRESSURE_RIGHT", positive=True)


    # for nozzle simulations w/ at least one BC different than (inlet, outlet, transparent)
    # -------------------------------------------------------------------------------------
    def initialPressure(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "PRESSURE", positive=True)

    def initialVelocity(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "VELOCITY")

    def initialTemperature(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "TEMPERATURE", positive=True)

    def initialDensity(self) -> float:
        return self._get_float("INITIAL CONDITIONS", "DENSITY", positive=True)
    



    # [NUMERICS]
    # ==========
    def CFLMax(self) -> float:
        return self._get_float("NUMERICS", "CFL_MAX", positive=True)

    def numericalScheme(self) -> str:
        return self._get_str("NUMERICS", "INTERCELL_FLUX_SCHEME")

    def entropyFixActiveBool(self) -> bool:
        return self._get_bool("NUMERICS", "ENTROPY_FIX_ACTIVE_BOOL")

    def entropyFixCoefficient(self) -> float:
        return self._get_float("NUMERICS", "ENTROPY_FIX_COEFFICIENT", positive=True)

    def MUSCLReconstructionBool(self) -> bool:
        return self._get_bool("NUMERICS", "MUSCL_RECONSTRUCTION_BOOL")

    def MUSCLReconstrFluxLimiter(self) -> str:
        return self._get_str("NUMERICS", "MUSCL_RECONSTR_FLUX_LIMITER")



    # [BOUNDARY CONDITIONS]
    # =====================
    def boundaryConditions(self) -> tuple[str, str]:
        left = self._get_str("BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_LEFT")
        right = self._get_str("BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_RIGHT")
        return left, right

    def inletConditionsType(self) -> Literal["total", "static"]:
        value = self._get_str("BOUNDARY CONDITIONS", "INLET_CONDITIONS_TYPE")
        if value not in ("total", "static"):
            raise ConfigError(
                f"{self.config_file} [BOUNDARY CONDITIONS]: 'INLET_CONDITIONS_TYPE' "
                f"must be 'total' or 'static', got '{value}'"
            )
        return value  # type: ignore[return-value]

    def inletConditionsValues(self) -> list[float]:
        raw = self._get_raw("BOUNDARY CONDITIONS", "INLET_CONDITIONS_VALUES")
        try:
            values = [float(v.strip()) for v in raw.split(",")]
        except ValueError:
            raise ConfigError(
                f"{self.config_file} [BOUNDARY CONDITIONS]: 'INLET_CONDITIONS_VALUES' must be "
                f"comma-separated numbers, got '{raw}'"
            )
        expected = 3 if self.inletConditionsType() == "total" else 2
        if len(values) != expected:
            raise ConfigError(
                f"{self.config_file} [BOUNDARY CONDITIONS]: 'INLET_CONDITIONS_VALUES' expects "
                f"{expected} values for '{self.inletConditionsType()}' conditions, "
                f"got {len(values)}"
            )
        return values

    def outletConditions(self) -> float:
        return self._get_float("BOUNDARY CONDITIONS", "OUTLET_CONDITIONS")



    # [FLUID]
    # =======
    def fluidName(self) -> str:
        return self._get_raw("FLUID", "FLUID_NAME")

    def fluidModel(self) -> str:
        return self._get_str("FLUID", "FLUID_MODEL")

    def fluidGamma(self) -> float:
        return self._get_float("FLUID", "FLUID_GAMMA", positive=True)

    def gasRConstant(self) -> float:
        return self._get_float("FLUID", "GAS_R_CONSTANT", positive=True)

    def fluidLibrary(self) -> str:
        return self._get_raw("FLUID", "FLUID_LIBRARY")

    def propertyExtractionMethod(self) -> str:
        valid = {"fluid", "abstractstate", "abstractstate_v2"}
        value = self._get_str("FLUID", "PROPERTY_EXTRACTION_METHOD")
        if value not in valid:
            raise ConfigError(
                f"{self.config_file} [FLUID]: 'PROPERTY_EXTRACTION_METHOD' "
                f"must be one of {valid}, got '{value}'"
            )
        return value



    # [OUTPUT]
    # ========
    def resultsDirectoryName(self) -> str:
        return self._get_raw("OUTPUT", "RESULTS_DIRECTORY_NAME")

    def writeInterval(self) -> int:
        return self._get_int("OUTPUT", "WRITE_INTERVAL")

    def printInfoResidualsBool(self) -> bool:
        return self._get_bool("OUTPUT", "PRINT_INFO_RESIDUALS_BOOL")

    def overwriteResults(self) -> bool:
        return self._get_bool("OUTPUT", "OVERWRITE_RESULTS")