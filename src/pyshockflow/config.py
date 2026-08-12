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

        # perform input file verification checks
        self.inputFileCheck()


    # ------------------------------------
    # input type verification helpers
    # ------------------------------------
    def _get_raw(self, section: str, key: str) -> str:
        """Allows for special characters, necessary for e.g. fluid name R1234ze(E) """
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

    def _get_str(self, section: str, key: str, default: str, inputOptions: list[str] | None = None) -> str:
        try:
            value =  self._get_raw(section, key).lower()
            if inputOptions is not None and value not in inputOptions:
                raise ConfigError(
                    f"{self.config_file} [{section}]: '{key}' must be one of {inputOptions}, got '{value}'"
                )
        except ConfigError:
            if default is not None:
                return default
            raise


    # ------------------------------------------------------------
    # check whether required inputs are present
    # ------------------------------------------------------------
    def inputFileCheck(self) -> None:
        """
        Three types of input verification are required. 
        1) verification that the necessary sections and keys are present in the config 
           file provided by the user based on the settings that are, and are not present.
        2) verification that the input values are of the correct type (int, float, bool, str)
        3) verification that the input values are within the acceptable set. This can be 
           a set of strings, or an acceptable range.
        
        Verification of these three are performed at different levels of the code. 
        (1) is performed in this method
        (2) is performed using the input type verification helpers specified above, 
            and through the use of the correct helper for each input parsing method, 
            which are specified below.
        (3) is performed in the input parsing methods below. 

        Hence the job of this method is purely to verify that the necessary sections and 
        keys are present in the input file for the analysis specified by the user. There is 
        some difficulty in doing so:
        a) some input options are only necessary for nozzle simulations, some input options 
        only for shocktube simulations. 
        b) some input options are conditionally required, when another input option is specified
        c) some input options are conditionally prohibited when another  input is specified

        The following method logic should be able to handle these situations.
        """
        # helper function
        # ===============
        def check_required_sections(required_sections: dict[str, list[str]]) -> None:
            """
            Check that all required sections and keys, specified in required_sections,
            are present in the config file. required_sections is a dictionary of form
            {"section": ["key1", "key2", ...]}
            """
            for section, keys in required_sections.items():
                if not self._parser.has_section(section):
                    raise ConfigError(f"{self.config_file}: missing section '{section}'")
                for key in keys:
                    if not self._parser.has_option(section, key):
                        raise ConfigError(f"{self.config_file} [{section}]: missing necessary setting in config file: '{key}'")
                    

        # specify required sections and keys for nozzle and shocktube simulations
        # =======================================================================
        requiredSectionsNKeysNozzle = {
            "GEOMETRY": ["EXPANSION_DEVICE_TYPE", "DEVICE_GEOMETRY_FILE_PATH"],
            "MESH": ["NUM_MESH_NODES", "MESH_REFINEMENT_BOOL"],
            "TIME": ["MAX_TIME"],
            "NUMERICS": ["INTERCELL_FLUX_SCHEME", "CFL_MAX"], 
            "BOUNDARY CONDITIONS": ["BOUNDARY_CONDITION_LEFT", "BOUNDARY_CONDITION_RIGHT"],
            "FLUID": ["FLUID_NAME", "FLUID_MODEL"],
            "OUTPUT": ["RESULTS_DIRECTORY_NAME"]
            }
        requiredSectionsNKeysShocktube = {
            "GEOMETRY": ["EXPANSION_DEVICE_TYPE", "DEVICE_GEOMETRY_FILE_PATH"],
            "MESH": ["NUM_MESH_NODES", "MESH_REFINEMENT_BOOL"],
            "TIME": ["MAX_TIME"], 
            "INITIAL CONDITIONS": ["PRESSURE_LEFT", "PRESSURE_RIGHT", "VELOCITY_LEFT", "VELOCITY_RIGHT"],
            "NUMERICS": ["INTERCELL_FLUX_SCHEME", "CFL_MAX"],
            "BOUNDARY CONDITIONS": ["BOUNDARY_CONDITION_LEFT", "BOUNDARY_CONDITION_RIGHT"],
            "FLUID": ["FLUID_NAME", "FLUID_MODEL"],
            "OUTPUT": ["RESULTS_DIRECTORY_NAME"]
            }

        # check for presence of required sections and keys
        # ================================================
        # In order to split the logic into nozzle and shocktube simulations, we need to
        # extract the expansion device type from the config file. However, at this stage
        # we cannot know if that key is specified, hence we need to handle the device type
        # check separately. Yes we will check for presence of this key twice in that case
        # but I did not want to confuse the reader at first by not specifying the expansion
        # device type check in the lists above.
        if not self._parser.has_section("GEOMETRY"):
            raise ConfigError(f"{self.config_file}: missing section 'GEOMETRY'")
        if not self._parser.has_option("GEOMETRY", "EXPANSION_DEVICE_TYPE"):
            raise ConfigError(f"{self.config_file} [GEOMETRY]: missing necessary setting in config file: 'EXPANSION_DEVICE_TYPE'")

        # Now that we are sure the key is there, check for the required sections and keys
        # using the helper function created for it.
        expansion_device_type = self._get_str("GEOMETRY", "EXPANSION_DEVICE_TYPE")
        if expansion_device_type == "nozzle":
            check_required_sections(requiredSectionsNKeysNozzle)
        elif expansion_device_type == "shocktube":
            check_required_sections(requiredSectionsNKeysShocktube)

        # check common conditionally required inputs
        # ==========================================
        # there are a multitude of conditionally required inputs Some only applicable
        # to nozzle simulations, some only applicable to shocktube simulations, and some to both.
        # for common conditional requirements are stored in dictionaries according to the 
        # following format:
        # [
        #   [
        #     ["section", "key", "value"],
        #     [{"section": ["key1", "key2", ...]}]
        #   ],
        #   [
        #     ["section", "key", "value"],
        #     [{"section": ["key1", "key2", ...]}]
        #   ]
        # ]
        #
        # For the first list, section is the section in which the conditional key resides. 
        # The conditional key is a required key, and the key, section and value are hence 
        # always specified. 
        # 
        # For the second list, the section is the section in which the conditionally required 
        # keys should reside to comply with the imposed format outlined in the input guidelines.
        # the key is the necessary key that should be present in the config file if the value 
        # of the required key in the first list in the input file carries a value = "value".

        condRequiredCommon = [
            [
                ["MESH", "MESH_REFINEMENT_BOOL", True], 
                [{"MESH": ["X_START_REFINEMENT", "X_END_REFINEMENT", "NUM_REFINEMENT_MESH_NODES"]}]
            ],
            [
                ["NUMERICS", "INTERCELL_FLUX_SCHEME", ["roe", "roe_arabi", "roe_vinokur"]], 
                [{"NUMERICS": ["ENTROPY_FIX_ACTIVE_BOOL", "ENTROPY_FIX_COEFFICIENT", "MUSCL_RECONSTRUCTION_BOOL", "MUSCL_RECONSTR_FLUX_LIMITER"]}]
            ],
            [
                ["BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_LEFT", "inlet"], 
                [{"BOUNDARY CONDITIONS": ["INLET_CONDITIONS_TYPE", "INLET_CONDITIONS_VALUES"]}]
            ],
            [
                ["BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_LEFT", "outlet"], 
                [{"BOUNDARY CONDITIONS": ["OUTLET_CONDITIONS"]}]
            ],
            [
                ["BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_RIGHT", "inlet"], 
                [{"BOUNDARY CONDITIONS": ["INLET_CONDITIONS_TYPE", "INLET_CONDITIONS_VALUES"]}]
            ],
            [
                ["BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_RIGHT", "outlet"], 
                [{"BOUNDARY CONDITIONS": ["OUTLET_CONDITIONS"]}]
            ],
            [
                ["FLUID", "FLUID_MODEL", "ideal"],
                [{"FLUID": ["FLUID_GAMMA", "GAS_R_CONSTANT"]}]
            ],
            [
                ["FLUID", "FLUID_MODEL", "real"],
                [{"FLUID": ["FLUID_LIBRARY"]}]
            ]
        ]
        for section, key, trigger_values, required in condRequiredCommon:
            if self._get_raw(section, key) in trigger_values:
                check_required_sections(required)

        # check expansion device specific conditionally required inputs
        # =============================================================
        # for both nozzle and shocktube simulations, there are some conditionally required
        # inputs that are specific to the expansion device type. They also require specific 
        # handling, which is dealt with below.
        if expansion_device_type == "nozzle":
            # If the boundary condition pair is not one of the allowed inlet/outlet/transparent
            # combinations, the initial conditions must be fully specified by the user.
            bc_left  = self._get_raw("BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_LEFT")
            bc_right = self._get_raw("BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_RIGHT")
            allowed_pairs = [
                ("transparent", "inlet"), ("inlet", "transparent"),
                ("inlet", "outlet"),      ("outlet", "inlet"),
                ("outlet", "transparent"), ("transparent", "outlet"),
            ]
            if (bc_left, bc_right) not in allowed_pairs:
                check_required_sections(
                    {"INITIAL CONDITIONS": ["PRESSURE", "VELOCITY", "TEMPERATURE", "DENSITY"]}
                )

        elif expansion_device_type == "shocktube":
            # Exactly one of {DENSITY, TEMPERATURE} (left and right) must be specified alongside
            # If DENSITY is given, TEMPERATURE is prohibited, and vice versa.
            has_density     = all(self._parser.has_option("INITIAL CONDITIONS", k)
                                for k in ["DENSITY_LEFT", "DENSITY_RIGHT"])
            has_temperature = all(self._parser.has_option("INITIAL CONDITIONS", k)
                                for k in ["TEMPERATURE_LEFT", "TEMPERATURE_RIGHT"])

            if has_density and has_temperature:
                raise ConfigError(
                    f"{self.config_file} [INITIAL CONDITIONS]: DENSITY and TEMPERATURE "
                    f"may not both be specified; provide one or the other."
                )
            if not has_density and not has_temperature:
                raise ConfigError(
                    f"{self.config_file} [INITIAL CONDITIONS]: either DENSITY_LEFT/DENSITY_RIGHT "
                    f"or TEMPERATURE_LEFT/TEMPERATURE_RIGHT must be specified."
                )

        return None

     

        

    # ------------------------------------------------------------
    # config file parsing functions for each section
    # ------------------------------------------------------------

    # [GEOMETRY]
    # ==========
    def expansionDeviceType(self) -> str:
        return self._get_str("GEOMETRY", "EXPANSION_DEVICE_TYPE", inputOptions=["nozzle", "shocktube"])

    def deviceGeometryFilePath(self) -> str:
        return self._get_raw("GEOMETRY", "DEVICE_GEOMETRY_FILE_PATH")



    # [MESH]
    # ======
    def numberOfMeshNodes(self) -> int:
        return self._get_int("MESH", "NUM_MESH_NODES")

    def meshRefinementBool(self) -> bool:
        return self._get_bool("MESH", "MESH_REFINEMENT_BOOL", default=False)

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
    def numericalScheme(self) -> str:
        return self._get_str("NUMERICS", "INTERCELL_FLUX_SCHEME", inputOptions=["godunov", "roe", "roe_arabi", "roe_vinokur"])

    def entropyFixActiveBool(self) -> bool:
        return self._get_bool("NUMERICS", "ENTROPY_FIX_ACTIVE_BOOL")

    def entropyFixCoefficient(self) -> float:
        return self._get_float("NUMERICS", "ENTROPY_FIX_COEFFICIENT", positive=True)

    def MUSCLReconstructionBool(self) -> bool:
        return self._get_bool("NUMERICS", "MUSCL_RECONSTRUCTION_BOOL")

    def MUSCLReconstrFluxLimiter(self) -> str:
        return self._get_str("NUMERICS", "MUSCL_RECONSTR_FLUX_LIMITER", inputOptions=["van albada", "van leer", "min-mod", "superbee", "none"])

    def CFLMax(self) -> float:
            return self._get_float("NUMERICS", "CFL_MAX", positive=True)



    # [BOUNDARY CONDITIONS]
    # =====================
    def boundaryConditions(self) -> tuple[str, str]:
        left = self._get_str("BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_LEFT", inputOptions=["inlet", "outlet", "transparent", "reflective", "periodic"])
        right = self._get_str("BOUNDARY CONDITIONS", "BOUNDARY_CONDITION_RIGHT", inputOptions=["inlet", "outlet", "transparent", "reflective", "periodic"])
        return left, right

    def inletConditionsType(self) -> str:
        value = self._get_str("BOUNDARY CONDITIONS", "INLET_CONDITIONS_TYPE", inputOptions=["total", "static"])
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
        return self._get_str("FLUID", "FLUID_MODEL", inputOptions=["ideal", "real"])

    def fluidGamma(self) -> float:
        return self._get_float("FLUID", "FLUID_GAMMA", positive=True)

    def gasRConstant(self) -> float:
        return self._get_float("FLUID", "GAS_R_CONSTANT", positive=True)

    def fluidLibrary(self) -> str:
        """
        options:
        FluidProp: ['StanMix', 'GasMix', 'PCP-SAFT', 'RefProp', 'qPCP-SAFT', 'HOGC-PCP-SAFT']
        CoolProp: ['CoolProp', 'REFPROP', 'HEOS'] *note that library CoolProp is the same as library HEOS
        HumidAir: ['Humid Air', 'Humid Air Mix']
        LuT: ['LuT']
        Feos: ['feos::HOGC-PCP-SAFT']
        """
        return self._get_raw("FLUID", "FLUID_LIBRARY", inputOptions = [
            "StanMix", "GasMix", "PCP-SAFT", "RefProp", "qPCP-SAFT", "HOGC-PCP-SAFT",
            "CoolProp", "REFPROP", "HEOS",
            "Humid Air", "Humid Air Mix",
            "LuT",
            "feos::HOGC-PCP-SAFT"
        ])

    def propertyExtractionMethod(self) -> str:
        valid = {"fluid", "abstractstate", "abstractstate_v2"}
        value = self._get_str("FLUID", "PROPERTY_EXTRACTION_METHOD", default = "abstractstate_v2")
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
        return self._get_int("OUTPUT", "WRITE_INTERVAL", default=250)

    def printInfoResidualsBool(self) -> bool:
        return self._get_bool("OUTPUT", "PRINT_INFO_RESIDUALS_BOOL", default=True)

    def overwriteResults(self) -> bool:
        return self._get_bool("OUTPUT", "OVERWRITE_RESULTS", default=False)