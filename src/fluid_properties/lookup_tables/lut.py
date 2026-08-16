import numpy as np
from scipy.optimize import brentq
from functools import lru_cache
from .grids import Grids
import json


PropInput = ['h', 'P', 'd', 'T', 'q', 's']
FunctionInput = ['hmass', 'p', 'rhomass', 'T', 'Q', 'smass']


class TableInterpolationError(Exception):
    pass


class LookUpTable:

    def __init__(self, abstractstate, arguments):

        import_lut = False
        node_values_file_name = None
        if 'import_lut' in arguments.keys():
            if arguments['import_lut']:
                import_lut = True
                node_values_file_name = arguments['node_values_file_name']
                with open(arguments['lut_data_file_name']) as file:
                    lut_data = json.load(file)

                input0 = lut_data[0]
                input1 = lut_data[1]
                x0_array = lut_data[2]
                x1_array = lut_data[3]
                variables = lut_data[4]

        if not import_lut:
            input0 = arguments['input_spec_0']
            input1 = arguments['input_spec_1']
            x0_array = arguments['input_array_0']
            x1_array = arguments['input_array_1']
            variables = arguments['variables']

        self.FallbackToEoS = False
        if 'fallback_to_eos' in arguments.keys():
            if arguments['fallback_to_eos']:
                self.FallbackToEoS = True
                self.EoS = abstractstate
                self.StashedInput0 = None
                self.StashedInput1 = None

        if 'kx' in arguments.keys():
            kx = arguments['kx']
        else:
            kx = 3

        if 'ky' in arguments.keys():
            ky = arguments['ky']
        else:
            ky = 3


        if input0 in PropInput:
            if FunctionInput[PropInput.index(input0)] not in variables:
                variables.append(FunctionInput[PropInput.index(input0)])
        else:
            raise Exception('input0 not recognized')

        if input1 in PropInput:
            if FunctionInput[PropInput.index(input1)] not in variables:
                variables.append(FunctionInput[PropInput.index(input1)])
        else:
            raise Exception('input1 not recognized')

        self.X0Node = np.array(x0_array)
        self.X1Node = np.array(x1_array)
        self.InputSpec = input0 + input1
        self.Grids = Grids(len(x0_array), len(x1_array), variables)

        self.CoolPropLanguage = False
        if 'coolprop_language' in arguments.keys():
            self.CoolPropLanguage = arguments['coolprop_language']


        self.Pcrit = abstractstate.p_critical()
        self.Tcrit = abstractstate.T_critical()
        self.Mmol = abstractstate.molar_mass()


        self.Grids.set_values(x0_array, x1_array, self.InputSpec, abstractstate, import_lut, node_values_file_name, kx, ky)
        self.LuTData = [None]*5
        self.LuTData[0] = input0
        self.LuTData[1] = input1
        self.LuTData[2] = list(x0_array)
        self.LuTData[3] = list(x1_array)
        self.LuTData[4] = variables

    def export_lut_data(self, filename):
        with open(filename + '.json', "w") as file:
            json.dump(self.LuTData, file)

    def update(self, inputspec, input0, input1):

        self.Input0 = None
        self.Input1 = None
        self.IsUpdateSuccessful = True
        self.FallbackInputSpec = self.InputSpec

        if len(inputspec) > 2:
            raise Exception('inputspec is too long (max 2 characters')
        if not isinstance(inputspec, str):
            raise Exception('inputspec should be a string')

        if inputspec[0] == self.InputSpec[1] or inputspec[1] == self.InputSpec[0]:
            # flip
            inputspec = inputspec[::-1]
            tmp = input0
            input0 = input1
            input1 = tmp

        if inputspec[0] == self.InputSpec[0] and inputspec[1] == self.InputSpec[1]:
            if not self.Grids.MinX0 <= input0 <= self.Grids.MaxX0 or not self.Grids.MinX1 <= input1 <= self.Grids.MaxX1:
                if not self.FallbackToEoS:
                    raise TableInterpolationError('LuT interpolation out of bonds')
            self.Input0 = input0
            self.Input1 = input1
            return
        # if some input spec is really different
        try:
            if inputspec[0] != self.InputSpec[0] and inputspec[1] == self.InputSpec[1]:
                self.loop_table0(inputspec, input0, input1)

            elif inputspec[0] == self.InputSpec[0] and inputspec[1] != self.InputSpec[1]:
                self.loop_table1(inputspec, input0, input1)

            else:
                self.loop_table2(inputspec, input0, input1)

        except:
            self.IsUpdateSuccessful = False
            if self.FallbackToEoS:
                self.Input0 = input0
                self.Input1 = input1
                self.FallbackInputSpec = inputspec

    def loop_table0(self, inputspec, input0, input1):

        self.Input1 = input1
        target_variable = FunctionInput[PropInput.index(inputspec[0])]
        target_variable_value = input0

        if abs(self.loop_table0_function(self.X0Node[0], input1, target_variable,  target_variable_value) / target_variable_value) <= 1e-5:
            self.Input0 = self.X0Node[0]
            return
        if abs(self.loop_table0_function(self.X0Node[-1], input1, target_variable, target_variable_value) / target_variable_value) <= 1e-5:
            self.Input0 = self.X0Node[-1]
            return

        self.Input0 = brentq(self.loop_table0_function, self.X0Node[0], self.X0Node[-1], args=(input1, target_variable, target_variable_value))

    def loop_table0_function(self, input0, input1, target_variable, target_variable_value):
        return self.interpolate(input0, input1, target_variable) - target_variable_value

    def loop_table1(self, inputspec, input0, input1):

        self.Input0 = input0
        target_variable = FunctionInput[PropInput.index(inputspec[1])]
        target_variable_value = input1

        if abs(self.loop_table1_function(self.X1Node[0], input0, target_variable, target_variable_value) / target_variable_value) <= 1e-5:
            self.Input1 = self.X1Node[0]
            return
        if abs(self.loop_table1_function(self.X1Node[-1], input0, target_variable, target_variable_value) / target_variable_value) <= 1e-5:
            self.Input1 = self.X1Node[-1]
            return

        self.Input1 = brentq(self.loop_table1_function, self.X1Node[0], self.X1Node[-1], args=(input0, target_variable, target_variable_value))

    def loop_table1_function(self, input1, input0, target_variable, target_variable_value):
        return self.interpolate(input0, input1, target_variable) - target_variable_value

    def loop_table2(self, inputspec, input0, input1):

        target_variables = [FunctionInput[PropInput.index(inputspec[0])],
                            FunctionInput[PropInput.index(inputspec[1])]]
        target_variables_value = [input0, input1]

        resolution = [3, 10, 30]
        for res in resolution:
            err = np.ndarray([res])
            input0_table_try = np.linspace(self.X0Node[0], self.X0Node[-1], res)
            for i in range(0, res):
                input0_table = input0_table_try[i]
                err[i] = self.loop_table2_function(input0_table, target_variables, target_variables_value)
                if abs(err[i]/target_variables_value[0]) <= 1e-6:
                    self.Input0 = input0_table
                    return
                if i > 0:
                    if err[i-1]*err[i] < 0:
                        self.Input0 = brentq(self.loop_table2_function, input0_table_try[i], input0_table_try[i-1], args=(target_variables, target_variables_value))
                        return

        self.IsUpdateSuccessful = False

    def loop_table2_function(self, input0, target_variables, target_variables_value):

        if abs(self.loop_table1_function(self.X1Node[0], input0, target_variables[1], target_variables_value[1]) / target_variables_value[1]) <= 1e-6:
            self.Input1 = self.X1Node[0]
            return
        if abs(self.loop_table1_function(self.X1Node[-1], input0, target_variables[1], target_variables_value[1]) / target_variables_value[1]) <= 1e-6:
            self.Input1 = self.X1Node[-1]
            return

        self.Input1 = brentq(self.loop_table1_function, self.X1Node[0], self.X1Node[-1], args=(input0, target_variables[1], target_variables_value[1]))

        return self.loop_table0_function(input0, self.Input1, target_variables[0], target_variables_value[0])

    @lru_cache(maxsize=30)
    def interpolate(self, input0, input1, variable):
        outofbounds = False
        if not self.Grids.MinX0 <= input0 <= self.Grids.MaxX0 or not self.Grids.MinX1 <= input1 <= self.Grids.MaxX1:
            outofbounds = True

        if self.IsUpdateSuccessful and not outofbounds:
            try:
                return getattr(self.Grids, variable)(input0, input1)[0, 0]
            except:
                self.IsUpdateSuccessful = False

        if not self.IsUpdateSuccessful or outofbounds:
            if self.FallbackToEoS:
                try:
                    self.update_eos(input0, input1)
                    return getattr(self.EoS, variable)()
                except:
                    raise TableInterpolationError('LuT interpolation out of bonds or unsuccessful. Fall back to EoS failed')
            else:
                raise TableInterpolationError('LuT interpolation out of bonds or unsuccessful')

    def update_eos(self, input0, input1):
        if any([input0 != self.StashedInput0, input1 != self.StashedInput1]):
            self.EoS.update(self.FallbackInputSpec, input0, input1)
            self.StashedInput0 = input0
            self.StashedInput1 = input1

    def p_critical(self):
        return self.Pcrit

    def T_critical(self):
        return self.Tcrit

    def molar_mass(self):
        return self.Mmol

    def p(self):
        return self.interpolate(self.Input0, self.Input1, 'p')

    def T(self):
        return self.interpolate(self.Input0, self.Input1, 'T')

    def rhomass(self):
        return self.interpolate(self.Input0, self.Input1, 'rhomass')

    def hmass(self):
        return self.interpolate(self.Input0, self.Input1, 'hmass')

    def smass(self):
        return self.interpolate(self.Input0, self.Input1, 'smass')

    def Q(self):
        q = self.interpolate(self.Input0, self.Input1, 'Q')
        # accounting for small oscillation in the spline.
        if 1.0 <= q <= 1.01:
            return 1.0
        elif 0 >= q >= -0.01:
            return 0.0
        else:
            return q

    def cvmass(self):
        return self.interpolate(self.Input0, self.Input1, 'cvmass')

    def cpmass(self):
        return self.interpolate(self.Input0, self.Input1, 'cpmass')

    def cp0mass(self):
        return self.interpolate(self.Input0, self.Input1, 'cp0mass')

    def speed_sound(self):
        return self.interpolate(self.Input0, self.Input1, 'speed_sound')

    def fundamental_derivative_of_gas_dynamics(self):
        return self.interpolate(self.Input0, self.Input1, 'fundamental_derivative_of_gas_dynamics')

    def viscosity(self):
        return self.interpolate(self.Input0, self.Input1, 'viscosity')

    def conductivity(self):
        return self.interpolate(self.Input0, self.Input1, 'conductivity')

    def compressibility_factor(self):
        return self.interpolate(self.Input0, self.Input1, 'compressibility_factor')

    def first_partial_deriv(self, of, wrt, const):
        if of == 'rhomass' and wrt == 'P' and const == 'T':
            return self.interpolate(self.Input0, self.Input1, 'drhomassdPcT')
