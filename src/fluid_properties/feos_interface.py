import os
import numpy as np
import base_interface
try:
    import feos.pcsaft as pcsaft
    import feos.eos as eos
    import feos.joback as joback
    from feos.si import MOL, KELVIN, JOULE, METER, KILOGRAM, PASCAL, SECOND, NAV, KB, WATT, SIArray1
    feos_installed = True
    feos_folder = os.path.dirname(os.path.realpath(__file__))

except ModuleNotFoundError:
    feos_installed = False


class FeosFluid(base_interface.Fluid):

    def __init__(self, library, name, arguments):
        if not feos_installed:
            raise ModuleNotFoundError('feos not found')
        super().__init__(library, name)
        self.IsPseudoFluid = False
        if 'is_pseudoFluid' in arguments.keys():
            self.IsPseudoFluid = arguments['is_pseudoFluid']

        if not self.IsPseudoFluid:
            self.get_cmp_cnc()
            self.set_Fluid()

    def set_Fluid(self):

        if self.Library == 'feos::HOGC-PCP-SAFT':
            file_pure = feos_folder + '/feosprop/pcsaft/gc_substances.json'
            file_segments = feos_folder + '/feosprop/pcsaft/groups_homo_gc_pcsaft.json'
            file_joback = feos_folder + '/feosprop/pcsaft/joback1987.json'
            self.JobackParameters = joback.Joback.from_json_segments(self.cmp, pure_path=file_pure, segments_path=file_joback)
            self.Parameters = pcsaft.PcSaftParameters.from_json_segments(self.cmp, pure_path=file_pure, segments_path=file_segments)
            self.AbstractState = FeosAbstractState(self)

        self.set_feos_data()

    def set_feos_data(self):
        self.CncFeos = SIArray1.linspace(0*MOL, 0*MOL, self.nCmp)
        for i in range(self.nCmp):
            self.CncFeos[i] = self.cnc[i]*MOL

    def all_prop(self):
        self.MmolCmp = []
        for i in range(self.nCmp):
            self.MmolCmp.append(self.Parameters.pure_records[i].molarweight)
        self.Mmol = np.dot(self.MmolCmp, self.cnc)/1e3
        self.AbstractState.critical_point()
        self.Tcrit = self.AbstractState.T()
        self.Pcrit = self.AbstractState.p()

    def PropsSI(self, prop, x_str, x, y_str, y):
        
        if any([isinstance(x, int), isinstance(x, float), isinstance(y, int), isinstance(y, float)]):
            x = np.array([x])
            y = np.array([y])
        else:
            x = np.array(x)
            y = np.array(y)
        
        if prop == 'Tcrit' or prop == 'Pcrit' or prop == 'Tmax':
            n = 1
            prop_value = np.zeros(n)
            if prop == 'Tcrit':
                prop_value[0] = self.Tcrit
            elif prop == 'Pcrit':
                prop_value[0] = self.Pcrit
            elif prop == 'Mmol':
                prop_value[0] = self.Mmol()

        else:
            n = len(x)
            prop_value = np.zeros(n)

            tmp_str = [x_str, y_str]
            tmp_values = [x, y]
            tmp_str = [i.upper() for i in tmp_str]
            # sort inputs in alphabetical order to reduce number of cases, and sort values accordingly
            tmp_values = [i for _, i in sorted(zip(tmp_str, tmp_values))]
            tmp_str.sort()
            inputstr = tmp_str[0] + tmp_str[1]

            for i in range(len(x)):

                self.AbstractState.update(inputstr, tmp_values[0][i], tmp_values[1][i])

                if prop == 'P':
                    prop_value[i] = self.AbstractState.p()
                elif prop == 'T':
                    prop_value[i] = self.AbstractState.T()
                elif prop == 'D':
                    prop_value[i] = self.AbstractState.rhomass()
                elif prop == 'H':
                    prop_value[i] = self.AbstractState.hmass()
                elif prop == 'S':
                    prop_value[i] = self.AbstractState.smass()
                elif prop == 'Q':
                    prop_value[i] = self.AbstractState.Q()
                elif prop == 'C':
                    prop_value[i] = self.AbstractState.cpmass()
                elif prop == 'A':
                    prop_value[i] = self.AbstractState.speed_sound()
                elif prop == 'V':
                    prop_value[i] = self.AbstractState.viscosity()
                elif prop == 'K':
                    prop_value[i] = self.AbstractState.conductivity()
                elif prop == 'ST':
                    pass
                elif prop == 'CV':
                    prop_value[i] = self.AbstractState.cvmass()

            if len(prop_value) == 1:
                return prop_value[0]

            return prop_value.tolist()


class FeosAbstractState:

    def __init__(self, feos_fluid, arguments={}):
        self.nCmp = feos_fluid.nCmp
        self.cnc = feos_fluid.cnc
        self.Parameters = feos_fluid.Parameters
        self.JobackParameters = feos_fluid.JobackParameters
        self.CoolPropLanguage = False
        if 'coolprop_language' in arguments.keys():
            self.CoolPropLanguage = arguments['coolprop_language']

        self.MmolCmp = []
        for i in range(self.nCmp):
            self.MmolCmp.append(self.Parameters.pure_records[i].molarweight)
        self.Mmol = np.dot(self.MmolCmp, self.cnc) / 1e3

        self.CncFeos = SIArray1.linspace(0 * MOL, 0 * MOL, self.nCmp)
        for i in range(self.nCmp):
            self.CncFeos[i] = self.cnc[i] * MOL

        self.EoS = eos.EquationOfState.pcsaft(self.Parameters).joback(self.JobackParameters)

    def critical_point(self):
        self.State = eos.State.critical_point(self.EoS, moles=self.CncFeos)

    def update(self, input_spec, input1, input2):

        if input_spec == 'DH':
            self.State = eos.State(self.EoS, density=input1 / self.Mmol * MOL / METER ** 3, molar_enthalpy=input2 * self.Mmol * JOULE / MOL, molefracs=self.cnc)

        elif input_spec == 'DP':
            self.State = eos.State(self.EoS, density=input1 / self.Mmol * MOL / METER ** 3, pressure=input2 * PASCAL, molefracs=self.cnc)

        elif input_spec == 'DQ':
            raise Exception('DQ input currently not supported')

        elif input_spec == 'DS':
            self.State = eos.State(self.EoS, density=input1 / self.Mmol * MOL / METER ** 3, molar_entropy=input2 * self.Mmol * JOULE / (MOL * KELVIN), molefracs=self.cnc)

        elif input_spec == 'DT':
            self.State = eos.State(self.EoS, density=input1 / self.Mmol * MOL / METER ** 3, temperature=input2 * KELVIN, molefracs=self.cnc)

        elif input_spec == 'Ph':
            self.State = eos.State(self.EoS, molar_enthalpy=input1 * self.Mmol * JOULE / MOL,  pressure=input2 * PASCAL, molefracs=self.cnc)

            # Tbub = brentq(PQ_Bubble_Point_Function, 150, 0.95 * self.Tcrit, args=(input2, self.EoS, self.))
            # bubble_point = eos.PhaseEquilibrium.bubble_point(self.EoS, temperature_or_pressure=Tbub * KELVIN,
            #                                                  liquid_molefracs=self.cnc)
            # hbub = bubble_point.liquid.molar_enthalpy()
            # Tdew = brentq(PQ_Dew_Point_Function, 150, 0.95 * self.Tcrit, args=(input2, self.EoS, self.))
            # dew_point = eos.PhaseEquilibrium.dew_point(self.EoS, temperature_or_pressure=Tdew * KELVIN,
            #                                            vapor_molefracs=self.cnc)
            # hdew = dew_point.vapor.molar_enthalpy()
            #
            # if input1 < hbub / (self.Mmol * JOULE / MOL):
            #
            #     Tinit = np.array([Tbub])
            #
            #
            # elif input1 > hdew / (self.Mmol * JOULE / MOL):
            #
            #     Tinit = np.array([Tdew])
            #
            #
            # else:
            #
            #     Tinit = np.linspace(Tbub, Tdew, 50, endpoint=True)
            #
            # count = 0
            # for i in range(len(Tinit)):
            #
            #     try:
            #
            #         self.State = eos.State(self.EoS, molar_enthalpy=input1 * self.Mmol * JOULE / MOL,
            #                                pressure=input2 * PASCAL, molefracs=self.cnc,
            #                                initial_temperature=Tinit[i] * KELVIN)
            #         break
            #     except:
            #         count += 1
            #
            #         if i == len(Tinit) - 1:
            #             raise Exception('PH calculation did not converge')
        elif input_spec == 'HQ':
            raise Exception('HQ input currently not supported')

        elif input_spec == 'HS':
            self.State = eos.State(self.EoS, molar_enthalpy=input1 * self.Mmol * JOULE / MOL, pressure=input2 * PASCAL, molefracs=self.cnc)

        elif input_spec == 'HT':
            self.State = eos.State(self.EoS, molar_enthalpy=input1 * self.Mmol * JOULE / MOL, temperature=input2 * KELVIN, molefracs=self.cnc)

        elif input_spec == 'PQ':
            if input2 == 0.0 or input2 == 0:
                bubble_point = eos.PhaseEquilibrium.bubble_point(self.EoS, temperature_or_pressure=input2 * PASCAL, liquid_molefracs=self.cnc)
                self.State = bubble_point.liquid

            elif input2 == 1.0 or input2 == 1:
                dew_point = eos.PhaseEquilibrium.dew_point(self.EoS, temperature_or_pressure=input2 * PASCAL, vapor_molefracs=self.cnc)
                self.State = dew_point.vapor

            else:
                raise Exception('Q inputs between 0 and 1 currently not supported')

        elif input_spec == 'PS':
            self.State = eos.State(self.EoS, pressure=input1 * PASCAL, molar_entropy=input2 * self.Mmol * JOULE / (MOL * KELVIN), molefracs=self.cnc)

        elif input_spec == 'PT':
            self.State = eos.State(self.EoS, pressure=input1 * PASCAL, temperature=input2 * KELVIN, molefracs=self.cnc)

        elif input_spec == 'QS':
            raise Exception('QS input currently not supported')

        elif input_spec == 'QT':
            if input1 == 0.0 or input1 == 0:
                bubble_point = eos.PhaseEquilibrium.bubble_point(self.EoS, temperature_or_pressure=input2 * KELVIN, liquid_molefracs=self.cnc)
                self.State = bubble_point.liquid

            elif input1 == 1.0 or input1 == 1:
                dew_point = eos.PhaseEquilibrium.dew_point(self.EoS, temperature_or_pressure=input2 * KELVIN, vapor_molefracs=self.cnc)
                self.State = dew_point.vapor

        elif input_spec == 'ST':
            self.State = eos.State(self.EoS, molar_entropy=input1 * self.Mmol * JOULE / (MOL * KELVIN), temperature=input2 * KELVIN, molefracs=self.cnc)

    def molar_mass(self):
        return self.FLuid.Mmol

    def p(self):
        return self.State.pressure() / PASCAL

    def T(self):
        return self.State.temperature / KELVIN

    def rhomass(self):
        return self.State.mass_density() / (KILOGRAM / METER ** 3)

    def hmass(self):
        return self.State.specific_enthalpy() / (JOULE / KILOGRAM)

    def smass(self):
        return self.State.specific_entropy() / (JOULE / (KILOGRAM * KELVIN))

    # def Q(self):
    # 
    #     if self.T() > self.Tcrit:
    #         return 1.0
    # 
    #     Tbub = brentq(PQ_Bubble_Point_Function, 150, 0.95 * self.Tcrit,
    #                   args=(self.State.pressure() / PASCAL, self.EoS, self.fluid))
    #     bubble_point = EoS.PhaseEquilibrium.bubble_point(self.EoS, temperature_or_pressure=Tbub * KELVIN,
    #                                                      liquid_molefracs=self.fluid.cnc)
    #     Tdew = brentq(PQ_Dew_Point_Function, 150, 0.95 * self.Tcrit,
    #                   args=(self.State.pressure() / PASCAL, self.EoS, self.fluid))
    #     dew_point = EoS.PhaseEquilibrium.dew_point(self.EoS, temperature_or_pressure=Tdew * KELVIN,
    #                                                vapor_molefracs=self.fluid.cnc)
    # 
    #     hbub = bubble_point.liquid.specific_enthalpy()
    #     hdew = dew_point.vapor.specific_enthalpy()
    # 
    #     Q = (self.State.specific_enthalpy() - hbub) / (hdew - hbub)
    # 
    #     if Q > 1:
    # 
    #         return 1.0
    # 
    #     elif Q < 0:
    # 
    #         return 0.0
    # 
    #     else:
    # 
    #         return Q

    def cvmass(self):
        return self.State.molar_isochoric_heat_capacity() / self.Mmol / (JOULE / (MOL * KELVIN))

    def cpmass(self):
        return self.State.molar_isobaric_heat_capacity() / self.Mmol / (JOULE / (MOL * KELVIN))

    def speed_sound(self):
        return self.State.speed_of_sound() / (METER / SECOND)

    # def fundamental_derivative_of_gas_dynamics(self):

    #     return self.State.

    def viscosity(self):
        return self.State.viscosity() / (PASCAL * SECOND)

    def conductivity(self):

        if self.nCmp == 1:
            conductivity = self.State.thermal_conductivity()
        else:
            raise NotImplementedError('Mixture thermal conductivity not implemented')

        return conductivity / (WATT / (METER * KELVIN))
