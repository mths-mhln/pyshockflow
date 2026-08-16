# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:40:47 2024

@author: lgalieti
"""
import numpy as np 
from . import fluid_properties as FP
from .group_contribution_methods.molecular_structure import MolecularStructure


def declare_fluid(n_cmp, concentrations,
                  Cocc, CHocc, CH2occ,
                  molecule_class, n_branches, single_group, single_group_occ, complete_octet = True, CH3occ = None,
                  ):
    """
    NBranch sets groups 'C_arom', 'CH_hex', 'CH_pent'
    single_groups sets groups 'CH=0', ')C=O', 'OCH2', 'OCH3', 'COO', 'OH', 'NH2', and each combination of '=CH2', '=CH', '=C('
    'CH3' completes octet rule
    'CH_arom, CH2_pent, CH2_hex close the ring'
    'Remaining are ")C(", ")CH", "CH2",

    MoleculeType + single_group = 2 choice variables x component
    ")C(", ")CH", "CH2" + single_group_occ + n_branches = 5 Continuous variables x component
    """

    if n_cmp > 1:  
        concentrations = np.append(concentrations, 1 - np.sum(concentrations))
    else:
        concentrations = [1]

    """
        [")C(", ")CH", "CH2" , single_group, ['C_arom', 'CH_hex', 'CH_pent'], [CH_arom, CH2_pent, CH2_hex],'CH3']
    """
    groups = np.full((n_cmp, 8), 'None', dtype="U10")
    groups_occ = np.full((n_cmp, 8), 0.0)
    for i in range(n_cmp):

        groups[i, 0] = ')C('
        groups_occ[i, 0] = Cocc[i]
        groups[i, 1] = ')CH'
        groups_occ[i, 1] = CHocc[i]
        groups[i, 2] = 'CH2'
        groups_occ[i, 2] = CH2occ[i]

        groups[i, 3] = 'CH3'
        if CH3occ is not None:
            groups_occ[i, 3] = CH3occ[i]

        groups[i, 4] = 'C_arom'
        groups[i, 5] = 'CH_arom'

        if molecule_class[i] in ['aromatic', 'cyclohexane', 'cyclopentane']:
            groups_occ[i, 4] = n_branches[i]
            if molecule_class[i] == 'aromatic':
                groups_occ[i, 5] = 6 - n_branches[i]
            elif molecule_class[i] == 'cyclohexane':
                groups[i, 4] = 'CH_hex'
                groups[i, 5] = 'CH2_hex'
                groups_occ[i, 5] = 6 - n_branches[i]
            elif molecule_class[i] == 'cyclopentane':
                groups[i, 4] = 'CH_pent'
                groups[i, 5] = 'CH2_pent'
                groups_occ[i, 5] = 5 - n_branches[i]

        if single_group[i] == 'HC=CH':
            groups[i, 6] = '=CH'
            groups_occ[i, 6] = 2*single_group_occ[i]

        elif single_group[i] == '2HC=CH2':
            groups[i, 6] = '=CH2'
            groups_occ[i, 6] = 2*single_group_occ[i]

        elif single_group[i] == ')C=C(':
            groups[i, 6] = '=C('
            groups_occ[i, 6] = 2*single_group_occ[i]

        elif single_group[i] == 'HC=CH2':
            groups[i, 6] = '=CH'
            groups_occ[i, 6] = single_group_occ[i]
            groups[i, 7] = '=CH2'
            groups_occ[i, 7] = single_group_occ[i]

        elif single_group[i] == 'HC=C(':
            groups[i, 6] = '=CH'
            groups_occ[i, 6] = single_group_occ[i]
            groups[i, 7] = '=C('
            groups_occ[i, 7] = single_group_occ[i]

        elif single_group[i] == '2HC=C(':
            groups[i, 6] = '=CH2'
            groups_occ[i, 6] = single_group_occ[i]
            groups[i, 7] = '=C('
            groups_occ[i, 7] = single_group_occ[i]

        else:
            groups[i, 6] = single_group[i]
            if single_group[i] != 'None':
                groups_occ[i, 6] = single_group_occ[i]

    molecular_structure = MolecularStructure(molecule_class, groups, groups_occ)
    # print(molecular_structure.Groups, molecular_structure.GroupOccurrences)
    if complete_octet:
        molecular_structure.complete_open_ends(['CH3']*n_cmp)
    fluid = FP.fluid('HOGC-PCP-SAFT', 'pseudofluid', is_pseudofluid=True)
    # print(molecular_structure.Groups, molecular_structure.GroupOccurrences)
    fluid.set_pseudofluid_groups(n_cmp, concentrations, molecular_structure.Groups, molecular_structure.GroupOccurrences)
    fluid.all_prop()
    fluid.all_prop()

    return fluid, molecular_structure


def check_fluid(fluid, T_min):
    
    fluid.Tcrit = FP.PropsSI('Tcrit', [], [], [], [], fluid)
    fluid.Pcrit = FP.PropsSI('Pcrit', [], [], [], [], fluid)
    P_sat = FP.PropsSI('P', 'T', T_min, 'Q', 0.0, fluid)
    
    if P_sat < 0:
        raise Exception('Fluid P sat check failed')
    
    h_in = FP.PropsSI('H', 'P', P_sat, 'Q', 0.0, fluid) 
    
    if h_in == -8888.8:
        raise Exception('Fluid enthalpy check failed')
    
    h_out = FP.PropsSI('H', 'P', P_sat, 'Q', 1.0, fluid) 
    
    if h_out == -8888.8:
        raise Exception('Fluid  enthalpy check failed')
    
    fluid.sigma = FP.PropsSI('sigma', [], [], [], [], fluid)
    fluid.Mmol = FP.PropsSI('Mmol', [], [], [], [], fluid)
    h = np.linspace(h_in, h_out, 10, endpoint=True)

    T_p = np.zeros(10)
    
    for i in range(0, 10):
        T_p[i] = FP.PropsSI('T', 'P', P_sat, 'H', h[i], fluid)       
 
    for T in T_p: 
        if T < 0: 
            raise Exception('Fluid temperature profile check failed')
            
    delta_T = np.diff(T_p)
    for delta in delta_T:
        if round(delta, 6) < 0:
            raise Exception('Fluid temperature profile check failed')
            

def check_spiky_temperature_profile(T_p, tol=0.5):
    if min(T_p) < 0:
        raise Exception('Spiky temperature profile check failed')
    delta_T = np.ediff1d(T_p)
    if min(delta_T + tol) < 0:  # here we allow a small "spike" because when the fluid is pure and there's a pressure drop, the temperature decreases
        raise Exception('Spiky temperature profile check failed')


def check_pinch_computation(T_p_H, T_p_C, DTpp):
    # sometimes the pinch computations do not converge leading to garbage profiles. Here we check it
    if min(T_p_H - T_p_C) - DTpp + 1e-2 < 0:
        raise Exception('Pinch computation check failed')  
 
        
def check_entropy_variation(HEX):
    delta_s = HEX.Entropy_Variation()
    if delta_s < 0:
        raise Exception('entropy variation check failed') 


def additional_miscellanea_checks(ORC):
    
    constraints = []
    # Check if the efficiency is higher than the Lorenz cycle one. ( this avoids very weird pseudofluids)
    constraints.append(ORC.Normalized_Efficiency_Lo() - 1)

    # Check vapor quality at condenser inlet ( this avoids very weird pseudofluids)
    constraints.append(-1*(FP.PropsSI('Q',  'P', ORC.condenser.P_in_H, 'H', ORC.condenser.h_in_H, ORC.condenser.fluid_H) - 0.0001))

    # Check temperature at condenser inlet ( this avoids very weird pseudofluids) this -1 is some small tolerance. Should not be needed
    constraints.append(ORC.condenser.T_in_H - ORC.regenerator.T_out_H - 1)
    constraints.append(-1*(FP.PropsSI('Q',  'P', ORC.condenser.P_in_H, 'H', ORC.condenser.h_in_H, ORC.condenser.fluid_H) - 0.0001))
    constraints.append(FP.PropsSI('Q',  'P', ORC.condenser.P_out_H, 'H', ORC.condenser.h_out_H, ORC.condenser.fluid_H) - 0.0001)

    # give a large number to the objective function if some constrain is violated. This means the net power is actually minimal
    check = np.array(constraints) > 0
    if True in check:
        raise Exception('miscellanea check failed')


def check_if_same_molecular_structure(molstruct1, molstruct2):

    if molstruct1.NCmp != molstruct2.NCmp:
        return -1

    found_same = []
    for i in range(molstruct1.NCmp):
        comp1groups = molstruct1.Groups[i, :]
        comp1groupocc = molstruct1.GroupOccurrences[i, :]

        found_same_cmp = False

        for j in range(molstruct2.NCmp):

            comp2groups = molstruct2.Groups[j, :]
            comp2groupocc = molstruct2.GroupOccurrences[j, :]

            groups_not_matching = False
            for group in comp1groups:
                if group not in comp2groups and group != 'None' and comp1groupocc[np.where(group == comp1groups)] > 0:
                    groups_not_matching = True
                    break
            for group in comp2groups:
                if group not in comp1groups and group != 'None' and comp2groupocc[np.where(group == comp2groups)] > 0:
                    groups_not_matching = True
                    break

            if groups_not_matching:
                continue

            groups_occ_not_matching = False
            for group in comp1groups:
                if group != 'None':
                    occ1 = comp1groupocc[np.where(group == comp1groups)]
                    occ2 = comp2groupocc[np.where(group == comp2groups)]
                    if occ1 != occ2:
                        groups_occ_not_matching = True
                        break

            if groups_occ_not_matching:
                continue

            found_same_cmp = True
            break

        found_same.append(found_same_cmp)

    if all(found_same):
        return 1
    else:
        return -1



