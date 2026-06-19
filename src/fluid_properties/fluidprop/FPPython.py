#=================================================================================================#
#                                                                                                 #
#  FluidProp 3        Teus van der Stelt        Copyright (c) 2018-2030. All Rights Reserved.     #
#  ===========                                  Asimptote/iTernergy, Heeswijk-Dinther             #
#                                                                                                 #
#-------------------------------------------------------------------------------------------------#
#                                                                                                 #   
#  Module name   : FluidProp.py                                                                   #   
#                                                                                                 #   
#  Description   : Python library for FluidProp.                                                  #   
#                                                                                                 #   
#  Build         : 3                                                                              #   
#  Release       : 1.0.3                                                                          #   
#                                                                                                 #   
#=================================================================================================#


from __future__ import print_function
from   ctypes   import byref, cdll, c_double, c_int, c_float, c_void_p
from sys import platform
import os
import numpy as np

FPP = [None]
ErrorMsg = [None]

#-------------------------------------------------------------------------------------------------#
def Init_FluidProp():
    
    # The library (.dll for windows, .so for linux, .dylib for macOS) is loaded here.
    if platform.startswith('linux'):
       FPP[0] = cdll.LoadLibrary(os.environ["FLUIDPROP"]+'/lib/FPPython.so')
    elif platform == 'darwin':
       FPP[0] = cdll.LoadLibrary('/Applications/FluidProp/lib/FPPython.dylib')
    elif platform == 'win32':
       FPP[0] = cdll.LoadLibrary('C:/Program Files/FluidProp/FPPython.dll')
    #    FPP[0] = cdll.LoadLibrary('C:/fluidprop/Sources/Libraries/API/Python/Windows/bin/x64/FPPython.dll')
    elif platform == 'win64':
       # FPP[0] = cdll.LoadLibrary('C:/Program Files/FluidProp/FPPython.dll')
       FPP[0] = cdll.LoadLibrary('C:/fluidprop/Sources/Libraries/API/Python/Windows/bin/x64/FPPython.dll')
    else:
       ErrorMsg[0] = "FluidProp Python error: unknown platform..."
  
    # Before you can use FluidProp in Windows you must create and initialize a FluidProp
    # COM object with the Init_FluidProp call. This step is only necessary in Windows 
    # and can be skipped for Linux and macOS.
    
    if platform.startswith('win'): 
        
        FPP[0].init_fluidprop()

#-------------------------------------------------------------------------------------------------#
def ReleaseObjects() :

    # At the end of program, you should clean up your memory by deleting the FluidProp 
    # objects created in the beginning of the program with the Init_FluidProp call and 
    # the FluidProp child objects created with the SetFluid or CreateObject calls. This 
    # step is only necessary in Windows and can be skipped for Linux and macOS.
    if platform.startswith('win'): FPP[0].releaseobjects()

#-------------------------------------------------------------------------------------------------#
def CreateObject( SubLib) : 

    b_sublib = SubLib.encode()
    l_sublib = c_int(len(SubLib))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].createobject( b_sublib, b_ErrorMsg, byref(l_sublib), byref(l_errmsg))

    ErrorMsg[0] = b_ErrorMsg.decode()

#-------------------------------------------------------------------------------------------------#
def GetErrorMsg() :
    return ErrorMsg[0].rstrip()

#-------------------------------------------------------------------------------------------------#
def FluidProp_Version( SubLib) :
 
    # Returns the version of FluidProp or of one its sub-libraries.
    # -------------------------------------------------------------

    b_sublib = SubLib.encode()
    l_sublib = c_int(len(SubLib))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '
    
    version = np.array([0, 0, 0, 0], dtype='i')

    FPP[0].version( b_sublib, version.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_sublib), byref(l_errmsg))
    
    ErrorMsg[0] = b_ErrorMsg.decode()

    return [ int(version[0]), int(version[1]), int(version[2]), int(version[3]) ] 

#-------------------------------------------------------------------------------------------------#
def GetFluidNames( LongShortCAS, SubLib) :

    # Returns a list of the available fluids in sub-library SubLib.

    # With LongShortCAS you can specify the way the fluid names are 
    # returned: "l(ong)","s(hort)", or "CAS" (= by CAS number).
    # -------------------------------------------------------------

    b_SubLib = SubLib.encode()
    l_SubLib = c_int(len(SubLib))

    b_LongShortCAS = LongShortCAS.encode()
    l_LongShortCAS = c_int(len(LongShortCAS))

    nFluids = c_int(0)

    FluidNames = np.empty( (1000, 256), dtype='c' )
    l_FluidNames = c_int(len(FluidNames))                   # dit is het aantal elementen....

    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '
     
    FPP[0].getfluidnames( b_LongShortCAS, b_SubLib, byref(nFluids), FluidNames.ctypes.data_as(c_void_p), b_ErrorMsg, 
                          byref(l_LongShortCAS), byref(l_SubLib), byref(l_FluidNames), byref(l_errmsg))

    FluidSet = list()
    i = 0
    while i < nFluids.value:
        FluidSet.append( FluidNames[i].tostring().decode())
        i += 1

    ErrorMsg[0] = b_ErrorMsg.decode()

    return FluidSet

#-------------------------------------------------------------------------------------------------#
def SetFluid( SubLib, Cmp = " ", Cnc = [ 1.0 ]) :
    

    b_sublib = SubLib.encode()
    l_sublib = c_int(len(SubLib))

    if SubLib == "StanMix" or SubLib == "freeStanMix":
       nCmp = len(Cnc)
    else:
       nCmp = len(Cmp)

    # Select component with longest name   
    lcmp = len(Cmp[0])
    if nCmp > 1 and SubLib != "StanMix" and SubLib != "freeStanMix":
       i = 1
       while i < nCmp:
          lcmp = max( lcmp, len(Cmp[i]))
          i += 1

    comp = np.empty((nCmp, lcmp), dtype='c')
    conc = np.asarray(Cnc)

    if nCmp == 1:
       comp[0] = Cmp[0]
       conc[0] = 1.0
    else:
       i = 0
       while i < nCmp:
          if SubLib == "StanMix" or SubLib == "freeStanMix":
             comp[0] = Cmp[0]
          else:
             # Make all component names same length by adding spaces
             comp[i] = Cmp[i] + (lcmp - len(Cmp[i])) * " " 
          i += 1
             
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].setfluid( b_sublib, byref(c_int(nCmp)), comp.ctypes.data_as(c_void_p), 
                     conc.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_sublib),
                     byref(c_int(lcmp)), byref(l_errmsg))  
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
def SetPseudoFluid(SubLib, nCmp, Cnc, sgmN, sigma, epsilonK, dipole_moment, quadrupole_moment, is_associating , association_potential_depth ):
    
    b_sublib  = SubLib.encode()
    l_sublib  = c_int(len(SubLib))
    conc      = np.asarray(Cnc)
    msg       = np.asarray(sgmN)
    sgm       = np.asarray(sigma)
    epsk      = np.asarray(epsilonK)
    dpl_mo    = np.asarray(dipole_moment)
    qdpl_mo   = np.asarray(quadrupole_moment)
    is_asc    = np.asarray(is_associating)
    asc_pot_d = np.asarray(association_potential_depth)
        
    l_errmsg = c_int(80)
    b_ErrorMsg   = b'No errors                                                                       '
      
    FPP[0].setfluidparameters(b_sublib, byref(c_int(nCmp)), 
                              conc.ctypes.data_as(c_void_p),   msg.ctypes.data_as(c_void_p), 
                              sgm.ctypes.data_as(c_void_p),    epsk.ctypes.data_as(c_void_p),
                              is_asc.ctypes.data_as(c_void_p), asc_pot_d.ctypes.data_as(c_void_p),
                              dpl_mo.ctypes.data_as(c_void_p), qdpl_mo.ctypes.data_as(c_void_p),
                              b_ErrorMsg,                      byref(l_sublib), byref(l_errmsg))
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
def SetGroups_PseudoFluid( nCmp, Cnc, Groupnames, Occurrences)   :
    
    Groupnames = np.char.ljust(Groupnames.astype('S32'), 32)
    nGroup = np.size(Groupnames, 1)
    Cnc_np           = np.asfortranarray(np.asarray(Cnc))
    Groupnames_np    = np.asfortranarray(np.asarray(Groupnames))
    Occurrences_np   = np.asfortranarray(np.asarray(Occurrences))

    FPP[0].setfluidgroups(byref(c_int(nCmp)), 
                          Cnc_np.ctypes.data_as(c_void_p) ,
                          byref(c_int(nGroup)), 
                          Groupnames_np.ctypes.data_as(c_void_p), 
                          Occurrences_np.ctypes.data_as(c_void_p),
                          byref(c_int(32)) )
    
#-------------------------------------------------------------------------------------------------#
def SetUnits( UnitSet, MassOrMole, PropSymbol, PropUnit) :

    # Definition of units or a units set (SI, Anglo-Sakson).
    # ------------------------------------------------------

    b_UnitSet    = UnitSet.encode()
    l_UnitSet    = c_int(len(UnitSet))
    b_MassOrMole = MassOrMole.encode()
    l_MassOrMole = c_int(len(MassOrMole))
    b_PropSymbol = PropSymbol.encode()
    l_PropSymbol = c_int(len(PropSymbol))
    b_PropUnit   = PropUnit.encode()
    l_PropUnit   = c_int(len(PropUnit))

    l_errmsg     = c_int(80)
    b_ErrorMsg   = b'No errors                                                                       '

    FPP[0].setunits( b_UnitSet, b_MassOrMole, b_PropSymbol, b_PropUnit, b_ErrorMsg, byref(l_UnitSet), 
                     byref(l_MassOrMole), byref(l_PropSymbol), byref(l_PropUnit), byref(l_errmsg))        

    ErrorMsg[0] = b_ErrorMsg.decode()

#-------------------------------------------------------------------------------------------------#
def SetRefState( Tref, Pref) :

    # Definition of a reference state for non-dimensional properties.
    # ---------------------------------------------------------------

    l_errmsg     = c_int(80)
    b_ErrorMsg   = b'No errors                                                                       '

    FPP[0].setrefstate( byref(c_double(Tref)), byref(c_double(Pref)), b_ErrorMsg, byref(l_errmsg))        

    ErrorMsg[0] = b_ErrorMsg.decode()

#-------------------------------------------------------------------------------------------------#
def CalcProp( PropSpec, InputSpec, Input1, Input2) :

    # Calculate property defined by PropSpec [user specified unit].
    # -------------------------------------------------------------

    b_PropSpec  = PropSpec.encode()
    l_propspec  = c_int(len(PropSpec))
    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].calcprop.restype = c_double

      
    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].calcprop( b_PropSpec, b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)), 
                              b_ErrorMsg, byref(l_propspec), byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].calcprop( b_PropSpec, b_InputSpec, byref(c_double(Input1[i])), 
                                     byref(c_double(Input2[i])), b_ErrorMsg, 
                                     byref(l_propspec), byref(l_inputspec), byref(l_errmsg))
                                     
    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].calcprop( b_PropSpec, b_InputSpec, byref(c_double(Input1[i])), 
                                     byref(c_double(Input2)), b_ErrorMsg, 
                                     byref(l_propspec), byref(l_inputspec), byref(l_errmsg))
    else:
       out = -8888.8
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])

    ErrorMsg[0] = b_ErrorMsg.decode()

    return float(out)

#-------------------------------------------------------------------------------------------------#
def Pressure( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].pressure.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].pressure( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                              b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].pressure( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2[i])),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].pressure( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2)),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       out = -8888.8
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Temperature( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].temperature.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].temperature( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                                 b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].temperature( b_InputSpec, byref(c_double(Input1[i])),
                                        byref(c_double(Input2[i])),
                                        b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].temperature( b_InputSpec, byref(c_double(Input1[i])),
                                        byref(c_double(Input2)),
                                        b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Density( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].density.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].density( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                             b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].density( b_InputSpec, byref(c_double(Input1[i])),
                                    byref(c_double(Input2[i])),
                                    b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].density( b_InputSpec, byref(c_double(Input1[i])),
                                    byref(c_double(Input2)),
                                    b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def SpecVolume( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].specvolume.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].specvolume( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                                b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].specvolume( b_InputSpec, byref(c_double(Input1[i])),
                                       byref(c_double(Input2[i])),
                                       b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))


    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].specvolume( b_InputSpec, byref(c_double(Input1[i])),
                                       byref(c_double(Input2)),
                                       b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Enthalpy( InputSpec, Input1, Input2) :
    
    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '
    
    FPP[0].enthalpy.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].enthalpy( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                              b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].enthalpy( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2[i])),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))


    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].enthalpy( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2)),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Entropy( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].entropy.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].entropy( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                              b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].entropy( b_InputSpec, byref(c_double(Input1[i])),
                                    byref(c_double(Input2[i])),
                                    b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))


    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].entropy( b_InputSpec, byref(c_double(Input1[i])),
                                    byref(c_double(Input2)),
                                    b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def VaporQual( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].vaporqual.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].vaporqual( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                               b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].vaporqual( b_InputSpec, byref(c_double(Input1[i])),
                                      byref(c_double(Input2[i])),
                                      b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))


    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].vaporqual( b_InputSpec, byref(c_double(Input1[i])),
                                      byref(c_double(Input2)),
                                      b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def IntEnergy( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].intenergy.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].intenergy( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                               b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].intenergy( b_InputSpec, byref(c_double(Input1[i])),
                                      byref(c_double(Input2[i])),
                                      b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].intenergy( b_InputSpec, byref(c_double(Input1[i])),
                                      byref(c_double(Input2)),
                                      b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def HeatCapV( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].heatcapv.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].heatcapv( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                              b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].heatcapv( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2[i])),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].heatcapv( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2)),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def HeatCapP( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].heatcapp.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].heatcapp( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                              b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].heatcapp( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2[i])),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].heatcapp( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2)),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def SoundSpeed( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].soundspeed.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].soundspeed( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                                b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].soundspeed( b_InputSpec, byref(c_double(Input1[i])),
                                       byref(c_double(Input2[i])),
                                       b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].soundspeed( b_InputSpec, byref(c_double(Input1[i])),
                                       byref(c_double(Input2)),
                                       b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Alpha( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].alpha.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].alpha( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                           b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].alpha( b_InputSpec, byref(c_double(Input1[i])),
                                  byref(c_double(Input2[i])),
                                  b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].alpha( b_InputSpec, byref(c_double(Input1[i])),
                                  byref(c_double(Input2)),
                                  b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Beta( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].beta.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].beta( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                          b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].beta( b_InputSpec, byref(c_double(Input1[i])),
                                 byref(c_double(Input2[i])),
                                 b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].beta( b_InputSpec, byref(c_double(Input1[i])),
                                 byref(c_double(Input2)),
                                 b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Chi( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].chi.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].chi( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                         b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].chi( b_InputSpec, byref(c_double(Input1[i])),
                                byref(c_double(Input2[i])),
                                b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].chi( b_InputSpec, byref(c_double(Input1[i])),
                                byref(c_double(Input2)),
                                b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Fi( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].fi.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].fi( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                        b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].fi( b_InputSpec, byref(c_double(Input1[i])),
                               byref(c_double(Input2[i])),
                               b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].fi( b_InputSpec, byref(c_double(Input1[i])),
                               byref(c_double(Input2)),
                               b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Ksi( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].ksi.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].ksi( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                         b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].ksi( b_InputSpec, byref(c_double(Input1[i])),
                                byref(c_double(Input2[i])),
                                b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].ksi( b_InputSpec, byref(c_double(Input1[i])),
                                byref(c_double(Input2)),
                                b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Psi( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].psi.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].psi( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                         b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].psi( b_InputSpec, byref(c_double(Input1[i])),
                                byref(c_double(Input2[i])),
                                b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].psi( b_InputSpec, byref(c_double(Input1[i])),
                                byref(c_double(Input2)),
                                b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Zeta( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].zeta.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].zeta( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                          b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].zeta( b_InputSpec, byref(c_double(Input1[i])),
                                 byref(c_double(Input2[i])),
                                 b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].zeta( b_InputSpec, byref(c_double(Input1[i])),
                                 byref(c_double(Input2)),
                                 b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Theta( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].theta.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].theta( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                           b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].theta( b_InputSpec, byref(c_double(Input1[i])),
                                  byref(c_double(Input2[i])),
                                  b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].theta( b_InputSpec, byref(c_double(Input1[i])),
                                  byref(c_double(Input2)),
                                  b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Kappa( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].kappa.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].kappa( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                           b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].kappa( b_InputSpec, byref(c_double(Input1[i])),
                                  byref(c_double(Input2[i])),
                                  b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].kappa( b_InputSpec, byref(c_double(Input1[i])),
                                  byref(c_double(Input2)),
                                  b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Gamma( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].gamma.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].gamma( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                           b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].gamma( b_InputSpec, byref(c_double(Input1[i])),
                                  byref(c_double(Input2[i])),
                                  b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].gamma( b_InputSpec, byref(c_double(Input1[i])),
                                  byref(c_double(Input2)),
                                  b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def Viscosity( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].viscosity.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].viscosity( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                               b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].viscosity( b_InputSpec, byref(c_double(Input1[i])),
                                      byref(c_double(Input2[i])),
                                      b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].viscosity( b_InputSpec, byref(c_double(Input1[i])),
                                      byref(c_double(Input2)),
                                      b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def ThermCond( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].thermcond.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].thermcond( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                               b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].thermcond( b_InputSpec, byref(c_double(Input1[i])),
                                      byref(c_double(Input2[i])),
                                      b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].thermcond( b_InputSpec, byref(c_double(Input1[i])),
                                      byref(c_double(Input2)),
                                      b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def SurfTens( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].surftens.restype = c_double

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       out = FPP[0].surftens( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)),
                              b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       out = np.zeros(min(len(Input1),len(Input2)))
       for i in range(len(out)):
           out[i] = FPP[0].surftens( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2[i])),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       out = np.zeros(len(Input1))
       for i in range(len(out)):
           out[i] = FPP[0].surftens( b_InputSpec, byref(c_double(Input1[i])),
                                     byref(c_double(Input2)),
                                     b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return out

#-------------------------------------------------------------------------------------------------#
def LiquidCmp( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    x = np.array(20*[0.0]) 

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       FPP[0].liquidcmp( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)), 
                         x.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
       x_return = list()
       for j in range(len(x)):
           x_return.append( float(x[j]))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       len_vec = min(len(Input1),len(Input2))
       x_return = np.zeros((len_vec, 20))
       for i in range(len_vec):
           FPP[0].liquidcmp( b_InputSpec, byref(c_double(Input1[i])), byref(c_double(Input2[i])), 
                             x.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
           for j in range(len(x)):
               x_return[i][j] = float(x[j])

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       len_vec = len(Input1)
       x_return = np.zeros((len_vec, 20))
       for i in range(len_vec):
           FPP[0].liquidcmp( b_InputSpec, byref(c_double(Input1[i])), byref(c_double(Input2)), 
                             x.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
           for j in range(len(x)):
               x_return[i][j] = float(x[j])
                            
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])
   
    ErrorMsg[0] = b_ErrorMsg.decode()

    return x_return

#-------------------------------------------------------------------------------------------------#
def VaporCmp( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    y = np.array(20*[0.0])

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       FPP[0].vaporcmp( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)), 
                        y.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
       y_return = list()
       for j in range(len(y)):
           y_return.append( float(y[j]))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       len_vec = min(len(Input1),len(Input2))
       y_return = np.zeros((len_vec, 20))
       for i in range(len_vec):
           FPP[0].vaporcmp( b_InputSpec, byref(c_double(Input1[i])), byref(c_double(Input2[i])), 
                            y.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
           for j in range(len(y)):
               y_return[i][j] = float(y[j])

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       len_vec = len(Input1)
       y_return = np.zeros((len_vec, 20))
       for i in range(len_vec):
           FPP[0].vaporcmp( b_InputSpec, byref(c_double(Input1[i])), byref(c_double(Input2)), 
                            y.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
           for j in range(len(y)):
               y_return[i][j] = float(y[j])
                            
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])

    ErrorMsg[0] = b_ErrorMsg.decode()

    return y_return

#-------------------------------------------------------------------------------------------------#
def FugaCoef( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    f = np.array(20*[0.0])

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       FPP[0].fugacoef( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)), 
                        f.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
       f_return = list()
       for j in range(len(f)):
           f_return.append( float(f[j]))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
        
       len_vec = min(len(Input1),len(Input2))
       f_return = np.zeros((len_vec, 20))
       for i in range(len_vec):
           FPP[0].fugacoef( b_InputSpec, byref(c_double(Input1[i])), byref(c_double(Input2[i])), 
                            f.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
           for j in range(len(f)):
               f_return[i][j] = float(f[j])

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :
                
       len_vec = len(Input1)
       f_return = np.zeros((len_vec, 20))
       for i in range(len_vec):
           FPP[0].fugacoef( b_InputSpec, byref(c_double(Input1[i])), byref(c_double(Input2)), 
                            f.ctypes.data_as(c_void_p), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
           for j in range(len(f)):
               f_return[i][j] = float(f[j])
                            
    else:
       ErrorMsg[0] = "Error: unknown type for Input1 or Input2..."
       print( ErrorMsg[0])

    ErrorMsg[0] = b_ErrorMsg.decode()

    return f_return

#-------------------------------------------------------------------------------------------------#
def AllProps( InputSpec, Input1, Input2) :

    b_InputSpec = InputSpec.encode()
    l_inputspec = c_int(len(InputSpec))
    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    P   = c_double();    alpha = c_double(); 
    T   = c_double();    beta  = c_double(); 
    v   = c_double();    chi   = c_double(); 
    d   = c_double();    fi    = c_double(); 
    h   = c_double();    ksi   = c_double(); 
    s   = c_double();    psi   = c_double(); 
    u   = c_double();    zeta  = c_double(); 
    q   = c_double();    theta = c_double(); 
    cv  = c_double();    kappa = c_double(); 
    cp  = c_double();    gamma = c_double(); 
    c   = c_double(); 
    eta = c_double();    lamda = c_double(); 

    x = np.array(20*[0.0]) 
    y = np.array(20*[0.0])

    if (type(Input1) is float or type(Input1) is int or type(Input1) is np.float64) and (type(Input2) is float or type(Input2) is int or type(Input2) is np.float64) :
        
       FPP[0].allprops( b_InputSpec, byref(c_double(Input1)), byref(c_double(Input2)), byref(P), 
                        byref(T), byref(v), byref(d), byref(h), byref(s), byref(u), byref(q),  
                        x.ctypes.data_as(c_void_p), y.ctypes.data_as(c_void_p), byref(cv),  
                        byref(cp), byref(c), byref(alpha), byref(beta), byref(chi), byref(fi), 
                        byref(ksi), byref(psi), byref(zeta), byref(theta), byref(kappa), byref(gamma), 
                        byref(eta) ,byref(lamda), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))   
       x_return = list()
       y_return = list()
       for j in range(len(x)):
           x_return.append( float(x[j]))
           y_return.append( float(y[j]))

    elif type(Input1) is np.ndarray and type(Input2) is np.ndarray:
    
       len_vec = min(len(Input1),len(Input2))
       x_return = np.zeros((len_vec, 20))
       y_return = np.zeros((len_vec, 20))
       for i in range(len_vec):
           FPP[0].allprops( b_InputSpec, byref(c_double(Input1[i])), byref(c_double(Input2[i])), byref(P), 
                            byref(T), byref(v), byref(d), byref(h), byref(s), byref(u), byref(q),  
                            x.ctypes.data_as(c_void_p), y.ctypes.data_as(c_void_p), byref(cv),  
                            byref(cp), byref(c), byref(alpha), byref(beta), byref(chi), byref(fi), 
                            byref(ksi), byref(psi), byref(zeta), byref(theta), byref(kappa), byref(gamma), 
                            byref(eta) ,byref(lamda), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
           for j in range(len(x)):
               x_return[i][j] = float(x[j])
               y_return[i][j] = float(y[j])

    elif type(Input1) is np.ndarray and (type(Input2) is float or type(Input2) is int) :

       len_vec = len(Input1)
       x_return = np.zeros((len_vec, 20))
       y_return = np.zeros((len_vec, 20))
       for i in range(len_vec):
           FPP[0].allprops( b_InputSpec, byref(c_double(Input1[i])), byref(c_double(Input2)), byref(P), 
                            byref(T), byref(v), byref(d), byref(h), byref(s), byref(u), byref(q),  
                            x.ctypes.data_as(c_void_p), y.ctypes.data_as(c_void_p), byref(cv),  
                            byref(cp), byref(c), byref(alpha), byref(beta), byref(chi), byref(fi), 
                            byref(ksi), byref(psi), byref(zeta), byref(theta), byref(kappa), byref(gamma), 
                            byref(eta) ,byref(lamda), b_ErrorMsg, byref(l_inputspec), byref(l_errmsg))
           for j in range(len(x)):
               x_return[i][j] = float(x[j])
               y_return[i][j] = float(y[j])

    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return P.value, T.value, v.value, d.value, h.value, s.value, u.value, q.value, x_return, y_return, cv.value, cp.value, c.value, alpha.value, beta.value, chi.value, fi.value, ksi.value, psi.value, zeta.value, theta.value, kappa.value, gamma.value, eta.value, lamda.value

#-------------------------------------------------------------------------------------------------#
def PhaseEnvelope() :

    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    nPnts = c_int(0)
    T = np.zeros(5000); 
    P = np.zeros(5000);
    x = np.zeros((20,5000)); 
    y = np.zeros((20,5000)); 

    FPP[0].saturationline( byref(nPnts), T.ctypes.data_as(c_void_p), P.ctypes.data_as(c_void_p),
                           x.ctypes.data_as(c_void_p), x.ctypes.data_as(c_void_p),
                           b_ErrorMsg, byref(l_errmsg))

    T_return = list()
    P_return = list()
    i = 0
    while i < len(P) and P[i] != 0.0:
        T_return.append( float(T[i]))
        P_return.append( float(P[i]))
        i += 1

    ErrorMsg[0] = b_ErrorMsg.decode()

    return T_return, P_return

#-------------------------------------------------------------------------------------------------#
def Solve( FuncSpec, FuncVal, InputSpec, Target, FixedVal, MinVal, MaxVal) :

    # Solves x or y from function z = f(x,y) by using a Brent iteration method.
    # -------------------------------------------------------------------------
     
    b_FuncSpec = FuncSpec.encode()
    l_FuncSpec = c_int(len(b_FuncSpec))
    b_InputSpec = InputSpec.encode()
    l_InputSpec = c_int(len(b_InputSpec))

    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].solve.restype = c_double

    out = FPP[0].solve( b_FuncSpec, byref(c_double(FuncVal)), b_InputSpec, byref(c_int(Target)), 
                        byref(c_double(FixedVal)), byref(c_double(MinVal)), byref(c_double(MaxVal)), 
                        b_ErrorMsg, byref(l_FuncSpec), byref(l_InputSpec), byref(l_errmsg))
    
    ErrorMsg[0] = b_ErrorMsg.decode()

    return float(out)
#-------------------------------------------------------------------------------------------------#
def Mmol() :

    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].mmol.restype = c_double
    out = FPP[0].mmol( b_ErrorMsg, byref(l_errmsg))
    
    ErrorMsg[0] = b_ErrorMsg.decode()

    return float(out)

#-------------------------------------------------------------------------------------------------#
def Tcrit() :

    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].tcrit.restype = c_double
    out = FPP[0].tcrit( b_ErrorMsg, byref(l_errmsg))
    
    ErrorMsg[0] = b_ErrorMsg.decode()

    return float(out)

#-------------------------------------------------------------------------------------------------#
def Pcrit() :

    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].pcrit.restype = c_double
    out = FPP[0].pcrit( b_ErrorMsg, byref(l_errmsg))
    
    ErrorMsg[0] = b_ErrorMsg.decode()
    
    return float(out)

#-------------------------------------------------------------------------------------------------#
def Tmin() :

    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].tmin.restype = c_double
    out = FPP[0].tmin( b_ErrorMsg, byref(l_errmsg))
    
    ErrorMsg[0] = b_ErrorMsg.decode()

    return float(out)

#-------------------------------------------------------------------------------------------------#
def Tmax() :

    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    FPP[0].tmax.restype = c_double
    out = FPP[0].tmax( b_ErrorMsg, byref(l_errmsg))
    
    ErrorMsg[0] = b_ErrorMsg.decode()

    return float(out)

#-------------------------------------------------------------------------------------------------#
def AllInfo() :

    l_errmsg = c_int(80)
    b_ErrorMsg = b'No errors                                                                       '

    M_mol  = c_double() 
    T_crit = c_double()  
    P_crit = c_double()  
    T_min  = c_double() 
    T_max  = c_double() 
    FPP[0].allinfo( byref(M_mol), byref(T_crit), byref(P_crit), byref(T_min), byref(T_max), 
                    b_ErrorMsg, byref(l_errmsg))

    ErrorMsg[0] = b_ErrorMsg.decode()

    return M_mol, T_crit, P_crit, T_min, T_max

#===================================================================================== EOF ==!
