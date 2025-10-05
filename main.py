from bloqade.analog.atom_arrangement import Square
import numpy as np
from quera_wrapper import QuEraWrapper
import json
def run(input_data, solver_params, extra_arguments):
    
    ##### THIS IS HOW YOU READ INPUT DATA FROM JSON #####
    positions = input_data['positions']
    indices = input_data.get('indices', list(range(len(positions))))
    #######################################################

    
    # Change the lattice spacing to vary the atom separation a, and thus also Rb/a
    delta_end=2*np.pi*6.8 #final detuning
    omega_max=2*np.pi*2.5 #max Rabi amplitude
    lattice_spacing = 7.0 #size of edges of square lattice
    
    C6 = 2*np.pi * 862690;
    Rb = (C6 / (omega_max) )** (1/6) # R_B during bulk of protocol
    sweep_time = 2.4 #time length of the protocol 
    rabi_amplitude_values = [0.0, omega_max, omega_max, 0.0]
    rabi_detuning_values = [-delta_end, -delta_end, delta_end, delta_end]
    durations = [0.8, sweep_time, 0.8]
    
    
    geometries = {
        1: Square(3, lattice_spacing=lattice_spacing),
        2: Square(11, lattice_spacing=lattice_spacing),
    }
    
    prog_list = {
        idx:(geometry.rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude_values)
        .detuning.uniform.piecewise_linear(durations, rabi_detuning_values) )for idx, geometry in geometries.items()
    }


    ##############################################################################################################
    ########################## ENTERING QCENTROID QUERA WRAPPER ##################################################
    ##############################################################################################################
    
    QuEraWrapper.program=prog_list[1]
    x=QuEraWrapper.run(shots=1000)
    r=x.report()

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    
    # Ensure counts are returned as {bitstring: count} with bitstrings in the order of indices
    counts = r.counts()
    # If needed, sort bitstrings by indices (usually not needed if QuEra returns in order)
    # But you can ensure the order by:
    ordered_counts = {}
    for bitstring, count in counts.items():
        # bitstring is already in the order of indices
        ordered_counts[bitstring] = count
    

    return ordered_counts