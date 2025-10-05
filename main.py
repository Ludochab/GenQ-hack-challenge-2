from bloqade.analog.atom_arrangement import Square
from bloqade.analog.atom_arrangement import AtomArrangement
import numpy as np
from quera_wrapper import QuEraWrapper
import json
from bloqade import start


def run(input_data, solver_params, extra_arguments):    
    ##### THIS IS HOW YOU READ INPUT DATA FROM JSON #####
    positions = input_data['positions']
    positions = scale_and_snap_positions(positions)
    indices = input_data.get('indices', list(range(len(positions))))
    #######################################################

    
    # Change the lattice spacing to vary the atom separation a, and thus also Rb/a
    delta_end=2*np.pi*6.8 #final detuning
    omega_max=2*np.pi*2.5 #max Rabi amplitude
    C6 = 2*np.pi * 862690;
    Rb = (C6 / (omega_max) )** (1/6) # R_B during bulk of protocol

    durations = [0.8, 2.4, 0.8]  # total sweep time = 4.0 μs
    rabi_amplitude_values = [0.0, omega_max, omega_max, 0.0]
    rabi_detuning_values = [-delta_end, -delta_end, delta_end, delta_end]

    arrangement = start.add_position([tuple(pos) for pos in positions])
    
    program = (
    arrangement.rydberg
    .rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude_values)
    .detuning.uniform.piecewise_linear(durations, rabi_detuning_values)
    )

    ##############################################################################################################
    ########################## ENTERING QCENTROID QUERA WRAPPER ##################################################
    ##############################################################################################################
    
    QuEraWrapper.program=program
    results=QuEraWrapper.run(shots=100,interaction_picture=True)
    counts=results.report().counts()

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################

    # If needed, sort bitstrings by indices (usually not needed if QuEra returns in order)
    # But you can ensure the order by:
    ordered_counts = {}
    for bitstring, count in counts.items():
        # bitstring is already in the order of indices
        ordered_counts[bitstring] = count
    
    return ordered_counts


def scale_and_snap_positions(positions, min_dist=4, max_abs=37.5):
    positions = np.array(positions)
    # Center
    center = positions.mean(axis=0)
    positions -= center

    # Scale to fit within [-max_abs, max_abs]
    max_dim = np.max(np.abs(positions))
    if max_dim > 0:
        scale = max_abs / max_dim
    else:
        scale = 1.0
    positions *= scale

    # Snap y to multiples of 4
    positions[:, 1] = 4 * np.round(positions[:, 1] / 4)

    # Check minimum distance and rescale if needed
    def min_pairwise_dist(pos):
        from scipy.spatial.distance import pdist
        return np.min(pdist(pos))

    while min_pairwise_dist(positions) < min_dist:
        positions *= min_dist / min_pairwise_dist(positions)

    return positions.tolist()