import cantera as ct
import json
import numpy as np

def idt_value(mech, condition,soln):
    
    #print("mech: ", mech, "; Type mech: ", type(mech))
    fuel = condition['fuel']
    oxidizer_comp = condition['oxidizer']
    phi = condition['phi']
    pressure = condition['pressure']
    temperature = condition['temperature']

    simtype = 'HP'
    if soln:
        gas = ct.Solution(mech) # only for full mech(True)
    else:
        gas = mech # for reduced mech(False)
    gas.set_equivalence_ratio( phi, fuel, oxidizer_comp )
    gas.TP = temperature, pressure
    #print("Type gas: ", type(gas))
    # Get equilibrium temperature for ignition break
    gas.equilibrate(simtype)
    T_equi = gas.T

    gas.set_equivalence_ratio( phi, fuel, oxidizer_comp )
    gas.TP = temperature, pressure
    
    # here it is constant pressure with IdealGasReactor
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    
    
    # set the tolerances for the solution and for the sensitivity coefficients
    sim.rtol = 1.0e-6
    sim.atol = 1.0e-15
    c = 0
    t_end = 0.05
    time = []
    temp = []
    states = ct.SolutionArray(gas, extra=['t'])
    # print(gas.X)
    # print(f"Initial sim.time: {sim.time}")
    # print(f"t_end: {t_end}")
    # print(f"Initial reactor temperature (r.T): {r.T}")
    # print(f"Equilibrium temperature (T_equi): {T_equi}")
    # print(f"Pressure: ", gas.P)
    # stateArray = []
    while sim.time < t_end and r.T < T_equi - 0.1:
        sim.step()
        time.append(sim.time)
        temp.append(r.T)
        states.append(r.thermo.state, t=sim.time)
        c += 1
        # stateArray.append(r.thermo.X)
    # print("Numer of steps: ", c)
    time = np.array(time)
    temp = np.array(temp)
    diff_temp = np.diff(temp)/ np.diff(time)

    ign_pos = np.argmax( diff_temp )
    ign = time[ign_pos] # convert to ms then *1000
    #print(ign) #in seconds
    # print("Ignition Delay is: {:.10f} ms".format(ign))
    return ign *1000 #Returns the ignition delay time (IDT) as a single value


# condition = {
#         "phi": 2.5,
#         "fuel": {"CH4": 1.0},
#         "oxidizer": {"O2": 2.0, "N2": 7.52},
#         "pressure": ct.one_atm,
#         "temperature": 1736.0,
#         "mass_flow_rate": 0.0989
#     }

# mech = 'reduced_mech_64_rxns.yaml'
# idt_value(mech, condition)