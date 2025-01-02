import cantera as ct


def evaluate_fitness(genome):

    try:
        gas = ct.Solution("gri30.yaml")
        reactions = gas.reactions()
        reduced_reactions = [
            reaction for i, reaction in enumerate(reactions) if genome[i] == 1
        ]
        reduced_mechanism = ct.Solution(
            thermo="IdealGas", 
            kinetics="GasKinetics", 
            species=gas.species(), 
            reactions=reduced_reactions)
        gas.TP = 1000, ct.one_atm
        gas.X = {"CH4": 1.0, "O2": 3.0, "N2": 3.76}

        #print(f"Initial Pressure: {gas.P:.2f} Pa")
        print(f"Temperature: {gas.T:.2f} K")

        reactor = ct.Reactor(contents=gas, energy='on') # If set to 'off', the energy equation is not solved
        sim = ct.ReactorNet([reactor])

        # Advance the simulation
        time = 0.0  # Start time
        end_time = 10.0  # End time (seconds)
        time_step = 0.5  # Time step (seconds)

        #sim.advance(1.0)  # Advance simulation

        #print("Time [s] | Temperature [K] | Pressure [Pa] | CH4 Mole Fraction")
        while time < end_time:
            time += time_step
            sim.advance(time)
            #print(f"{time:8.3f} | {reactor.T:14.3f} | {reactor.thermo.P:12.1f} | {reactor.thermo['CH4'].X[0]:18.6f}")
        #print("----------------------------------------------------------------\n")


        target_temperature = 1200
        fitness = abs(reactor.T - target_temperature)
        return fitness # returns to genetic-algo class, evolve function
    
    except Exception as e:
        print(f"Error during fitness evaluation: {e}")
        return 1e6  # Penalize invalid solutions
