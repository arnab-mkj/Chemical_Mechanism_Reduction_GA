# **Detailed Workflow of the Genetic Algorithm for Chemical Mechanism Reduction**

This document provides a comprehensive explanation of the workflow of the program, including the mechanisms, formulas, and dependencies used in each step. The goal of the program is to reduce a chemical mechanism using a genetic algorithm (GA) while maintaining the accuracy of key combustion properties such as temperature, species mole fractions, and ignition delay.

---

## **1. Overview of the Workflow**

The program uses a **genetic algorithm (GA)** to optimize a chemical mechanism by reducing the number of reactions while preserving its predictive accuracy. The workflow consists of the following steps:

1. **Initialization**:
   - Define the GA parameters (population size, genome length, mutation rate, etc.).
   - Load the original chemical mechanism (e.g., GRI-Mech 3.0).
   - Initialize the population of genomes, where each genome represents a subset of active reactions.

2. **Fitness Evaluation**:
   - For each genome, create a reduced mechanism and run simulations to evaluate its performance.
   - Calculate the fitness score based on temperature, species mole fractions, and ignition delay.

3. **Genetic Operations**:
   - Perform selection, crossover, and mutation to generate a new population.
   - Replace the old population with the new one.

4. **Evolution**:
   - Repeat the fitness evaluation and genetic operations for a specified number of generations.
   - Track the best genome and its fitness score over generations.

5. **Output**:
   - Save the best reduced mechanism in YAML format.
   - Save fitness history and species mole fractions for analysis.

---

## **2. Detailed Steps and Mechanisms**

### **Step 1: Initialization**

#### **Genetic Algorithm Parameters**
- **Population Size**: Number of genomes in each generation (e.g., 30).
- **Genome Length**: Number of reactions in the original mechanism (e.g., 325 for GRI-Mech 3.0).
- **Crossover Rate**: Probability of combining two genomes (e.g., 0.8).
- **Mutation Rate**: Probability of flipping a reaction's active/inactive state (e.g., 0.1).
- **Number of Generations**: Total iterations of the GA (e.g., 100).

#### **Population Initialization**
- Each genome is a binary array where:
  - `1` indicates the reaction is active.
  - `0` indicates the reaction is inactive.
- Diversity is introduced by randomly deactivating some reactions in the initial population.

**Formula**:
$
\text{Genome} = [1, 1, 0, 1, \dots, 0]
$

---

### **Step 2: Fitness Evaluation**

#### **2.1 Create Reduced Mechanism**
- The genome is used to filter the reactions in the original mechanism.
- A reduced mechanism is created using only the active reactions.

**Dependencies**:
- **Cantera**: Used to load the original mechanism and create the reduced mechanism.

**Validation**:
- Ensure the reduced mechanism has at least 50 reactions to avoid instability.

---

#### **2.2 Run Simulation**
- The reduced mechanism is simulated in a constant-pressure reactor using **Cantera**.
- Initial conditions:
  - Temperature: 1000 K
  - Pressure: 1 atm
  - Species: Stoichiometric mixture of CH₄, O₂, and N₂.

**Mechanism**:
The simulation solves the governing equations for mass, energy, and species conservation:
$
\frac{dT}{dt} = \frac{1}{\rho c_p} \sum_k h_k \dot{\omega}_k
$
$
\frac{dY_k}{dt} = \frac{\dot{\omega}_k}{\rho}
$
Where:
- $T$: Temperature
- $Y_k$: Mass fraction of species $k$
- $\dot{\omega}_k$: Production rate of species $k$
- $\rho$: Density
- $c_p$: Specific heat capacity

---

#### **2.3 Calculate Fitness**
The fitness score is a weighted combination of three components:

1. **Temperature Fitness**:
   - Measures the deviation of the final temperature from the target temperature.
   $
   f_{\text{temperature}} = |T_{\text{actual}} - T_{\text{target}}|
   $

2. **Species Fitness**:
   - Measures the deviation of key species mole fractions from their target values.
   $
   f_{\text{species}} = \frac{1}{N_s} \sum_{i=1}^{N_s} \text{calculate\_difference}(X_i^{\text{actual}}, X_i^{\text{target}})
   $
   Where $N_s$ is the number of target species.

3. **Ignition Delay Fitness**:
   - Measures the deviation of the ignition delay time from the target value.
   $
   f_{\text{ignition}} = |t_{\text{actual}} - t_{\text{target}}|
   $

**Combined Fitness**:
$
f_{\text{total}} = w_T f_{\text{temperature}} + w_S f_{\text{species}} + w_I f_{\text{ignition}}
$
Where $w_T, w_S, w_I$ are the weights for temperature, species, and ignition delay fitness, respectively.

---

### **Step 3: Genetic Operations**

#### **3.1 Selection**
- **Tournament Selection**:
  - Randomly select two genomes and choose the one with the lower fitness score.

**Formula**:
$
\text{Selected Genome} = \arg\min(f_{\text{genome1}}, f_{\text{genome2}})
$

---

#### **3.2 Crossover**
- **Single-Point Crossover**:
  - Combine two parent genomes at a random crossover point to create two offspring.

**Mechanism**:
$
\text{Child1} = [\text{Parent1}_{1:c}, \text{Parent2}_{c+1:n}]
$
$
\text{Child2} = [\text{Parent2}_{1:c}, \text{Parent1}_{c+1:n}]
$

---

#### **3.3 Mutation**
- **Bit-Flip Mutation**:
  - Randomly flip bits in the genome with a probability equal to the mutation rate.

**Mechanism**:
$
\text{Mutated Genome}[i] = 1 - \text{Genome}[i] \quad \text{if } \text{rand} < \text{mutation rate}
$

---

### **Step 4: Evolution**
- Repeat the fitness evaluation and genetic operations for the specified number of generations.
- Track the best genome and its fitness score in each generation.

---

### **Step 5: Output**

1. **Save Reduced Mechanism**:
   - Save the best genome as a YAML file using **Cantera**.

2. **Save Fitness History**:
   - Save the fitness scores of all generations for analysis.

3. **Save Species Mole Fractions**:
   - Save the mole fractions of key species for the best genome.

---

## **3. Key Dependencies**

1. **Cantera**:
   - Used for chemical kinetics simulations and mechanism handling.

2. **NumPy**:
   - Used for numerical operations and genome manipulation.

3. **Matplotlib**:
   - Used for real-time plotting of fitness scores.

4. **YAML**:
   - Used for saving the reduced mechanism in a readable format.

---

## **4. Summary**

The program uses a genetic algorithm to optimize a chemical mechanism by reducing the number of reactions while maintaining accuracy. The fitness evaluation combines temperature, species mole fractions, and ignition delay to guide the optimization process. The best reduced mechanism is saved for further analysis and use.