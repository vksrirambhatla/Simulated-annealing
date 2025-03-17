import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core.structure import Molecule, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from simanneal import Annealer
from pymatgen.io.cif import CifWriter

# Load your molecular structure
molecule = Molecule.from_file("your_molecular_structure.xyz")

# Load your experimental powder pattern data
data = np.loadtxt("your_powder_data.txt")
two_theta_exp = data[:, 0]
intensity_exp = data[:, 1]

# Define a function to calculate the theoretical powder pattern
def calculate_pattern(molecule, two_theta_exp):
    xrd_calculator = XRDCalculator()
    pattern = xrd_calculator.get_pattern(molecule, two_theta_range=(min(two_theta_exp), max(two_theta_exp)))
    return pattern.y

# Define the objective function for simulated annealing
def objective_function(params):
    # Update molecule positions based on params
    molecule.translate_sites(range(len(molecule)), params[:3])
    molecule.rotate_sites(range(len(molecule)), params[3:], anchor=(0, 0, 0))
    
    # Calculate the theoretical pattern
    intensity_theo = calculate_pattern(molecule, two_theta_exp)
    
    # Calculate the difference between experimental and theoretical patterns
    diff = np.sum((intensity_exp - intensity_theo) ** 2)
    return diff

# Define the Annealer class
class CrystalAnnealer(Annealer):
    def move(self):
        # Randomly perturb the parameters
        self.state += np.random.uniform(-0.1, 0.1, size=self.state.shape)
    
    def energy(self):
        return objective_function(self.state)

# Initial guess for the parameters (translation and rotation)
initial_params = np.zeros(6)

# Print initial cell parameters
print("Initial cell parameters (translation and rotation):", initial_params)

# Create the annealer and set the initial state
annealer = CrystalAnnealer(initial_params)
annealer.steps = 1000

# Run the annealing process
best_params, best_energy = annealer.anneal()

# Update the molecule with the best parameters
molecule.translate_sites(range(len(molecule)), best_params[:3])
molecule.rotate_sites(range(len(molecule)), best_params[3:], anchor=(0, 0, 0))

# Calculate the best fit pattern
intensity_best_fit = calculate_pattern(molecule, two_theta_exp)

# Print the best fit cell parameters
print("Best fit cell parameters (translation and rotation):", best_params)

# Save the solved structure to a CIF file
structure = Structure.from_sites(molecule.sites)
cif_writer = CifWriter(structure)
cif_writer.write_file("solved_structure.cif")

# Plot the experimental and best fit patterns
plt.plot(two_theta_exp, intensity_exp, label="Experimental")
plt.plot(two_theta_exp, intensity_best_fit, label="Best Fit")
plt.xlabel("2θ (degrees)")
plt.ylabel("Intensity")
plt.legend()
plt.show()
