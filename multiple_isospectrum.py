import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def plot_isospectral_gallery_single(parameter_sets, save_pdf=False, filename="isospectral_gallery.pdf"):
    """
    Plot ALL isospectral pairs in ONE figure to show variety of shapes
    """
    
    # Create single figure
    plt.figure(figsize=(10, 6))
    
    # Energy range for plotting
    k_plot = np.linspace(0.1, 10.0, 1000)
    E_plot = k_plot**2
    
    # Different color combinations for each pair
    color_combinations = [
        ('red', 'blue'),      # Pair 1
        ('green', 'orange'),  # Pair 2  
        ('brown', 'purple'),  # Pair 3
        ('pink', 'gray'),     # Pair 4
        ('olive', 'cyan'),    # Pair 5
        ('magenta', 'teal'),  # Pair 6
        ('gold', 'navy')      # Pair 7
    ]
    
    for idx, params in enumerate(parameter_sets):
        α1_DDDP, α2_DDDP, β1_TDDP, β2_TDDP, β3_TDDP, x1_TDDP, x2_TDDP, x3_TDDP = params
        
        T_DDDP, T_TDDP = [], []
        
        for k in k_plot:
            # DDDP system
            beta1_DDDP = α1_DDDP / k
            beta2_DDDP = α2_DDDP / k
            
            M1_DDDP = np.array([
                [1 + 1j*beta1_DDDP, 1j*beta1_DDDP * np.exp(2j*k*1.0)],
                [-1j*beta1_DDDP * np.exp(-2j*k*1.0), 1 - 1j*beta1_DDDP]
            ])
            M2_DDDP = np.array([
                [1 + 1j*beta2_DDDP, 1j*beta2_DDDP * np.exp(-2j*k*1.0)],
                [-1j*beta2_DDDP * np.exp(2j*k*1.0), 1 - 1j*beta2_DDDP]
            ])
            M_DDDP = M2_DDDP @ M1_DDDP
            T_DDDP.append(1 / np.abs(M_DDDP[0,0])**2)
            
            # TDDP system
            beta1_TDDP = β1_TDDP / k
            beta2_TDDP = β2_TDDP / k
            beta3_TDDP = β3_TDDP / k
            
            M1_TDDP = np.array([
                [1 + 1j*beta1_TDDP, 1j*beta1_TDDP * np.exp(2j*k*abs(x1_TDDP))],
                [-1j*beta1_TDDP * np.exp(-2j*k*abs(x1_TDDP)), 1 - 1j*beta1_TDDP]
            ])
            M2_TDDP = np.array([
                [1 + 1j*beta2_TDDP, 1j*beta2_TDDP * np.exp(2j*k*abs(x2_TDDP))],
                [-1j*beta2_TDDP * np.exp(-2j*k*abs(x2_TDDP)), 1 - 1j*beta2_TDDP]
            ])
            M3_TDDP = np.array([
                [1 + 1j*beta3_TDDP, 1j*beta3_TDDP * np.exp(2j*k*abs(x3_TDDP))],
                [-1j*beta3_TDDP * np.exp(-2j*k*abs(x3_TDDP)), 1 - 1j*beta3_TDDP]
            ])
            M_TDDP = M3_TDDP @ M2_TDDP @ M1_TDDP
            T_TDDP.append(1 / np.abs(M_TDDP[0,0])**2)
        
        # Get color combination for this pair
        DDDP_color, TDDP_color = color_combinations[idx % len(color_combinations)]
        
        # Plot with different colors for each pair
        plt.plot(E_plot, T_DDDP, color=DDDP_color, linewidth=1.5, alpha=0.8)
        plt.plot(E_plot, T_TDDP, color=TDDP_color, linewidth=1.5, alpha=0.8, linestyle='--')
    
    plt.xlabel('Energy E')
    plt.ylabel('Transmission T(E)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Add caption in lower right
    plt.text(0.98, 0.02, 'Isospectral pairs with varied interference patterns', 
             transform=plt.gca().transAxes, fontsize=10, 
             ha='right', va='bottom', style='italic',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save to PDF if requested
    if save_pdf:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    plt.show()

# Updated main section:
if __name__ == "__main__":
    # Your collected parameter sets
    parameter_sets = [
        # Case 1
        (-3.650078, -1.914443, 2.949053, 0.702677, 1.914443, -0.335778, 0.335380, 2.335702),
        
        # Case 2
        (+2.999394, -0.106228, +0.271739, -0.165504, -2.999394, -0.209924, 0.210009, 2.209791),
        
        # Case 3
        (-0.812730, -2.298201, +2.398335, -0.100013, +0.812730, -0.244848, 0.245100, 2.244837),
        
        # Case 4
        (+0.639407, +0.737779, +0.737759, +2.421160, -1.777676, -2.200269, -0.201579, 0.202050),
        
        # Case 5
        (-2.994347, -3.032027, -2.994335, -0.112360, -2.920118, -2.202622, -0.201980, 0.202647),
        
        # Case 6
        (+0.543251, -3.964109, +2.994861, +0.967739, -0.543251, -0.356952, 0.357213, 2.357016),
        
        # Case 7
        (-1.971139, +0.104975, +2.093799, -0.122103, -0.104975, -0.499907, 0.501001, 2.499839)
    ]
    
    plot_isospectral_gallery_single(parameter_sets, save_pdf=True, filename="isospectral_gallery.pdf")