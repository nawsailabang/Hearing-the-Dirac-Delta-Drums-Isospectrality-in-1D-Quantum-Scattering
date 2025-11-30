import numpy as np
import matplotlib.pyplot as plt

def plot_isospectral_gallery_complete(parameter_sets_complete, save_pdf=False, filename="isospectral_gallery_complete.pdf"):
    """
    Plot ALL isospectral pairs with complete parameter sets (including positions)
    """
    
    # Create single figure
    plt.figure(figsize=(12, 8))
    
    # k range for plotting
    k_plot = np.linspace(0.1, 3.2, 1500)
    
    # Different color combinations for each pair
    color_combinations = [
        ('blue', 'red'),           # Pair 1
        ('green', 'orange'),       # Pair 2  
        ('brown', 'purple'),       # Pair 3
        ('cyan', 'magenta'),       # Pair 4
        ('olive', 'teal'),         # Pair 5
        ('navy', 'maroon'),        # Pair 6
        ('darkgreen', 'darkorange') # Pair 7
    ]
    
    # Line styles with thicker lines
    linewidth_solid = 2.5
    linewidth_dashed = 2.5
    alpha_value = 0.9
    
    for idx, params in enumerate(parameter_sets_complete):
        # Extract complete parameters: α1, α2, xa, xb, β1, β2, β3, x1, x2, x3
        α1, α2, xa, xb, β1, β2, β3, x1, x2, x3 = params
        
        T_DDDP, T_TDDP = [], []
        
        for k in k_plot:
            # Two-delta system
            beta1_DDDP = α1 / k
            beta2_DDDP = α2 / k
            
            M1_DDDP = np.array([
                [1 + 1j*beta1_DDDP, 1j*beta1_DDDP * np.exp(2j*k*xa)],
                [-1j*beta1_DDDP * np.exp(-2j*k*xa), 1 - 1j*beta1_DDDP]
            ])
            M2_DDDP = np.array([
                [1 + 1j*beta2_DDDP, 1j*beta2_DDDP * np.exp(2j*k*xb)],
                [-1j*beta2_DDDP * np.exp(-2j*k*xb), 1 - 1j*beta2_DDDP]
            ])
            M_DDDP = M2_DDDP @ M1_DDDP
            T_DDDP.append(1 / np.abs(M_DDDP[0,0])**2)
            
            # Three-delta system
            beta1_TDDP = β1 / k
            beta2_TDDP = β2 / k
            beta3_TDDP = β3 / k
            
            M1_TDDP = np.array([
                [1 + 1j*beta1_TDDP, 1j*beta1_TDDP * np.exp(2j*k*x1)],
                [-1j*beta1_TDDP * np.exp(-2j*k*x1), 1 - 1j*beta1_TDDP]
            ])
            M2_TDDP = np.array([
                [1 + 1j*beta2_TDDP, 1j*beta2_TDDP * np.exp(2j*k*x2)],
                [-1j*beta2_TDDP * np.exp(-2j*k*x2), 1 - 1j*beta2_TDDP]
            ])
            M3_TDDP = np.array([
                [1 + 1j*beta3_TDDP, 1j*beta3_TDDP * np.exp(2j*k*x3)],
                [-1j*beta3_TDDP * np.exp(-2j*k*x3), 1 - 1j*beta3_TDDP]
            ])
            M_TDDP = M3_TDDP @ M2_TDDP @ M1_TDDP
            T_TDDP.append(1 / np.abs(M_TDDP[0,0])**2)
        
        # Get color combination for this pair
        DDDP_color, TDDP_color = color_combinations[idx % len(color_combinations)]
        
        # Plot with thicker lines for better visibility
        plt.plot(k_plot, T_DDDP, color=DDDP_color, linewidth=linewidth_solid, alpha=alpha_value,
                label=f'Two-delta {idx+1}' if idx == 0 else "")
        plt.plot(k_plot, T_TDDP, color=TDDP_color, linewidth=linewidth_dashed, alpha=alpha_value, 
                linestyle='--', label=f'Three-delta {idx+1}' if idx == 0 else "")
    
    plt.xlabel('Wave number k', fontsize=12)
    plt.ylabel('Transmission T(k)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xlim(0, 3)
    
    
    
    # Add caption
    plt.text(0.98, 0.02, 'Multiple isospectral pairs (0 < k < 3)', 
             transform=plt.gca().transAxes, fontsize=11, 
             ha='right', va='bottom', style='italic',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    
    if save_pdf:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # All cases from your images
    parameter_sets_complete = [
        # Case 1
        (-2.0148, +1.3972, -0.0822, +1.0719, +1.9008, -0.5402, -1.9000, +0.0021, +0.7353, +1.1351),
        
        # Case 2
        (-3.5000, +3.5000, +0.9703, +1.9999, -3.0000, +0.4830, -2.8603, -0.1209, +0.6569, +1.2724),
        
        # Case 3
        (-1.0519, +0.5871, -0.6891, +0.1048, -0.8551, -0.4996, +0.9250, -0.1910, +0.1950, +0.5827),
        
        # Case 4
        (+1.0197, -1.6319, -1.0443, +0.0742, -1.4672, -0.5618, +1.4710, +0.2524, +0.6742, +1.3504),
        
        # Case 5
        (-3.5000, +1.9247, +0.2547, +1.0456, +3.0000, -0.4894, -3.0000, -0.2412, +0.1458, +0.5333),
        
        # Case 6
        (-0.4898, -3.5000, +0.0687, +1.9448, -0.4959, +0.4809, -2.8108, -0.0430, +1.4997, +1.8966),
        
        # Case 7
        (-2.4054, +2.5847, -1.8003, +0.6083, +2.3882, +0.4995, -1.9359, -1.2459, +0.7174, +1.2570),
        
        # Case 8
        (-0.4931, -3.5000, -0.6005, +1.5050, -0.4930, +0.4812, -2.8139, -1.4493, +0.3234, +0.7196)
    ]
    
    # Plot all cases with thicker lines
    plot_isospectral_gallery_complete(parameter_sets_complete, save_pdf=True, filename="isospectral_gallery_complete.pdf")
