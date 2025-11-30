import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

def create_isospectral_explorer_wide_range():
    """Interactive explorer with WIDER parameter ranges and E < 10"""
    
    # Initial parameters - more diverse starting point
    init_params = {
        'alpha1': 1.5, 'alpha2': -1.2,  # Two-delta system
        'x1_dd': -1.0, 'x2_dd': 1.0,    # Two-delta positions (now adjustable)
        'beta1': 0.5, 'beta2': -2.0, 'beta3': 1.8,  # Three-delta system
        'x1_td': -1.8, 'x2_td': 0.5, 'x3_td': 2.2  # Three-delta positions
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 1])
    
    ax_trans = fig.add_subplot(gs[0, :])
    ax_potential = fig.add_subplot(gs[1, :])
    ax_diff = fig.add_subplot(gs[2, :])
    
    plt.subplots_adjust(bottom=0.5)  # More room for additional sliders
    
    # Energy range for plotting - ONLY E < 10
    k_plot = np.linspace(0.1, np.sqrt(10), 600)  # E = k^2 < 10
    E_plot = k_plot**2
    
    def compute_transmission(params):
        """Compute transmission for both systems - with proper matrix multiplication order"""
        T_DDDP, T_TDDP = [], []
        
        for k in k_plot:
            # Two-delta system (now with adjustable positions)
            # IMPORTANT: Matrices must be multiplied in order of increasing x
            beta1 = params['alpha1'] / k
            beta2 = params['alpha2'] / k
            
            # Sort two-delta positions and corresponding strengths
            dd_positions = [(params['x1_dd'], beta1), (params['x2_dd'], beta2)]
            dd_positions.sort(key=lambda x: x[0])  # Sort by position
            
            # Create matrices in order of increasing x
            M_total_dd = np.eye(2, dtype=complex)
            for x_pos, beta_val in dd_positions:
                M = np.array([
                    [1 + 1j*beta_val, 1j*beta_val * np.exp(2j*k*x_pos)],
                    [-1j*beta_val * np.exp(-2j*k*x_pos), 1 - 1j*beta_val]
                ])
                M_total_dd = M @ M_total_dd  # Multiply in proper order
            
            T_DDDP.append(1 / np.abs(M_total_dd[0,0])**2)
            
            # Three-delta system
            beta1_t = params['beta1'] / k
            beta2_t = params['beta2'] / k  
            beta3_t = params['beta3'] / k
            
            # Sort three-delta positions and corresponding strengths
            td_positions = [
                (params['x1_td'], beta1_t), 
                (params['x2_td'], beta2_t), 
                (params['x3_td'], beta3_t)
            ]
            td_positions.sort(key=lambda x: x[0])  # Sort by position
            
            # Create matrices in order of increasing x
            M_total_td = np.eye(2, dtype=complex)
            for x_pos, beta_val in td_positions:
                M = np.array([
                    [1 + 1j*beta_val, 1j*beta_val * np.exp(2j*k*x_pos)],
                    [-1j*beta_val * np.exp(-2j*k*x_pos), 1 - 1j*beta_val]
                ])
                M_total_td = M @ M_total_td  # Multiply in proper order
            
            T_TDDP.append(1 / np.abs(M_total_td[0,0])**2)
        
        return np.array(T_DDDP), np.array(T_TDDP)
    
    # Initial plot
    T_DDDP, T_TDDP = compute_transmission(init_params)
    diff = T_DDDP - T_TDDP
    
    # Transmission plot
    line1, = ax_trans.plot(E_plot, T_DDDP, 'b-', linewidth=2, label='Two-delta')
    line2, = ax_trans.plot(E_plot, T_TDDP, 'r--', linewidth=2, label='Three-delta')
    ax_trans.set_xlabel('Energy E')
    ax_trans.set_ylabel('Transmission T(E)')
    ax_trans.legend()
    ax_trans.grid(True, alpha=0.3)
    ax_trans.set_ylim(-0.1, 1.1)
    ax_trans.set_xlim(0, 10)  # Ensure x-axis only shows E < 10
    
    # Potential plot
    ax_potential.clear()
    # Two-delta (plot in actual positions)
    ax_potential.stem([init_params['x1_dd'], init_params['x2_dd']], 
                     [init_params['alpha1'], init_params['alpha2']], 
                     linefmt='b-', markerfmt='bo', basefmt=' ', label='Two-delta')
    # Three-delta  
    ax_potential.stem([init_params['x1_td'], init_params['x2_td'], init_params['x3_td']],
                     [init_params['beta1'], init_params['beta2'], init_params['beta3']],
                     linefmt='r-', markerfmt='rs', basefmt=' ', label='Three-delta')
    ax_potential.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax_potential.set_xlabel('Position x')
    ax_potential.set_ylabel('Potential Strength')
    ax_potential.legend()
    ax_potential.grid(True, alpha=0.3)
    ax_potential.set_xlim(-6, 6)  # Wider x-range for positions
    ax_potential.set_ylim(-10, 10)  # Wider y-range for strengths
    
    # Add value labels
    label_offset = 1.0  # Larger offset
    
    for x, y, color in zip([init_params['x1_dd'], init_params['x2_dd']], 
                          [init_params['alpha1'], init_params['alpha2']], 
                          ['blue', 'blue']):
        va = 'bottom' if y > 0 else 'top'
        y_pos = y + label_offset if y > 0 else y - label_offset
        ax_potential.text(x, y_pos, f'{y:.2f}', 
                         ha='center', va=va, 
                         color=color, weight='bold', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    for x, y, color in zip([init_params['x1_td'], init_params['x2_td'], init_params['x3_td']],
                          [init_params['beta1'], init_params['beta2'], init_params['beta3']], 
                          ['red', 'red', 'red']):
        va = 'bottom' if y > 0 else 'top'
        y_pos = y + label_offset if y > 0 else y - label_offset
        ax_potential.text(x, y_pos, f'{y:.2f}', 
                         ha='center', va=va, 
                         color=color, weight='bold', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Difference plot
    line_diff, = ax_diff.plot(E_plot, diff, 'g-', linewidth=1.5)
    ax_diff.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax_diff.set_xlabel('Energy E')
    ax_diff.set_ylabel('Difference')
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_ylim(-1, 1)
    ax_diff.set_xlim(0, 10)  # Ensure x-axis only shows E < 10
    
    # Add error metrics
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    error_text = f'Max difference: {max_diff:.3f}\nRMS difference: {rms_diff:.3f}'
    error_text_obj = ax_diff.text(0.02, 0.98, error_text, transform=ax_diff.transAxes, 
                    va='top', ha='left', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Create sliders with constraints
    slider_y = 0.4
    slider_height = 0.02
    slider_spacing = 0.03
    
    # Two-delta strength sliders
    ax_alpha1 = plt.axes([0.25, slider_y, 0.65, slider_height])
    ax_alpha2 = plt.axes([0.25, slider_y - slider_spacing, 0.65, slider_height])
    
    # Two-delta position sliders with constraints
    ax_x1_dd = plt.axes([0.25, slider_y - 2*slider_spacing, 0.65, slider_height])
    ax_x2_dd = plt.axes([0.25, slider_y - 3*slider_spacing, 0.65, slider_height])
    
    # Three-delta strength sliders
    ax_beta1 = plt.axes([0.25, slider_y - 4*slider_spacing, 0.65, slider_height])
    ax_beta2 = plt.axes([0.25, slider_y - 5*slider_spacing, 0.65, slider_height])  
    ax_beta3 = plt.axes([0.25, slider_y - 6*slider_spacing, 0.65, slider_height])
    
    # Three-delta position sliders with constraints
    ax_x1_td = plt.axes([0.25, slider_y - 7*slider_spacing, 0.65, slider_height])
    ax_x2_td = plt.axes([0.25, slider_y - 8*slider_spacing, 0.65, slider_height])
    ax_x3_td = plt.axes([0.25, slider_y - 9*slider_spacing, 0.65, slider_height])
    
    # Create sliders
    sliders = {
        # Two-delta strengths
        'alpha1': Slider(ax_alpha1, 'α₁ (2-delta)', -10.0, 10.0, valinit=init_params['alpha1']),
        'alpha2': Slider(ax_alpha2, 'α₂ (2-delta)', -10.0, 10.0, valinit=init_params['alpha2']),
        
        # Two-delta positions with constraints
        'x1_dd': Slider(ax_x1_dd, 'x₁ (2-delta)', -6.0, 5.0, valinit=init_params['x1_dd']),
        'x2_dd': Slider(ax_x2_dd, 'x₂ (2-delta)', -5.0, 6.0, valinit=init_params['x2_dd']),
        
        # Three-delta strengths
        'beta1': Slider(ax_beta1, 'β₁ (3-delta)', -10.0, 10.0, valinit=init_params['beta1']),
        'beta2': Slider(ax_beta2, 'β₂ (3-delta)', -10.0, 10.0, valinit=init_params['beta2']),
        'beta3': Slider(ax_beta3, 'β₃ (3-delta)', -10.0, 10.0, valinit=init_params['beta3']),
        
        # Three-delta positions with constraints
        'x1_td': Slider(ax_x1_td, 'x₁ (3-delta)', -6.0, 3.5, valinit=init_params['x1_td']),
        'x2_td': Slider(ax_x2_td, 'x₂ (3-delta)', -3.5, 3.5, valinit=init_params['x2_td']),
        'x3_td': Slider(ax_x3_td, 'x₃ (3-delta)', -3.5, 6.0, valinit=init_params['x3_td'])
    }
    
    def enforce_constraints(params):
        """Enforce position constraints: x1 < x2 for 2-delta, x1 < x2 < x3 for 3-delta"""
        constrained_params = params.copy()
        
        # Two-delta constraint: x1_dd < x2_dd
        x1_dd, x2_dd = params['x1_dd'], params['x2_dd']
        if x1_dd >= x2_dd:
            # Move them apart by a small epsilon
            epsilon = 0.1
            constrained_params['x1_dd'] = min(x1_dd, x2_dd - epsilon)
            constrained_params['x2_dd'] = max(x1_dd + epsilon, x2_dd)
        
        # Three-delta constraint: x1_td < x2_td < x3_td
        x1_td, x2_td, x3_td = params['x1_td'], params['x2_td'], params['x3_td']
        positions = [x1_td, x2_td, x3_td]
        sorted_positions = sorted(positions)
        
        # If positions are not strictly increasing, enforce the constraint
        if positions != sorted_positions or len(set(positions)) < 3:
            epsilon = 0.1
            constrained_params['x1_td'] = sorted_positions[0]
            constrained_params['x2_td'] = sorted_positions[1] + epsilon
            constrained_params['x3_td'] = sorted_positions[2] + 2*epsilon
        
        return constrained_params
    
    def update_slider_limits():
        """Update slider limits based on current values to enforce constraints"""
        current_vals = {name: slider.val for name, slider in sliders.items()}
        
        # Two-delta constraints
        sliders['x1_dd'].ax.set_xlim(-6.0, current_vals['x2_dd'] - 0.1)
        sliders['x2_dd'].ax.set_xlim(current_vals['x1_dd'] + 0.1, 6.0)
        
        # Three-delta constraints  
        sliders['x1_td'].ax.set_xlim(-6.0, current_vals['x2_td'] - 0.1)
        sliders['x2_td'].ax.set_xlim(current_vals['x1_td'] + 0.1, current_vals['x3_td'] - 0.1)
        sliders['x3_td'].ax.set_xlim(current_vals['x2_td'] + 0.1, 6.0)
    
    def update(val):
        # Get current slider values
        params = {name: slider.val for name, slider in sliders.items()}
        
        # Enforce constraints
        constrained_params = enforce_constraints(params)
        
        # Update sliders if constraints were applied
        for name, val in constrained_params.items():
            if abs(val - params[name]) > 1e-10:  # Only update if changed significantly
                sliders[name].set_val(val)
        
        # Update slider limits
        update_slider_limits()
        
        # Recompute transmissions with constrained parameters
        T_DDDP, T_TDDP = compute_transmission(constrained_params)
        diff = T_DDDP - T_TDDP
        
        # Update plots
        line1.set_ydata(T_DDDP)
        line2.set_ydata(T_TDDP)
        line_diff.set_ydata(diff)
        
        # Update potential plot
        ax_potential.clear()
        
        # Plot two-delta in sorted order
        dd_positions = [(constrained_params['x1_dd'], constrained_params['alpha1']), 
                       (constrained_params['x2_dd'], constrained_params['alpha2'])]
        dd_positions.sort(key=lambda x: x[0])
        dd_x = [pos[0] for pos in dd_positions]
        dd_y = [pos[1] for pos in dd_positions]
        
        ax_potential.stem(dd_x, dd_y, 
                         linefmt='b-', markerfmt='bo', basefmt=' ', label='Two-delta')
        
        # Plot three-delta in sorted order
        td_positions = [
            (constrained_params['x1_td'], constrained_params['beta1']), 
            (constrained_params['x2_td'], constrained_params['beta2']), 
            (constrained_params['x3_td'], constrained_params['beta3'])
        ]
        td_positions.sort(key=lambda x: x[0])
        td_x = [pos[0] for pos in td_positions]
        td_y = [pos[1] for pos in td_positions]
        
        ax_potential.stem(td_x, td_y,
                         linefmt='r-', markerfmt='rs', basefmt=' ', label='Three-delta')
        
        ax_potential.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax_potential.set_xlabel('Position x')
        ax_potential.set_ylabel('Potential Strength')
        ax_potential.legend()
        ax_potential.grid(True, alpha=0.3)
        ax_potential.set_xlim(-6, 6)
        ax_potential.set_ylim(-10, 10)
        
        # Update value labels
        for x, y, color in zip(dd_x, dd_y, ['blue', 'blue']):
            va = 'bottom' if y > 0 else 'top'
            y_pos = y + label_offset if y > 0 else y - label_offset
            ax_potential.text(x, y_pos, f'{y:.2f}', 
                             ha='center', va=va, 
                             color=color, weight='bold', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        for x, y, color in zip(td_x, td_y, ['red', 'red', 'red']):
            va = 'bottom' if y > 0 else 'top'
            y_pos = y + label_offset if y > 0 else y - label_offset
            ax_potential.text(x, y_pos, f'{y:.2f}', 
                             ha='center', va=va, 
                             color=color, weight='bold', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Update error text
        max_diff = np.max(np.abs(diff))
        rms_diff = np.sqrt(np.mean(diff**2))
        error_text = f'Max difference: {max_diff:.3f}\nRMS difference: {rms_diff:.3f}'
        ax_diff.texts[0].remove()  # Remove old text
        ax_diff.text(0.02, 0.98, error_text, transform=ax_diff.transAxes, 
                    va='top', ha='left', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    for slider in sliders.values():
        slider.on_changed(update)
    
    # Initialize slider limits
    update_slider_limits()
    
    # Add reset button
    resetax = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    
    def reset(event):
        for name, slider in sliders.items():
            slider.reset()
        update_slider_limits()
    
    button.on_clicked(reset)
    
    # Add some preset buttons for interesting starting points
    preset_ax1 = plt.axes([0.1, 0.02, 0.15, 0.04])
    preset_btn1 = Button(preset_ax1, 'Preset: Symmetric')
    
    def load_preset1(event):
        # A symmetric configuration that satisfies constraints
        sliders['alpha1'].set_val(1.0)
        sliders['alpha2'].set_val(-1.0)
        sliders['x1_dd'].set_val(-1.0)
        sliders['x2_dd'].set_val(1.0)
        sliders['beta1'].set_val(0.8)
        sliders['beta2'].set_val(-0.5)
        sliders['beta3'].set_val(0.7)
        sliders['x1_td'].set_val(-1.5)
        sliders['x2_td'].set_val(0.0)
        sliders['x3_td'].set_val(1.5)
        update_slider_limits()
    
    preset_btn1.on_clicked(load_preset1)
    
    plt.show()
    return fig

# Run the interactive explorer
if __name__ == "__main__":
    create_isospectral_explorer_wide_range()