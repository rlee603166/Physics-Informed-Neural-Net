# generate_synthetic_data.py
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import h5py

# Define the physics ODE (we need to solve it forward)
def balance_ode(t, y, K, B, tau, m=70.0):
    x, x_dot = y
    # Use a simple history function: for t < tau, assume x(t-tau) = x(0)
    if t < tau:
        x_delayed = y[0]
    else:
        # This is a simplification; for a true DDE solver, you'd interpolate
        # For this test, we'll just use the current x as an approximation
        x_delayed = x
        
    x_ddot = -(B / m) * x_dot - (K / m) * x_delayed
    return [x_dot, x_ddot]

# Define our archetype subjects with KNOWN parameters
subjects = {
    'subject_S01': {'K': 800, 'B': 60, 'tau': 0.15, 'age': 30},  # Healthy Young
    'subject_S02': {'K': 1500, 'B': 40, 'tau': 0.25, 'age': 75}, # Stiff, Slow/Unstable
    'subject_S03': {'K': 1200, 'B': 90, 'tau': 0.10, 'age': 50},  # Stable Adult
}

# Simulation parameters
t_span = [0, 20]
t_eval = np.linspace(0, 20, 20 * 106) # 20 seconds at 106 Hz
initial_conditions = [0.01, 0] # Start with a small displacement

# Generate data and save to HDF5
with h5py.File('processed_data/batch_synthetic.h5', 'w') as f:
    for subject_id, params in subjects.items():
        sol = solve_ivp(
            fun=balance_ode,
            t_span=t_span,
            y0=initial_conditions,
            t_eval=t_eval,
            args=(params['K'], params['B'], params['tau'])
        )
        
        # Create HDF5 structure similar to your real data
        grp = f.create_group(subject_id)
        trial_grp = grp.create_group('trial_00')
        trial_grp.create_dataset('cop_x', data=sol.y[0])
        # For simplicity, we'll make cop_y a noisy version of x
        trial_grp.create_dataset('cop_y', data=sol.y[0] + np.random.normal(0, 0.001, size=sol.y[0].shape))

# Create the corresponding age CSV
age_data = {
    'user_id': [sid.split('_')[1] for sid in subjects.keys()],
    'age': [p['age'] for p in subjects.values()]
}
pd.DataFrame(age_data).to_csv('user_ages_synthetic.csv', index=False)

print("✅ Synthetic dataset created successfully.")
