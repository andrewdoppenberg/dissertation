"""
This script generates and plots random Beta distributions to illustrate varied 
strengths of prior evidence. In the Zollman effect and Fazelpour models, agents 
start with Beta-distributed prior beliefs about the success rates of different theories/arms.
The script visualizes two distributions with different total prior pseudo-observations 
(S = alpha + beta) to demonstrate how the shape and variance of the prior changes 
depending on the quantity of initial evidence. The resulting figure is saved 
as a PDF in the figures directory.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, '../figures')

# Ensure the figures directory exists
os.makedirs(figures_dir, exist_ok=True)

# Font settings for Garamond size 18
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Garamond", "EB Garamond", "serif"],
    "font.size": 12
})

# Parameters consistent with previous request
a1, b1 = np.random.uniform(1, 10, size=2)
a2, b2 = np.random.uniform(1, 100, size=2)

x = np.linspace(0, 1, 5000)
y1 = beta.pdf(x, a1, b1)
y2 = beta.pdf(x, a2, b2)

plt.figure(figsize=(5.5, 3.5))
plt.plot(x, y1, label='S = 10', linewidth=1)
plt.plot(x, y2, label='S = 100', linewidth=1)

plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.tight_layout()

# Save to the specific relative path
plt.savefig(os.path.join(figures_dir, 'random_beta_distributions.pdf'))