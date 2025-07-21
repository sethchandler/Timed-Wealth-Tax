# app.py
# A Flask server to run the life-cycle model optimization.

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.optimize import minimize

# --- 1. Initialize Flask App and Enable CORS ---
# CORS is necessary to allow your Ghost blog (on a different domain)
# to make requests to this server.
app = Flask(__name__)
CORS(app)

# --- 2. Core Mathematical Functions from the Paper ---
# These are the direct Python translations of the model's equations.

def kappa(r, rho, gamma):
    return (r * (gamma - 1) + rho) / gamma

def U2(r, rho, gamma, T, A, B):
    # Guard against infeasible paths where terminal wealth is too high
    if A <= B * np.exp(-r * T):
        return -np.inf
    
    # Guard against taking a fractional power of a negative number
    base = A * np.exp(r * T) - B
    if base < 0:
        return -np.inf

    k = kappa(r, rho, gamma)
    
    # Handle the case where gamma is very close to 1 (log utility)
    if abs(1 - gamma) < 1e-9:
        gamma = 1.0 + 1e-9

    numer = (base**(1 - gamma)) * (1 - np.exp(-k * T)) * (k**(-gamma))
    denom = (1 - gamma) * ((np.exp(r * T) - np.exp((r - rho) * T / gamma))**(1 - gamma))
    
    result = numer / denom
    return result if np.isfinite(result) else -np.inf

def lifetimeU(w1, w2, p):
    # Guard against negative wealth inputs from the optimizer
    if w1 <= 0 or w2 <= 0:
        return -np.inf

    u1 = U2(p['r'], p['rho'], p['gamma'], p['t1'], p['w0'], w1)
    u2 = U2(p['r'], p['rho'], p['gamma'], p['t2'], w1 * (1 - p['tau']), w2)
    
    # Handle eta close to 1
    eta = p['eta']
    if abs(1 - eta) < 1e-9:
        eta = 1.0 + 1e-9
    u_bequest = p['beta'] * (w2**(1 - eta)) / (1 - eta)

    if not (np.isfinite(u1) and np.isfinite(u2) and np.isfinite(u_bequest)):
        return -np.inf

    return u1 + np.exp(-p['rho'] * p['t1']) * u2 + u_bequest

# --- 3. Define the API Endpoint ---
# This endpoint will receive parameters, run the optimization, and return the result.
@app.route('/optimize', methods=['POST'])
def optimize_model():
    try:
        # Get parameters from the incoming JSON request
        params = request.get_json()

        # The objective function for the minimizer (minimize negative utility)
        def objective_function(vars):
            w1, w2 = vars
            return -lifetimeU(w1, w2, params)

        # An initial guess to start the optimization
        initial_guess = [params.get('w0', 50000) * 1.5, params.get('w0', 50000) * 1.2]
        
        # Define bounds to ensure wealth is positive
        bounds = [(1e-6, None), (1e-6, None)]

        # Run the SciPy optimizer
        result = minimize(
            objective_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success:
            w1_opt, w2_opt = result.x
            # Return the results as a JSON object
            return jsonify({
                'success': True,
                'w1_opt': w1_opt,
                'w2_opt': w2_opt
            })
        else:
            return jsonify({'success': False, 'message': 'Optimization failed to converge.'}), 500

    except Exception as e:
        # Return an error message if anything goes wrong
        return jsonify({'success': False, 'message': str(e)}), 500

# --- 4. Health Check Endpoint ---
# A simple route to verify the server is running.
@app.route('/')
def index():
    return "Life-Cycle Model Optimization Server is running."

if __name__ == '__main__':
    # This allows running the app locally for testing
    app.run(debug=True)
