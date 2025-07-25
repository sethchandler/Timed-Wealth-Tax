# app.py
# A Flask server to run the life-cycle model optimization.
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import numpy as np
from scipy.optimize import minimize
import os

# --- 1. Initialize Flask App and Enable CORS ---
app = Flask(__name__)
CORS(app)

# --- 2. Core Mathematical Functions from the Paper ---
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

# --- 3. HTML Template ---
# Embed the HTML directly in the Python file for easier deployment
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Life-Cycle Savings Model</title>
    
    <!-- External Libraries -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>

    <style>
        body { font-family: 'Inter', sans-serif; }
        .katex { font-size: 1.1em; }
        .slider-container { display: grid; grid-template-columns: 40px 1fr 60px; gap: 1rem; align-items: center; }
    </style>
</head>
<body class="bg-gray-50 text-gray-800">

    <div class="container mx-auto p-4 md:p-8">
        <div class="bg-white p-8 rounded-xl shadow-lg">
            
            <div class="text-center border-b pb-4 mb-8">
                <h1 class="text-4xl font-semibold text-gray-900">A Life-Cycle Consumption-Bequest Model with a One-Time Wealth Tax</h1>
                <p class="text-lg text-gray-600 mt-2">By Seth J. Chandler</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                <div class="lg:col-span-1">
                    <div class="sticky top-8">
                        <h2 class="text-2xl font-semibold mb-4">Model Parameters</h2>
                        <div id="controls" class="space-y-4"></div>
                    </div>
                </div>

                <div class="lg:col-span-2">
                    <div class="prose max-w-none">
                        <section id="introduction">
                            <h3>1. Introduction</h3>
                            <p>We study a continuous-time life-cycle problem in which an individual derives utility from consumption over two phases and from a terminal bequest. A proportional wealth tax is levied at the end of the first phase. The agent chooses optimal wealth <em>just before the tax</em> ($w_1$) and the final bequest level ($w_2$).</p>
                            <p>This model is motivated by real-world scenarios where households face significant financial consequences based on their declared assets at a specific point in time. A primary example is the asset test for determining eligibility for government aid programs, such as the Pell Grant for higher education in the United States. In such cases, the reduction in aid can be seen as a de facto tax on savings. This paper uses a one-time wealth tax, $\\tau$, as a formal analogue to model this phenomenon and determine the optimal financial strategy for an individual facing such a disincentive to save.</p>
                        </section>
                        
                        <section id="interactive-plots" class="mt-8">
                            <h3 class="text-2xl font-semibold mb-4">Interactive Results</h3>
                            <div class="bg-gray-100 p-4 rounded-lg mb-4 text-center">
                                <p>Optimal Pre-Tax Wealth ($w_1^*$): <strong id="w1_result" class="text-blue-600">Calculating...</strong></p>
                                <p>Optimal Bequest ($w_2^*$): <strong id="w2_result" class="text-blue-600"></strong></p>
                            </div>
                            <div id="wealth-plot" class="w-full h-96"></div>
                            <div id="consumption-plot" class="w-full h-96 mt-4"></div>
                        </section>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Use relative URL for same-origin requests
            const backendUrl = '/optimize';

            const params = {
                r:    { min: 0.00, max: 0.10, step: 0.001, value: 0.03, label: 'r (interest rate)' },
                rho:  { min: 0.00, max: 0.10, step: 0.001, value: 0.04, label: '\\\\rho (impatience)' },
                gamma:{ min: 1.1,  max: 5.0,  step: 0.1,   value: 2.0,  label: '\\\\gamma (cons. risk aversion)' },
                eta:  { min: 1.1,  max: 5.0,  step: 0.1,   value: 2.0,  label: '\\\\eta (beq. risk aversion)' },
                t1:   { min: 1,    max: 40,   step: 1,     value: 20,   label: 't_1 (phase 1 length)' },
                t2:   { min: 1,    max: 50,   step: 1,     value: 30,   label: 't_2 (phase 2 length)' },
                beta: { min: 0,    max: 50,   step: 1,     value: 10,   label: '\\\\beta (bequest weight)' },
                tau:  { min: 0,    max: 0.9,  step: 0.01,  value: 0.30, label: '\\\\tau (tax rate)' },
                w0:   { min: 1000, max: 100000, step: 1000, value: 50000, label: 'w_0 (initial wealth)' },
            };

            const controlsContainer = document.getElementById('controls');
            Object.keys(params).forEach(key => {
                const p = params[key];
                const controlEl = document.createElement('div');
                controlEl.innerHTML = `
                    <label for="slider-${key}" class="block text-sm font-medium text-gray-700 katex-label">${p.label}</label>
                    <div class="slider-container mt-1">
                        <span class="katex-symbol">${key}</span>
                        <input type="range" id="slider-${key}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
                        <input type="number" id="value-${key}" value="${p.value}" class="w-full text-center border-gray-300 rounded-md shadow-sm">
                    </div>`;
                controlsContainer.appendChild(controlEl);
                document.getElementById(`slider-${key}`).addEventListener('input', (e) => {
                    document.getElementById(`value-${key}`).value = e.target.value;
                    update();
                });
                document.getElementById(`value-${key}`).addEventListener('change', (e) => {
                    document.getElementById(`slider-${key}`).value = e.target.value;
                    update();
                });
            });

            renderMathInElement(document.body, { delimiters: [{left: "$", right: "$", display: false}] });
            
            let debounceTimer;
            async function update() {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(async () => {
                    const w1ResultEl = document.getElementById('w1_result');
                    const w2ResultEl = document.getElementById('w2_result');
                    w1ResultEl.textContent = 'Calculating...';
                    w2ResultEl.textContent = '';

                    const currentParams = {};
                    Object.keys(params).forEach(key => {
                        currentParams[key] = parseFloat(document.getElementById(`slider-${key}`).value);
                    });

                    try {
                        const response = await fetch(backendUrl, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(currentParams)
                        });

                        if (!response.ok) {
                            throw new Error(`Server error: ${response.statusText}`);
                        }

                        const data = await response.json();
                        if (!data.success) {
                            throw new Error(`Optimization failed: ${data.message}`);
                        }
                        
                        const { w1_opt, w2_opt } = data;
                        w1ResultEl.textContent = w1_opt.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
                        w2ResultEl.textContent = w2_opt.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
                        
                        plotPaths(w1_opt, w2_opt, currentParams);

                    } catch (error) {
                        console.error("Error fetching or processing data:", error);
                        w1ResultEl.textContent = 'Error';
                        w2ResultEl.textContent = 'See console for details.';
                    }
                }, 250);
            }

            function plotPaths(w1_opt, w2_opt, p) {
                const kappa = (r, rho, gamma) => (r * (gamma - 1) + rho) / gamma;
                const k = kappa(p.r, p.rho, p.gamma);

                const c01 = k * (Math.exp(p.r * p.t1) * p.w0 - w1_opt) / (Math.exp(p.r * p.t1) - Math.exp((p.r - p.rho) * p.t1 / p.gamma));
                const c02 = k * (Math.exp(p.r * p.t2) * w1_opt * (1 - p.tau) - w2_opt) / (Math.exp(p.r * p.t2) - Math.exp((p.r - p.rho) * p.t2 / p.gamma));

                const t_vals1 = Array.from({length: 101}, (_, i) => i * p.t1 / 100);
                const t_vals2 = Array.from({length: 101}, (_, i) => p.t1 + i * p.t2 / 100);

                const w_path1 = t => Math.exp(p.r*t)*p.w0 - c01*Math.exp(p.r*t)*(1-Math.exp(-k*t))/k;
                const w_path2 = t => Math.exp(p.r*(t-p.t1))*w1_opt*(1-p.tau) - c02*Math.exp(p.r*(t-p.t1))*(1-Math.exp(-k*(t-p.t1)))/k;
                const c_path1 = t => c01 * Math.exp((p.r - p.rho) / p.gamma * t);
                const c_path2 = t => c02 * Math.exp((p.r - p.rho) / p.gamma * (t - p.t1));
                
                const plotLayout = { margin: { l: 60, r: 20, t: 40, b: 50 }, xaxis: { title: 'Time (years)' }, legend: { x: 1, xanchor: 'right', y: 1 } };

                Plotly.newPlot('wealth-plot', [
                    { x: t_vals1, y: t_vals1.map(w_path1), type: 'scatter', mode: 'lines', name: 'Phase 1 Wealth', line: { color: 'royalblue' } },
                    { x: t_vals2, y: t_vals2.map(w_path2), type: 'scatter', mode: 'lines', name: 'Phase 2 Wealth', line: { color: 'firebrick' } },
                    { x: [p.t1, p.t1], y: [w_path2(p.t1), w_path1(p.t1)], type: 'scatter', mode: 'lines', name: 'Tax Event', line: { color: 'gray', dash: 'dash' } }
                ], { ...plotLayout, title: 'Optimal Wealth Path', yaxis: { title: 'Wealth ($)' } });

                Plotly.newPlot('consumption-plot', [
                     { x: t_vals1, y: t_vals1.map(c_path1), type: 'scatter', mode: 'lines', name: 'Phase 1 Consumption', line: { color: 'royalblue' } },
                     { x: t_vals2, y: t_vals2.map(c_path2), type: 'scatter', mode: 'lines', name: 'Phase 2 Consumption', line: { color: 'firebrick' } },
                     { x: [p.t1, p.t1], y: [c_path2(p.t1), c_path1(p.t1)], type: 'scatter', mode: 'lines', name: 'Tax Event', line: { color: 'gray', dash: 'dash' } }
                ], { ...plotLayout, title: 'Optimal Consumption Path', yaxis: { title: 'Consumption ($ per year)' } });
            }

            update();
        });
    </script>
</body>
</html>"""

# --- 4. Route to serve the main HTML page ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# --- 5. API Endpoint for optimization ---
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
        max_w1 = w0 * np.exp(r * t1)
        max_w2 = max_w1 * (1 - tau) * np.exp(r * t2)
        
        # Lower bounds: small positive values to avoid numerical issues
        min_wealth = min(10, w0 * 0.001)  # At least $100 or 0.1% of initial wealth
        
        bounds = [
            (min_wealth, max_w1),     # w1 bounds
            (min_wealth, max_w2) ]     # w2 bounds

        
        #bounds = [(1e-6, None), (1e-6, None)]
        
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

if __name__ == '__main__':
    # For production, bind to the PORT environment variable that Render provides
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
