# Memory configuration
import os
import glob
import json
import pickle
#os.environ['JAX_LOG_COMPILES'] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import matplotlib.pyplot as plt
from astropy.time import Time
import tqdm
import sys

jax.config.update("jax_enable_x64", True)

# Import gravitational wave functions
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import phase_marginalized_likelihood as likelihood_function_phase_marginalized
from jimgw.single_event.likelihood import original_likelihood as likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD

# Import custom BlackJAX nested sampling kernels
from custom_kernels import (
    acceptance_walk_sampler,
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical
)

# Initialize waveform
waveform = RippleIMRPhenomD(f_ref=50)

# Noise curve paths (using local files)
asd_paths = {
    "H1": "aLIGO_O4_high_asd.txt",
    "L1": "aLIGO_O4_high_asd.txt", 
    "V1": "AdV_asd.txt",
}

# Need to reorder sampled_config to ravel order
def get_ravel_order(particles_dict):
    """Determine the order that ravel_pytree uses for flattening."""
    example = jax.tree_util.tree_map(lambda x: x[0], particles_dict)
    flat, _ = jax.flatten_util.ravel_pytree(example)
    
    # Create a test dict with unique values to identify the order
    test_dict = {key: float(i) for i, key in enumerate(particles_dict.keys())}
    test_flat, _ = jax.flatten_util.ravel_pytree(test_dict)
    
    # The order is determined by the positions in the flattened array
    order = []
    for val in test_flat:
        for key, test_val in test_dict.items():
            if abs(val - test_val) < 1e-10:
                order.append(key)
                break
    return order

# Loop through all injection directories
injection_dirs = sorted(glob.glob("4s_injections/injection_*"), key=lambda x: int(x.split('_')[-1]))

start_index = 95
end_index = 100

selected_dirs = injection_dirs[start_index:end_index]

for i, injection_dir in enumerate(selected_dirs):
    print(f"\n=== Processing {injection_dir} ===")
    
    # Load injection parameters for this injection
    params_file = os.path.join(injection_dir, "injection_params.json")
    with open(params_file, 'r') as f:
        injection_params_raw = json.load(f)
    
    # Get injection time in format expected by prior
    #t_c = injection_params_raw["geocent_time"] - 1126259642.413 # center time corresponds to t=0.0s
    #print(f"Injection time: {injection_params_raw['geocent_time']}")
    # Convert to the format expected by the code
    injection_params = {
        "M_c": jnp.array(injection_params_raw["chirp_mass"]),
        "q": jnp.array(injection_params_raw["mass_ratio"]),
        "s1_z": jnp.array(injection_params_raw["chi_1"]),
        "s2_z": jnp.array(injection_params_raw["chi_2"]),
        "d_L": jnp.array(injection_params_raw["luminosity_distance"]),
        "iota": jnp.array(injection_params_raw["theta_jn"]),
        "t_c": jnp.array(0.0),
        "phase_c": jnp.array(injection_params_raw["phase"]),
        "ra": jnp.array(injection_params_raw["ra"]),
        "dec": jnp.array(injection_params_raw["dec"]),
        "psi": jnp.array(injection_params_raw["psi"]),
        "eta": jnp.array(injection_params_raw["mass_ratio"] / (1 + injection_params_raw["mass_ratio"]) ** 2),
    }

    # Load detector data as JAX arrays for this injection
    detector_data = {
        'frequencies': jnp.array(np.load(os.path.join(injection_dir, 'frequency_array.npy'))),
        'H1': jnp.array(np.load(os.path.join(injection_dir, 'H1_strain.npy'))),
        'L1': jnp.array(np.load(os.path.join(injection_dir, 'L1_strain.npy'))),
        'V1': jnp.array(np.load(os.path.join(injection_dir, 'V1_strain.npy')))
    }

    print(f"Loaded detector data as JAX arrays")

    # Configure detector frequency range
    freq_range = {'min': 20.0, 'max': 1024.0}

    freq_mask = (detector_data['frequencies'] >= freq_range['min']) & (detector_data['frequencies'] <= freq_range['max'])
    filtered_frequencies = detector_data['frequencies'][freq_mask]

    # Configure detectors
    detectors = [H1, L1, V1]
    detector_names = ['H1', 'L1', 'V1']

    for det, name in zip(detectors, detector_names):
        det.frequencies = filtered_frequencies  
        det.data = detector_data[name][freq_mask]

    print(f"Configured {len(detectors)} detectors with {len(filtered_frequencies)} frequency points")

    # Load PSD data for all detectors
    def load_psd_data(asd_paths):
        psd_data = {}
        for name, path in asd_paths.items():
            f_np, asd_vals_np = np.loadtxt(path, unpack=True)
            psd_data[name] = {
                'frequencies': jnp.array(f_np),
                'psd': jnp.array(asd_vals_np**2)
            }
        return psd_data

    psd_data = load_psd_data(asd_paths)

    @jax.jit
    def interpolate_psd(det_frequencies, psd_frequencies, psd_values):
        return jnp.interp(det_frequencies, psd_frequencies, psd_values)

    # Configure detector PSDs
    for det in detectors:
        det.psd = interpolate_psd(
            det.frequencies,
            psd_data[det.name]['frequencies'],
            psd_data[det.name]['psd']
        )

    print(f"Configured PSDs for {len(detectors)} detectors")

    # Define sampled parameters (excludes phase_c for phase marginalization)
    sample_keys = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "psi", "ra", "dec", "phase_c"]

    # Get sample_keys in correct ravel order
    test_particles = {}
    for i, key in enumerate(sample_keys):
        test_particles[key] = jax.random.uniform(jax.random.PRNGKey(42), 100)

    sample_keys = get_ravel_order(test_particles)

    # Parameter configuration
    param_config = {
        "M_c": {"min": 20.0, "max": 50.0, "prior": "uniform", "wraparound": False, "angle": 1.0},
        "q": {"min": 0.5, "max": 1.0, "prior": "uniform", "wraparound": False, "angle": 1.0},
        "s1_z": {"min": -0.8, "max": 0.8, "prior": "uniform", "wraparound": False, "angle": 1.0},
        "s2_z": {"min": -0.8, "max": 0.8, "prior": "uniform", "wraparound": False, "angle": 1.0},
        "iota": {"min": 0.0, "max": jnp.pi, "prior": "sine", "wraparound": False, "angle": 1.0},
        #"d_L": {"min": 100.0, "max": 5000.0, "prior": "beta", "wraparound": False, "angle": 1.0},
        "d_L": {"min": 100.0, "max": 5000.0, "prior": "powerlaw", "wraparound": False, "angle": 1.0},
        "t_c": {"min": 1126259642.313 - injection_params_raw["geocent_time"], "max": 1126259642.513 - injection_params_raw["geocent_time"], "prior": "uniform", "wraparound": False, "angle": 1.0},
        #"t_c": {"min": -0.1, "max": 0.1, "prior": "uniform", "wraparound": False, "angle": 1.0},
        "phase_c": {"min": 0.0, "max": 2*jnp.pi, "prior": "uniform", "wraparound": True, "angle": 2*jnp.pi},
        "psi": {"min": 0.0, "max": jnp.pi, "prior": "uniform", "wraparound": True, "angle": jnp.pi},
        "ra": {"min": 0.0, "max": 2*jnp.pi, "prior": "uniform", "wraparound": True, "angle": 2*jnp.pi},
        "dec": {"min": -jnp.pi/2, "max": jnp.pi/2, "prior": "cosine", "wraparound": False, "angle": 1.0},
    }

    print(f"t_c: {param_config['t_c']}")

    sampled_config = {key: param_config[key] for key in sample_keys}
    n_dims = len(sample_keys)

    # Pre-compute parameter arrays
    param_mins = jnp.array([sampled_config[key]["min"] for key in sample_keys])
    param_maxs = jnp.array([sampled_config[key]["max"] for key in sample_keys])
    param_prior_types = jnp.array([
        0 if sampled_config[key]["prior"] == "uniform" else
        1 if sampled_config[key]["prior"] == "sine" else
        2 if sampled_config[key]["prior"] == "cosine" else
        3 for key in sample_keys
    ])

    print(f"Sampling over {n_dims} parameters: {sample_keys}")
    print(f"Wraparound parameters: {[key for key in sample_keys if sampled_config[key]['wraparound']]}")

    # Constants for likelihood computation
    post_trigger_duration = 2
    duration = 4
    epoch = duration - post_trigger_duration
    print(f"Injection time: {injection_params_raw['geocent_time']}")
    gmst = Time(injection_params_raw["geocent_time"], format="gps").sidereal_time("apparent", "greenwich").rad
    #gmst = Time(1126259642.413, format="gps").sidereal_time("apparent", "greenwich").rad
    print(f"GMST: {gmst}")
    # Column labels for plotting
    column_to_label = {
        "M_c": r"$M_c$", "q": r"$q$", "d_L": r"$d_L$", "iota": r"$\iota$", "ra": r"$\alpha$",
        "dec": r"$\delta$", "s1_z": r"$s_{1z}$", "s2_z": r"$s_{2z}$", "t_c": r"$t_c$", 
        "psi": r"$\psi$", "phase_c": r"$\phi_c$",
    }

    # Vectorized prior transforms for unit cube [0,1] -> physical space
    @jax.jit
    def uniform_transform(u, a, b):
        return a + u * (b - a)

    @jax.jit  
    def sine_transform(u):
        return jnp.arccos(1 - 2 * u)

    @jax.jit
    def cosine_transform(u):  
        return jnp.arcsin(2 * u - 1)

    @jax.jit
    def beta_transform(u, a, b):
        """Beta(3,1) transform using analytical inverse CDF."""
        beta_sample = jnp.cbrt(u)  # Cube root for Beta(3,1)
        return a + (b - a) * beta_sample

    @jax.jit
    def powerlaw_transform(u, alpha, min, max):
        return (min ** (1+alpha) + u * (max ** (1+alpha) - min ** (1+alpha))) ** (1/(1+alpha))

    @jax.jit
    def prior_transform_fn(u_params):
        """Transform unit cube to physical parameters."""
        u_values, _ = jax.flatten_util.ravel_pytree(u_params)
        
        # Apply transforms based on prior type
        uniform_vals = uniform_transform(u_values, param_mins, param_maxs)
        sine_vals = sine_transform(u_values)
        cosine_vals = cosine_transform(u_values)  
        #beta_vals = beta_transform(u_values, param_mins, param_maxs)
        powerlaw_vals = powerlaw_transform(u_values, 2, param_mins, param_maxs)

        x_values = jnp.where(
            param_prior_types == 0, uniform_vals,
            jnp.where(
                param_prior_types == 1, sine_vals,
                jnp.where(
                    param_prior_types == 2, cosine_vals,
                    #beta_vals
                    powerlaw_vals
                )
            )
        )
        
        # Reconstruct parameter dict
        example_params = {key: 0.0 for key in sample_keys}
        _, unflatten_fn = jax.flatten_util.ravel_pytree(example_params)
        return unflatten_fn(x_values)

    def loglikelihood_fn(params):
        """Phase marginalized likelihood function."""
        p = injection_params.copy()
        p.update(params)
        p["gmst"] = gmst
        p["eta"] = p["q"] / (1 + p["q"]) ** 2
        
        waveform_sky = waveform(filtered_frequencies, p)
        align_time = jnp.exp(-1j * 2 * jnp.pi * filtered_frequencies * (epoch + p["t_c"]))
        return likelihood_function(p, waveform_sky, detectors, filtered_frequencies, align_time)

    # Vectorized prior functions for physical space (used in unit cube wrapper)
    @jax.jit
    def uniform_logprob(x, a, b):
        return jnp.where((x >= a) & (x <= b), -jnp.log(b - a), -jnp.inf)

    @jax.jit
    def sine_logprob(x):
        return jnp.where((x >= 0.0) & (x <= jnp.pi), jnp.log(jnp.sin(x) / 2.0), -jnp.inf)

    @jax.jit
    def cosine_logprob(x):
        return jnp.where(jnp.abs(x) < jnp.pi / 2, jnp.log(jnp.cos(x) / 2.0), -jnp.inf)

    @jax.jit
    def beta_logprob(x, a, b):
        u = (x - a) / (b - a)
        logpdf = (2.0 * jnp.log(u) + 0.0 * jnp.log(1 - u) - jax.scipy.special.betaln(3.0, 1.0) - jnp.log(b - a))
        return jnp.where((x >= a) & (x <= b), logpdf, -jnp.inf)

    @jax.jit
    def powerlaw_logprob(x, alpha, min, max):
        logpdf = alpha*jnp.log(x) + jnp.log(1+alpha) - jnp.log(max ** (1+alpha) - min ** (1+alpha))
        return jnp.where((x >= min) & (x <= max), logpdf, -jnp.inf)

    @jax.jit
    def logprior_fn(params):
        """Vectorized prior function using ravel_pytree."""
        param_values, _ = jax.flatten_util.ravel_pytree(params)
        
        uniform_priors = uniform_logprob(param_values, param_mins, param_maxs)
        sine_priors = sine_logprob(param_values)
        cosine_priors = cosine_logprob(param_values)
        #beta_priors = beta_logprob(param_values, param_mins, param_maxs)
        powerlaw_priors = powerlaw_logprob(param_values, 2, param_mins, param_maxs)

        priors = jnp.where(
            param_prior_types == 0, uniform_priors,
            jnp.where(
                param_prior_types == 1, sine_priors,
                jnp.where(
                    param_prior_types == 2, cosine_priors,
                    #beta_priors
                    powerlaw_priors
                )
            )
        )
        
        return jnp.sum(priors)

    # Setup for unit cube sampling
    n_live = 1400 # to match 1000 live point run in bilby
    n_delete = int(n_live * 0.5)
    #n_live = 500
    #n_delete = 1

    rng_key = jax.random.PRNGKey(i)
    rng_key, init_key = jax.random.split(rng_key, 2)

    # Create example parameter structure for initialization
    example_params = {key: 0.0 for key in sample_keys}

    # Initialize unit cube particles [0,1]^n
    unit_cube_particles = init_unit_cube_particles(init_key, example_params, n_live)

    # Create periodic mask for unit cube stepper
    periodic_mask = jax.tree_util.tree_map(lambda _: False, example_params)
    for key in sample_keys:
        if sampled_config[key]["wraparound"]:
            periodic_mask[key] = True

    # Create unit cube functions
    # unit_cube_fns = create_unit_cube_functions(
    #     physical_loglikelihood_fn=loglikelihood_fn,
    #     prior_transform_fn=prior_transform_fn,
    #     mask_tree=periodic_mask
    # )
    unit_cube_fns = create_unit_cube_functions(
        physical_loglikelihood_fn=loglikelihood_fn,
        prior_transform_fn=prior_transform_fn,
        mask_tree=periodic_mask
    )

    # Initialize unit cube nested sampler
    # nested_sampler = bilby_adaptive_de_sampler_unit_cube(
    #     logprior_fn=unit_cube_fns['logprior_fn'],
    #     loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],
    #     n_target=60,
    #     max_mcmc=5000,
    #     num_delete=n_delete,
    #     stepper_fn=unit_cube_fns['stepper_fn']
    # )
    nested_sampler = acceptance_walk_sampler(
        logprior_fn=unit_cube_fns['logprior_fn'],
        loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],
        nlive=n_live,
        n_target=60,
        max_mcmc=5000,
        num_delete=n_delete,
        stepper_fn=unit_cube_fns['stepper_fn'],
        max_proposals=1000
    )
    state = nested_sampler.init(unit_cube_particles)

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = nested_sampler.step(subk, state)
        return (state, k), dead_point

    def terminate(state):
        dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
        return jnp.isfinite(dlogz) and dlogz < 0.1

    # Run Nested Sampling
    dead = []
    with tqdm.tqdm(desc=f"Dead points for {os.path.basename(injection_dir)}", unit=" dead points") as pbar:
        #print("Starting Nested Sampling")
        #while not state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        while not state.logZ_live - state.logZ < jnp.log(0.001):
        #while not terminate(state): #same term condition as bilby
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)  # Update progress bar

    from blackjax.ns.utils import finalise
    from anesthetic import NestedSamples

    column_to_label = {
        "M_c": r"$M_c$",
        "q": r"$q$",
        "d_L": r"$d_L$",
        "iota": r"$\iota$",
        "ra": r"$\alpha$",
        "dec": r"$\delta$",
        "s1_z": r"$s_{1z}$",
        "s2_z": r"$s_{2z}$",
        "t_c": r"$t_c$",
        "psi": r"$\psi$",
        "phase_c": r"$\phi_c$",
    }

    final_state = finalise(state, dead)
    #os.makedirs(injection_dir+'/mix1', exist_ok=True)
    #with open(injection_dir+'/mix1/final_state_nlive1400.pkl', 'wb') as f:
    os.makedirs(injection_dir+'/priorloop1', exist_ok=True)
    with open(injection_dir+'/priorloop1/final_state_nlive1400.pkl', 'wb') as f:
        pickle.dump(final_state, f)

    # Transform unit cube particles back to physical space
    physical_particles = transform_to_physical(final_state.particles, prior_transform_fn)

    logL_birth = final_state.loglikelihood_birth.copy()
    logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)
    samples = NestedSamples(
        physical_particles,
        logL=final_state.loglikelihood,
        logL_birth=logL_birth,
        labels=column_to_label,
        logzero=jnp.nan,
        dtype=jnp.float64,
    )

    # Save to CSV in the injection directory
    injection_name = os.path.basename(injection_dir)
    #output_filename = os.path.join(injection_dir, f"nlive1400/results_nlive1400.csv")
    output_filename = os.path.join(injection_dir, f"priorloop1/results_nlive1400.csv")
    samples.to_csv(output_filename)

    # Save timings from progress bar
    #with open(injection_dir+'/nlive1400/timings_nlive1400.pkl', 'wb') as f:
    #    pickle.dump(pbar.format_dict, f)
    with open(injection_dir+'/priorloop1/timings_nlive1400.pkl', 'wb') as f:
        pickle.dump(pbar.format_dict, f)
    
    print(f"Results saved to {output_filename}")

print("\n=== All injections processed ===")
