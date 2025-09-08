import bilby
import json
import os
import numpy as np

from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.setup_logger()

outdir = "bilby_4s_powerlawprior"
label = "aligned_spin_injections"

bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(88170235)

detectors = ["H1", "L1", "V1"]

injection_parameters = dict(
    chirp_mass=35.0,
    mass_ratio=0.9,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0,
    tilt_2=np.pi,
    chi_1=0.4,
    chi_2=-0.3,
    luminosity_distance=1000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)


maximum_frequency = 1024
minimum_frequency = 20
sampling_frequency = 2048

duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = injection_parameters["geocent_time"] + post_trigger_duration
start_time = end_time - duration

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomD",
    reference_frequency=50.0,
    minimum_frequency=20.0,
    maximum_frequency=maximum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
#ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

#ifos = bilby.gw.detector.InterferometerList.from_pickle('./ifo_newparams_H1L1V1.pkl')
ifos = bilby.gw.detector.InterferometerList.from_pickle('./4s_H1L1V1.pkl')

#freqs = np.load('injected_frequency_array.npy')
#H1_strain = np.load('injected_H1_strain.npy')
#L1_strain = np.load('injected_L1_strain.npy')
#V1_strain = np.load('injected_V1_strain.npy')

#strain_lookup = {
#    'H1': H1_strain,
#    'L1': L1_strain,
#    'V1': V1_strain
#}

for ifo in ifos:
#    ifo.set_strain_data_from_frequency_domain_strain(frequency_domain_strain=strain_lookup[ifo.name], frequency_array=freqs)
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency

#np.save("debug_frequency_array.npy", ifos[0].frequency_array)
#np.save("debug_H1_data.npy", ifos[0].strain_data.frequency_domain_strain)
#np.save("debug_L1_data.npy", ifos[1].strain_data.frequency_domain_strain)
#np.save("debug_V1_data.npy", ifos[2].strain_data.frequency_domain_strain)

#injection_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(36, 29)
#injection_parameters["mass_ratio"] = bilby.gw.conversion.component_masses_to_mass_ratio(36, 29)
        
priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
priors["chirp_mass"] = bilby.core.prior.Uniform(
    25.0, 50.0, name="chirp_mass"
)
priors["mass_ratio"] = bilby.core.prior.Uniform(
    0.25, 1.0, name="mass_ratio"
)
# Set the prior on trigger time
priors["geocent_time"] = bilby.core.prior.Uniform(
    injection_parameters["geocent_time"] - 0.1, injection_parameters["geocent_time"] + 0.1, name="geocent_time"
)
#priors["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(minimum=100.0, maximum=5000.0, cosmology='Planck15', name='luminosity_distance')
priors['luminosity_distance'] = bilby.core.prior.analytical.Beta(alpha=3.0, beta=1.0, minimum=100.0, maximum=5000.0, name='luminosity_distance')
#priors["luminosity_distance"] = bilby.core.prior.analytical.PowerLaw(alpha=2.0, minimum=100.0, maximum=5000.0, name='luminosity_distance')
priors['chi_1'] = bilby.core.prior.Uniform(minimum=-1.0, maximum=1.0, name='chi_1')
priors['chi_2'] = bilby.core.prior.Uniform(minimum=-1.0, maximum=1.0, name='chi_2')
priors['mass_1'] = bilby.core.prior.Constraint(minimum=1, maximum=1000, name='mass_1')
priors['mass_2'] = bilby.core.prior.Constraint(minimum=1, maximum=1000, name='mass_2')

#priors['psi'] = bilby.core.prior.Uniform(minimum=0, maximum=np.pi, name='psi', unit=None, boundary='periodic')
#priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phase', unit=None, boundary='periodic')

#for key in [
#    "chi_1",
#    "chi_2",
#    "psi",
#    "ra",
#    "dec",
#    "geocent_time",
#    "phase",
#    "mass_ratio",
#    "chirp_mass",
#    "luminosity_distance",
#    "theta_jn",

#]:
#    priors[key] = injection_parameters[key]

# Define the likelihood
# Distance marginalization is always used in LVK analyses
# This further reduces the space to 10 parameters.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifos,
    waveform_generator,
    priors=priors,
    time_marginalization=False,
    phase_marginalization=False,
    distance_marginalization=False,
)

# Run dynesty
# This uses the acceptance walk implemented in bilby
result = bilby.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    outdir=outdir,
    label=label,
    nlive=1000,
    naccept=60,
    check_point_plot=True,
    plot=True,
    check_point_delta_t=1800,
    use_ratio=True,
    print_method="interval-60",
    sample="acceptance-walk",
    npool=16,
    do_clustering=False,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)
