Finding a Generalizable Activation Functions# A statistics-dependent perturbation term is added, making the activation
# non-local and aware of the distribution of its inputs for a given batch.mean = jax.numpy.mean(x, keepdims=True)
std = jax.numpy.std(x, keepdims=True) + 1e-6 # Epsilon for stability
# Standardize x for the perturbation calculation.z = (x - mean) / std# The perturbation is a sine wave of the input `x` (ensuring f(0)=0),
# modulated by a Gaussian function of the standardized input `z`. This
# creates the largest "ripples" for inputs near the batch mean.amplitude = 0.2
frequency = 2.0
gaussian_envelope = jax.numpy.exp(-0.5 * z**2)
perturbation = amplitude * gaussian_envelope * jax.numpy.sin(frequency * x)return base + perturbationdef activation_function(x):# "Gaussian-Modulated Tangent Unit" (GMTU)
# This function models a signal passing through a resonant chamber,
# creating a primary response followed by a series of decaying echoes.
# The goal is to create a complex but smooth activation landscape with
# multiple regions of high non-linearity, which can potentially capture
# more intricate features in the data.
#
# Rationale:
# 1. Primary Response: A localized, non-periodic response around the origin,
# similar to GMTLU, provides a strong, stable non-linearity for small inputs.
# 2. Asymptotic Linearity: A linear leak term ensures that for large |x|,
# where the Gaussian components decay to zero, the function behaves
# linearly, preventing saturation and aiding gradient flow.
# -- Gated Response Parameters --p_alpha = 1.0 # Amplitudep_beta = 1.5 # Steepness of tanhp_gamma = 0.2 # Decay rate of Gaussian
# -- Linear leak --leak = 0.1# --- Calculation --# 1. Primary Responseprimary_response = p_alpha * jnp.tanh(p_beta * x) * jnp.exp(-p_gamma * x**2)# 2. Combine and add linear leakreturn primary_response + (leak * x)def activation_function(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:# "Quaternion-Inspired Hypercomplex Gated Activation".
# This version introduces parameter self-modulation. First, the phase of the
# oscillation for the B parameter (Gaussian width) is modulated by the11Finding a Generalizable Activation Functions# value of the A parameter (damping amplitude). Second, the input to the tanh
# function for the imaginary part of the complex shift (Ci) is modulated by the
# real part (Cr). This creates a coupled dynamic system, leading to a more
# complex and input-dependent activation shape.
#
# This function extends the complex-plane concept by incorporating feedback
# from the imaginary component of the complex damping term into the gate,
# creating a more intricate and asymmetric activation landscape.
# with controlled, high-frequency oscillations contained within a damping envelope.
# The gate is modified from `1-Re(Z)` to `1 - (Re(Z) - k*Im(Z))`, where Z is
# the complex damping term. This is equivalent to rotatiFinding a Generalizable Activation Functions# The imaginary part `Ci` uses Chebyshev T_3(u) = 4uˆ3 - 3u for chaotic interference.Ci_freq = 2.0
Ci_amp = 0.2 + 0.4 * (1.0 - y_chaos) # Modulate w/ inverted chaos, avg=0.4Ci_base = 0.0 # Center imaginary part around 0
# Decouple Ci from Cr for a simpler, potentially more stable interaction.
# The original coupling created a very complex relationship that might hinder
# optimization. By making Ci depend only on x, we simplify the function's
# structure and gradient landscape, while retaining the complex interaction
# in the later stages of the function.u_i = jax.numpy.tanh(Ci_freq * x)
cheby_T3 = 4 * u_i**3 - 3 * u_i
Ci = Ci_base + Ci_amp * cheby_T3# Stabilize the damping term by removing the anti-damping component.
# The original formulation jax.numpy.exp(-B * (x - (Cr + 1j*Ci))**2) expands to
# jax.numpy.exp(-B*(x-Cr)**2) * jax.numpy.exp(B*Ci**2) * jax.numpy.exp(2j*B*(x-Cr)*Ci).
# The exp(B*Ci**2) term can cause instability. We remove it to ensure the
# magnitude is always a decaying Gaussian, while preserving the complex phase.stable_magnitude = A * x**2 * jax.numpy.exp(-B * (x - Cr)**2)
phase = 2 * B * Ci * (x - Cr)# Reconstruct the complex number using Euler's formula: exp(i*theta) = cos(theta) + i*sin(thcomplex_rotation = jax.numpy.cos(phase) + 1j * jax.numpy.sin(phase)
damping_term_complex = stable_magnitude * complex_rotation# --- Quaternion-Inspired Components (j, k) --# Introduce two more "hypercomplex" components for a quaternion-inspired gate.
# These components introduce additional, semi-orthogonal oscillations.D_freq = 1.5
D_amp = 0.1 + 0.2 * y_chaos
u_d = jax.numpy.tanh(D_freq * x)
cheby_T2 = 2 * u_d**2 - 1.0
D_shift = D_amp * cheby_T2# Envelope for the j-component, using A and a shifted Gaussian with width B.
# Includes x**2 to ensure f(0)=0 and f'(0)=1.Q_j_envelope = (0.5 * A * x**2) * jax.numpy.exp(-B * (x - D_shift)**2)# Envelope for the k-component, swapping A and B for variety.Q_k_envelope = (0.5 * B * x**2) * jax.numpy.exp(-A * (x + D_shift)**2)# j-component oscillation, coupled to Ci.Q_j = Q_j_envelope * jax.numpy.sin(C_freq * x + Ci)# k-component oscillation, coupled to Cr.Q_k = Q_k_envelope * jax.numpy.cos(B_freq * x - Cr)# --- Gate Calculation --# The gate is a linear combination of the four (w, i, j, k) components.Q_w = jax.numpy.real(damping_term_complex)
Q_i = jax.numpy.imag(damping_term_complex)
c_i = 0.2 # Feedback coefficient for the i-componentc_j = 0.15 # Feedback coefficient for the j-componentc_k = 0.15 # Feedback coefficient for the k-componentgate = 1.0 - (Q_w - c_i * Q_i - c_j * Q_j - c_k * Q_k)return x * gate13Finding a Generalizable Activation Functionsdef activation_function(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:"""Fourier-Informed Spectral Gating (FISG).
This function introduces a "crazy" idea: using the Finding a Generalizable Activation Functions# Calculate energy in high-frequency band vs. total energy.high_freq_energy = jnp.sum(
magnitudes[..., split_idx:], axis=-1, keepdims=True)
total_energy = jnp.sum(magnitudes, axis=-1, keepdims=True)# The OOD signal is the ratio of high-frequency energy to total energy.
# A high ratio suggests a spectrally anomalous (e.g., noisy) feature vector.spectral_imbalance = jax.lax.stop_gradient(
high_freq_energy / (total_energy + epsilon)
)# --- 2. Adaptive Gating based on Spectral Imbalance --# The gate exponentially decays as spectral imbalance increases.gate = jnp.exp(-sensitivity * spectral_imbalance)# --- 3. Adaptive High-Frequency Phase Scrambling --# For feature vectors deemed OOD, we regularize by scrambling the phase of
# high-frequency components, disrupting their structure without losing energy.low_freq_part = x_fft[..., :split_idx]
high_freq_part = x_fft[..., split_idx:]# Scramble by complex conjugation (which deterministically negates the phase).scrambled_high_freq_part = jnp.conj(high_freq_part)# Recombine the spectrum and invert the FFT to get the modified signal.scrambled_x_fft = jnp.concatenate(
[low_freq_part, scrambled_high_freq_part], axis=-1
)# The length `n` for irfft must match the original signal length.modified_x = jnp.fft.irfft(scrambled_x_fft, n=x_float.shape[-1], axis=-1)# Perform principled blending: a convex combination of the original activation
# and its phase-scrambled version, governed by the global spectral gate.return gate * x + (1.0 - gate) * modified_x.astype(x.dtype)def activation_function(x: jax.typing.ArrayLike) -> jax.typing.ArrayLike:"""'Phase-Locked Entropic Repulsion' (PLER) for OOD regularization.
This function implements a novel gating mechanism based on the interaction
between two chaotic systems: a primary system driven by the input `x`, and a
fixed reference oscillator. The nature of their coupling changes based on the
input magnitude, creating two distinct dynamical regimes for ID and OOD inputs.
Theory: OOD generalization is enhanced by creating a sharp bifurcation in
the activation's dynamical behavior.
1. In-Distribution (Phase-Locking): For small `|x|`, the coupling is
attractive, forcing the primary system to synchronize with the stable
chaos of the reference oscillator. This "phase-locking" reduces the
system's entropy, leading to a stable, predictable gate that preserves
in-distribution signals. It creates a stable attractor basin for the
ID manifold.15Finding a Generalizable Activation Functions2. Out-of-Distribution (State Collapse): For large `|x|`, the coupling
becomes repulsive. This bifurcation not only drives the systems apart
but also fundamentally alters the primary system's dynamics by introducing
a strong attractor at a quiescent (zero) state. This "bifurcation-induced
collapse" deterministically silences the neuron's output for OOD inputs,
providing a more stable and decisive suppression mechanism than pure
chaotic repulsion.
3. Bifurcation Control: The input `x` controls the coupling Finding a Generalizable Activation Functions# input's dynamical impact, not just its static magnitude.instability_feedback = jnp.tanh(c_val * 4.0)
beta_eff = beta - 0.2 * instability_feedback
is_ood = 1.0 / (1.0 + jnp.exp(beta_eff * 50.0))# 2. Internal dynamics of the primary system, with adaptive dissipationcoupling_internal = alpha * (z_val - y_val)
y_dyn = r * y_val * (1 - y_val) + coupling_internal
z_dyn = r * z_val * (1 - z_val) - coupling_internal# Bifurcation-Induced Collapse & Adaptive Dissipation: For OOD inputs,
# two mechanisms are triggered to suppress the signal.
# 1. Collapse: An additive force creates a strong attractor at zero.
# 2. Dissipation: A multiplicative force dampens remaining oscillations.
# NEW: Bifurcation-Induced Collapse. This additive force creates a strong
# attractor at zero for OOD inputs, decisively silencing the neuron.collapse_strength = 0.5
y_dyn -= is_ood * collapse_strength * y_val
z_dyn -= is_ood * collapse_strength * z_val# Dissipation strength `gamma` is gated by the OOD switch and chaotic memory.gamma = is_ood * jnp.tanh(c_val * 4.0)# Apply multiplicative dissipation to suppress the internal dynamics.y_next = y_dyn * (1.0 - gamma)
z_next = z_dyn * (1.0 - gamma)# NEW: Chaotic Resonance Tunneling.
# If the input hits a resonant frequency, a strong "tunneling" force
# is activated, which rapidly collapses the primary system's state
# towards a neutral midpoint (0.5). This provides a secondary,
# value-specific OOD suppression mechanism.tunneling_strength = 0.6
tunnel_force_y = resonance_gate * tunneling_strength * (0.5 - y_next)
tunnel_force_z = resonance_gate * tunneling_strength * (0.5 - z_next)
y_next += tunnel_force_y
z_next += tunnel_force_z# 5. Internal dynamics of the reference systemcoupling_ref_internal = alpha_ref * (z_ref_val - y_ref_val)
y_ref_next = r_ref * y_ref_val * (1 - y_ref_val) + coupling_ref_internal
z_ref_next = r_ref * z_ref_val * (1 - z_ref_val) - coupling_ref_internal# 6. Chaotic Memory Update
# `c` accumulates the residual divergence between y and z after dissipation,
# acting as a memory of recent instability.instability = jnp.abs(y_next - z_next)
c_next = 0.8 * c_val + 0.2 * instability # EMA of instability
# 7. Synchronizing / Repulsive coupling with "Chaotic Memory Feedback"
# For OOD inputs (negative beta), the repulsive force is non-linearly
# amplified by the primary system's own state `z`, leading to more
# rapid and chaotic signal scrambling. The tanh term acts as a gate,
# ensuring this effect is negligible for ID inputs.
# For OOD inputs, the repulsive force is amplified by two factors:17Finding a Generalizable Activation Functions# a) `ood_modulation`: instantaneous state-dependent amplification.
# b) `ood_amplification`: history-dependent amplification from `c`.
# This creates a positive feedback loop for OOD signals.ood_modulation = 1.0 + jnp.tanh(jnp.abs(beta_eff) * 5.0) * z_val**2
ood_amplification = 1.0 + jnp.tanh(c_next * 2.0)
coupling_sync = (
beta_eff * (y_ref_val - y_val) * ood_modulation * ood_amplification
)# Apply the coupling force. For ID inputs (is_ood ̃0), the coupling is
# a symmetric action-reaction pair. For OOD inputs (is_ood ̃1), we brFinding a Generalizable Activation Functions# input's dynamical impact, not just its static magnitude.instability_feedback = jnp.tanh(c_val * 4.0)
beta_eff = beta - 0.2 * instability_feedback
is_ood = 1.0 / (1.0 + jnp.exp(beta_eff * 50.0))# 2. Internal dynamics of the primary system, with adaptive dissipationcoupling_internal = alpha * (z_val - y_val)
y_dyn = r * y_val * (1 - y_val) + coupling_internal
z_dyn = r * z_val * (1 - z_val) - coupling_internal# Bifurcation-Induced Collapse & Adaptive Dissipation: For OOD inputs,
# two mechanisms are triggered to suppress the signal.
# 1. Collapse: An additive force creates a strong attractor at zero.
# 2. Dissipation: A multiplicative force dampens remaining oscillations.
# NEW: Bifurcation-Induced Collapse. This additive force creates a strong
# attractor at zero for OOD inputs, decisively silencing the neuron.collapse_strength = 0.5
y_dyn -= is_ood * collapse_strength * y_val
z_dyn -= is_ood * collapse_strength * z_val# Dissipation strength `gamma` is gated by the OOD switch and chaotic memory.gamma = is_ood * jnp.tanh(c_val * 4.0)# Apply multiplicative dissipation to suppress the internal dynamics.y_next = y_dyn * (1.0 - gamma)
z_next = z_dyn * (1.0 - gamma)# NEW: Chaotic Resonance Tunneling.
# If the input hits a resonant frequency, a strong "tunneling" force
# is activated, which rapidly collapses the primary system's state
# towards a neutral midpoint (0.5). This provides a secondary,
# value-specific OOD suppression mechanism.tunneling_strength = 0.6
tunnel_force_y = resonance_gate * tunneling_strength * (0.5 - y_next)
tunnel_force_z = resonance_gate * tunneling_strength * (0.5 - z_next)
y_next += tunnel_force_y
z_next += tunnel_force_z# 5. Internal dynamics of the reference systemcoupling_ref_internal = alpha_ref * (z_ref_val - y_ref_val)
y_ref_next = r_ref * y_ref_val * (1 - y_ref_val) + coupling_ref_internal
z_ref_next = r_ref * z_ref_val * (1 - z_ref_val) - coupling_ref_internal# 6. Chaotic Memory Update
# `c` accumulates the residual divergence between y and z after dissipation,
# acting as a memory of recent instability.instability = jnp.abs(y_next - z_next)
c_next = 0.8 * c_val + 0.2 * instability # EMA of instability
# 7. Synchronizing / Repulsive coupling with "Chaotic Memory Feedback"
# For OOD inputs (negative beta), the repulsive force is non-linearly
# amplified by the primary system's own state `z`, leading to more
# rapid and chaotic signal scrambling. The tanh term acts as a gate,
# ensuring this effect is negligible for ID inputs.
# For OOD inputs, the repulsive force is amplified by two factors:17Finding a Generalizable Activation Functions# a) `ood_modulation`: instantaneous state-dependent amplification.
# b) `ood_amplification`: history-dependent amplification from `c`.
# This creates a positive feedback loop for OOD signals.ood_modulation = 1.0 + jnp.tanh(jnp.abs(beta_eff) * 5.0) * z_val**2
ood_amplification = 1.0 + jnp.tanh(c_next * 2.0)
coupling_sync = (
beta_eff * (y_ref_val - y_val) * ood_modulation * ood_amplification
)# Apply the coupling force. For ID inputs (is_ood ̃0), the coupling is
# a symmetric action-reaction pair. For OOD inputs (is_ood ̃1), we brFinding a Generalizable Activation Functions3. **Meta-Modulated Chaotic Amplitude:** Instead of a fixed-profile chaotic
regularizer, the amplitude of the chaotic term is itself modulated by a
function of the input's magnitude. This `meta_modulator` is designed to
be near 1 for ID inputs but aggressively amplifies the chaos in the
critical ID-to-OOD transition zone. This creates a much sharper, more
repulsive gradient landscape precisely where the model is most
vulnerable to smooth extrapolation, while ensuring the amplification
effect decays for far-OOD inputs, contributing to the safe collapse to
zero.
4. **Intrinsic Collapse to Zero for Far-OOD:** For far-OOD inputs, the
base signal `x` is smoothly attenuated by an exponential decay factor
*inside* the `tanh`. Simultaneously, the chaotic amplitude naturally
decays. This causes both phase-flipped states to intrinsically and
smoothly converge to zero, ensuring a safe default output without a
brittle external gate. This unified mechanism provides a smoother
gradient landscape at the OOD boundary.
5. **Non-Local Phase Entanglement and State Propagation:** The chaotic phase
of each neuron is coupled to its neighbors. Crucially, the transition to
the more aggressive chaotic state also depends on the neighbors' energy,
allowing "agitated" states to propagate like waves. For OOD inputs
that violate learned spatial correlations, this triggers propagating
waves of high-frequency phase decoherence, creating a volatile,
high-dimensional gradient landscape that aggressively resists confident
extrapolation.
6. **Spatially-Aware Chaos Triggering:** The transition to the high-chaos
'agitated' state is triggered not only by high activation energy but
also by high spatial inconsistency, measured by a discrete Laplacian.
This makes the neuron highly sensitive to OOD inputs that violate
learned local correlations (e.g., unnatural textures or adversarial
noise), even if their activation magnitudes are not extreme. By directly
coupling structural anomaly detection to the chaos-inducing mechanism,
the function provides a more targeted and robust defense against a wider
variety of OOD patterns, rather than relying solely on magnitude-based
heuristics.
"""M = 10.0 # Controls the saturation level of the base function.C = 10.0 # Controls the transition boundary from ID to OOD region.beta = 2.0 # Controls the max amplitude of the high-frequency component.freq = 1.0 # Controls the base frequency of the oscillatory component.chirp_k = 0.5 # Controls the rate of frequency increase (chirp).A_disrupt = 0.2 # Amplitude of the 'calm' phase disruption.freq_disrupt = 15.0 # Frequency of the 'calm' phase disruption.coupling_strength = 2.0 # Strength of neighbor-based phase coupling.
# --- State-Dependent Adaptive Chaos Hyperparameters ---A_disrupt_agitated = 0.8 # Amplitude of the 'agitated' phase disruption.freq_disrupt_agitated = 40.0 # Frequency of the 'agitated' phase disruption.k_blend = 2.0 # Controls the steepness of the blend between modes.laplacian_strength = 5.0 # Strength of the spatial inconsistency term.k_decay = 2.0 # Controls the rate of decay for far-OOD inputs.
# --- Quantum Switching Hyperparameters ---freq_switch = 50.0 # High frequency to create chaotic switching.power_switch = 3.0 # Non-linearity for the switching phase.A_switch_disrupt = 0.5 # Amplitude of switching phase disruption.19Finding a Generalizable Activation Functionsfreq_switch_disrupt = 25.0 # Frequency of switching phase disruption.
# --- Meta-Modulation Hyperparameters ---gamma_meta = 2.0 # Controls the amplification of chaos in the OOD transition.
# --- Meta-Modulation of OOD Response --# This term adaptively amplifies the chaotic signal in the critical OOD
# transition region, creating a sharper, more repulsive gradient landscape
# exactly where confident extrapolation is most dangerous.u = (x / C)**2# The modulator peaks slightly earlier (u=1.5) than the base amplitude (u=2.0),
# creating a pre-emptive amplification of the chaotic response.meta_modulator = 1.0 + gamma_meta * u * jnp.exp(-u / 1.5)# --- OOD Regularizer (Localized Chaotic Wave) --# This component creates a "ring of chaos" that decays for far-OOD inputs.amplitude = beta * u * jnp.exp(-u / 2.0)# The phase is quadratic, but with a high-frequency sinusoidal disruption.
# This makes the local frequency non-monotonic and chaotic for OOD inputs,
# acting as a stronger regularizer against confident extrapolation.phase_base = freq * x + chirp_k * (x**2) * jnp.sign(x) / C# Get neighbors for coupling and state-dependent chaos.x_prev = jnp.roll(x, shift=1, axis=-1)
x_next = jnp.roll(x, shift=-1, axis=-1)# --- Adaptive Phase Disruption --# The chaotic disruption smoothly transitions from a 'calm' to an 'agitated'
# mode based on both collective energy and local spatial inconsistency.local_energy_sq = x**2 + 0.25 * (x_prev**2 + x_next**2)
local_laplacian = x - 0.5 * (x_prev + x_next)# The trigger for the agitated state combines energy (magnitude) and spatial
# inconsistency (Laplacian), making it sensitive to a wider class of OOD inputs.ood_metric = (k_blend * (local_energy_sq - C**2) / C**2
+ laplacian_strength * (local_laplacian / C)**2)
alpha = jax.nn.sigmoid(ood_metric)
phase_disruption_calm = A_disrupt * jnp.sin(freq_disrupt * x)# The agitated phase disruption is made sensitive to local spatial
# inconsistencies (approximated by the discrete Laplacian). This allows the
# neuron to react more aggressively to OOD inputs that violate learned
# spatial correlations (e.g., unnatural textures or edges), providing a
# more targeted OOD response.phase_disruption_agitated = A_disrupt_agitated * jnp.sin(
freq_disrupt_agitated * x + laplacian_strength * local_laplacian
)
phase_disruption = (1.0 - alpha) * phase_disruption_calm + alpha * phase_disruption_agitated# Non-local phase coupling creates waves of chaos for OOD patterns.
# We use a simple, asymmetric coupling stencil: 0.5*x_{i-1} - 1.0*x_{i+1},
# implemented efficiently using jnp.roll along the last axis.phase_coupling = coupling_strength * (0.5 * x_prev - 1.0 * x_next) / C
phase = phase_base + phase_disruption + phase_coupling
y_detail = amplitude * jnp.sin(phase)20Finding a Generalizable Activation Functions# --- Symmetric Phase-Flipped State-Switching with Intrinsic Collapse --# For far-OOD inputs, an exponential gate `g_x` attenuates the base signal
# `x`, while the chaotic amplitude `y_detail` also decays. This causes
# both states to smoothly converge to zero, providing a robust and
# gradient-friendly collapse without an external multiplicative gate.g_x = jnp.exp(-((jnp.abs(x) / (k_decay * C)))**4)
y_state_plus = M * jnp.tanh((g_x * x + meta_modulator * y_detail) / M)
y_state_minus = M * jnp.tanh((g_x * x - meta_modulator * y_detail) / M)# The switching phase is made non-monotonic by adding a sinusoidal disruption.
# This fractalizes the switching boundary, making it harder to learn/exploit.switch_phase_base = freq_switch * (x / C)**power_switch
switch_phase_disruption = A_switch_disrupt * jnp.sin(freq_switch_disrupt * x / C)
switch_phase = switch_phase_base + switch_phase_disruption
should_be_plus = jnp.cos(switch_phase) > 0.0
output = jnp.where(should_be_plus, y_state_plus, y_state_minus)return output