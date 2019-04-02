from pyCompressSensing.signal_frame import *
from pyCompressSensing.l1min import *

sf = SignalFrame()
signal = sf.read_wave('../signals_wav/C_01.wav')
signal.fft()
signal.sampler_gauss(signal_length=0.3)

signal_sampled = signal.temporal_sampled
phi = signal.phi

l1 = L1min()
signal_freq_recovered = l1.solver(signal_sampled, phi)

#l1.plot_recovery(signal_freq_recovered)
l1.plot_score(signal.temporal, signal_freq_recovered)

# signal.plot_score(signal_recovered)

# cs = CompressSensing()
# signal_cos1 = cs.periodic_signal_maker(plot=True)
# sampling_instants_cos1, phi_cos1 = signal_cos1.sampler_gauss()
# signal_cos1.fft()
# signal_cos1_recovered, *_ = signal_cos1.recover(sampling_instants_cos1, phi_cos1)
# signal_cos1.plot_score(signal_cos1_recovered)

# cs = CompressSensing()
#
# A = [15, 2, 5, 3, 10]
# f = [50, 100, 150, 200, 300]
#
# cos2 = cs.periodic_signal_maker(A0=A, f0=f, fe=2000, t=1, noise_level=0, plot=False)
# instants_cos2, phi_cos2 = cos2.sampler_gauss()
# cos2.fft()
# cos2_recovered, *_ = cos2.recover(instants_cos2, phi_cos2, w=10.48)

# cs = CompressSensing()
# # cos2.plot_score(cos2_recovered)
#
# A = [15, 2, 8, 3, 10]
# f = [50, 100, 150, 200, 300]
#
# cos3 = cs.periodic_signal_maker(A0=A, f0=f, fe=2000, t=1, noise_level=1, plot=True)
