import pyCompressSensing as cs
from pyCompressSensing.l1min import *


sf = SignalFrame()
s01_uni = sf.read_wave('../signals_wav/C_01.wav')
s01_uni.sampler_uniform(rate=0.1)

signal_sampled_uni = s01_uni.temporal_sampled
phi_uni = s01_uni.phi

l1 = L1min()
s01_recovered_uni = l1.solver(signal_sampled_uni, phi_uni, w=300000,
                              max_iter=10000, plot=True, verbose=1)

#cos1 = sf.signal_gen(a=[2], f=[50], observation_time=1, noise_level=0, plot=True)

#cos1.sampler_gauss(rate=0.03)

# sf = SignalFrame()
# signal = sf.read_wave('../signals_wav/C_01.wav')
# signal.fft()
# signal.sampler_gauss(signal_length=0.3)
#
# signal_sampled = signal.temporal_sampled
# phi = signal.phi
#
# l1 = L1min()
# signal_freq_recovered = l1.solver(signal_sampled, phi)
#
# l1.plot_recovery(signal_freq_recovered)
# l1.plot_score(signal.temporal, signal_freq_recovered)