# from pyCompressSensing.SignalFrame import SignalFrame
from pyCompressSensing.l1min import *
import os

os.chdir('/Users/colin/Documents/Etude/MSBGD/projet_Safran/compress_sensing_toolbox')

sf = SignalFrame()
s01_uni = sf.read_wave('signals_wav/C_01.wav', coeff_amplitude=1/10000)
s01_uni.sampler_uniform(rate=1, plot=False)

signal_sampled_uni = s01_uni.temporal_sampled
phi_uni = s01_uni.phi

l1 = L1min()
s01_recovered_uni = l1.solver(signal_sampled_uni, phi_uni, w=0.01, max_iter=5000, plot=True, verbose=1, lambda_n=1,
                              gamma=0.5, cv_criterium=1e-5)

l1.plot_score(s01_uni, s01_recovered_uni)

# cos1 = sf.signal_gen(a=[2], f=[50], observation_time=1, noise_level=0, plot=True)

# cos1.sampler_gauss(rate=0.03)

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