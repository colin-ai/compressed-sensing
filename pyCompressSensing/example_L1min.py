from pyCompressSensing.signal_frame import *
from pyCompressSensing.l1min import *


sf = SignalFrame()
cos1 = sf.signal_gen(a=[2], f=[50], observation_time=1, noise_level=0, plot=True)

cos1.sampler_gauss(rate=0.03)

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