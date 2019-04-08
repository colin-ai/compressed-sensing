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

l1.plot_recovery(signal_freq_recovered)
l1.plot_score(signal.temporal, signal_freq_recovered)