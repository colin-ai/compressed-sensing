import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.fftpack import fft, ifft, fftshift, ifftshift, rfft, irfft
from scipy.stats import truncnorm
from scipy.sparse import csr_matrix


class SignalFrame:
    """ SignalFrame class provides tools to read wave signal, generate periodic signal with or without noise
    and perform a random sampling

    Attributes
    ----------

    temporal : bytearray, shape = [len]
        Signal in temporal basis.

    temporal_sampled : bytearray, shape = [len]
        Sampled signal in temporal basis.

    freq : bytearray
        Signal in frequency basis.

    freq_sampled : bytearray
        Sampled signal in frequency basis.

    len : int
        Signal length.

    phi : scipy.sparse.csr_matrix object with rate*len elements
        Sampling matrix in compressed sparse row matrix format.

    """

    def __init__(self):
        self.temporal = np.array(0)
        self.temporal_sampled = np.array(0)
        self.freq = np.array(0)
        self.freq_sampled = np.array(0)
        self.obs_time = 'NA'
        self.fundamental = 'NA'
        self.amplitude = 'NA'

        self.len = 0
        self.phi = np.array(0)

    def read_wave(self, filename):
        """ Convert wav file in numpy array.

            Parameters
            ----------
            filename : basestring
                Path to input wav

            Returns
            -------
            temporal : array, shape = [len]
                Return signal expressed in temporal basis, in numpy array format

        """

        wf = wave.open(filename)
        self.len = wf.getnframes()  # Returns number of audio frames
        frames = wf.readframes(self.len)  # Read len frame, as a string of bytes
        self.temporal = np.frombuffer(frames, dtype=np.int16)  # 1-D numpy array
        wf.close()

        return self

    @staticmethod
    def cos_gen(a, f, t_grid):
        """ Generate a cosinus signal of one harmonic  :

            Parameters
            ----------
            a : float
                Signal intenisty

            f : float
                Signal frequency [Hz]

            t_grid : bytearray
                Temporal grid for for computation of cosinus [s]

            Returns
            -------
            signal_cos : bytearray, shape = [len]
                signal periodic built from cos function

        """

        cos = a * np.cos(2 * np.pi * f * t_grid)

        return cos

    def noise(self, noise_level, signal_ampltiude, std):
        """
        Generate nosie from truncated normal law

        Parameters
        ----------
        noise_level : float
            Noise level applied to amplitude of truncated normal law as follow:
                noise_level * max(a)
            In case of signal composed of several frequencies,
            noise level is computed from the maximum amplitude.

        std : float
            Standard deviation of truncated normal law

        signal_ampltiude: bytearray, float or int
            Signal amplitude(s)

        Returns
        -------

        """

        start_trunc = -np.max(signal_ampltiude) * noise_level
        end_trunc = np.max(signal_ampltiude) * noise_level
        mean = 0
        std = std
        a, b = (start_trunc - mean) / std, (end_trunc - mean) / std
        gaussian_noise = truncnorm.rvs(a, b, loc=mean, scale=std, size=self.len)

        return gaussian_noise

    def signal_gen(self, a=[1, 10], f=[10, 100], observation_time=1, noise_level=0, std=1, plot=False):
        """ Create a periodic signal with 1 or more cosinus.
            According SciPy :
            Because of Fast Fourier Transform applied to temporal signal,
             this function is most efficient when n is a power of two, and least efficient when n is prime.

            Noise generated from truncated normal law may be add.

            Parameters
            ----------
            a : bytearray, default = [2,15]
                Signal amplitude

            f : bytearray, default = [50,100]
                Signal frequency [Hz]

            observation_time : float, default = 1
                Observation time of signal

            noise_level : float
                Noise level applied to amplitude of truncated normal law as follow:
                    noise_level * max(a)
                In case of signal composed of several frequencies, noise level is computed from the maximum amplitude.

            std : float
                Standard deviation of truncated normal law

            plot : bool
                If True, plot periodic signal(s) in temporal basis, gaussian noise, superposition of both
                and signal in frequency basis

            Returns
            -------
        """

        # TODO : warning take real of part fft

        self.amplitude = a
        self.fundamental = f
        self.obs_time = observation_time

        n_freq = len(f)         # Number of frequency of the signal

        sampling_def = 20       # samplin definition : Number of points describing the highest frequency
        f_sampling = np.max(self.fundamental) * sampling_def
        N = int(f_sampling * observation_time + 1)

        self.len = N
        signal_noiseless = np.zeros(N).astype(float)

        t_grid = np.linspace(0, self.obs_time, self.len)
        f_grid = np.linspace(0., f_sampling / 2, self.len)

        for i in range(n_freq):
            signal_noiseless += self.cos_gen(a[i], f[i], t_grid)

        if noise_level != 0:
            noise = self.noise(noise_level, a, std)
            self.temporal = signal_noiseless + noise

        self.rfft()

        if plot and noise_level != 0:

            plt.figure(figsize=(14, 12))

            plt.subplot(411)
            plt.plot(t_grid, signal_noiseless)
            plt.xlim((0, 5/np.min(f)))
            plt.title(f'Signal amplitude = {a}, Frequency = {f}, \n  Sampling frequency = {f_sampling}, '
                      f'Observation time = {self.obs_time} s', fontsize=12)
            plt.xlabel('Time [s]', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)

            plt.subplot(412)
            plt.title(f'Mean of gaussian noise = 0 and standard deviation = {std}', fontsize=12)
            plt.plot(t_grid, noise)
            plt.xlim((0, 5/np.min(f)))
            plt.xlabel('Time [s]', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)

            plt.subplot(413)
            plt.title('Noisy signal', fontsize=12)
            plt.plot(t_grid, self.temporal)
            plt.xlim((0, 5/np.min(f)))
            plt.xlabel('Time [s]', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)

            plt.subplot(414)
            plt.plot(f_grid, abs(self.freq))
            plt.xlim(0, np.max(f) * 1.1)
            plt.title('Signal in frequency basis', fontsize=12)
            plt.xlabel('Frequency [Hz]', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)

            plt.subplots_adjust(hspace=0.5)

        else:
            plt.figure(figsize=(14, 9))

            plt.subplot(211)
            plt.plot(t_grid, self.temporal)
            plt.xlim((0, 5/np.min(f)))
            plt.title(f'Signal amplitude = {a}, Frequency = {f}, \n  Sampling frequency = {f_sampling}, '
                      f'Observation time = {self.obs_time} s', fontsize=12)
            plt.xlabel('Time [s]', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)

            plt.subplot(212)
            plt.plot(f_grid, abs(self.freq))
            plt.xlim(0, np.max(f) * 1.1)
            plt.title('Signal in frequency basis', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.xlabel('Frequency [Hz]', fontsize=12)

            plt.subplots_adjust(hspace=0.3)

            plt.show()

        return self

    def fft(self):
        """ Compute the normalized Fast Fourier Transformation of temporal signal.
        """

        self.freq = fft(self.temporal)/self.len

    def rfft(self):
        """ Compute the normalized Real Fast Fourier tranform of temporal signal.
        """

        self.freq = rfft(self.temporal)/self.len

    def ifft(self):
        """ Compute inverse fast Fourier transformation of frequency signal.
        """

        self.temporal = ifft(ifftshift(self.freq))

    def irfft(self):
        """ Compute inverse fast Fourier transformation of frequency signal.
        """

        self.temporal = irfft(self.freq*self.len)

    def describe(self):
        """ Properties of temporal/frequency signal
        """

        print(f'Fundamental frequencies : {self.fundamental} Hz\n'
              f'Signal amplitudes : {self.amplitude}\n'
              f'Observation time : {self.obs_time} s\n'
              f'Signal definition : {self.len} instants')
        try:
            print(f'Nyquist–Shannon criterion (rate) : {2*(self.obs_time*np.max(self.fundamental))/(self.len-1)}\n')
        except TypeError:
            print('Nyquist–Shannon criterion (rate) : \'NA\' \n')

    def plot(self, basis):
        """ Plot signal in temporal or frequency basis.

            Parameters
            ----------
            basis : string
                Define support of signal:
                    temporal = temporal
                    freq = frequency

            Returns
            -------
            Plot : matplotlib.pyplot.plt
                Curve of input signal with matplotlib

        """
        plt.figure(figsize=(10, 6))

        if basis == 'temporal':
            plt.plot(self.temporal)
            plt.title('Signal in temporal basis', fontsize=12)
            plt.xlabel('Time [s]', fontsize=12)

        if basis == 'freq':
            signal_f_std = self.freq
            f_grid = np.arange(0, self.len)
            plt.plot(f_grid, abs(signal_f_std))
            plt.title('Signal in frequency basis', fontsize=12)
            plt.xlabel('Frequency [Hz]', fontsize=12)

        plt.ylabel('Amplitude', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    def sampler_gauss(self, rate=0.5, trunc=1, std=1, verbose=False, plot=True):
        """ Method to compute the list of sampling instants according truncated normal law and sampling matrix.
            Value of truncated normal law is restricted to [0, dt] with dt = step-temporal. Mean is fixed at
            middle step temporal.

            Parameters
            ----------
                trunc : float (default=1)
                    Signal lenght which have to be considered.
                        1 = all instants considered
                        0 = no instant considered

                rate : float (default=0.5)
                    Rate sampling computed as : considered_instants / initial_instants
                        1 = without sampling
                        0 = no one instant keep

                std : float (default=1)
                    Standard deviation of truncated normal law.

                verbose : boolean (default=0)
                    If verbose=1, displays parameters of sampling and truncated normal law.
                    Plot the curve of truncated normal law.

                plot : boolean (default=True)
                   Plot sampling instants on input signal

            Return
            ------
                sampling_instants : array of shape = [rate*len]
                    Sampling instants of input signal.


        """

        signal_trunc = self.temporal[:int(trunc * len(self.temporal))]
        N_trunc = len(signal_trunc)  # nombre d'échantillonnage total du signal d'entrée

        dt = int(1/rate)  # pas de temps
        t_grid = np.arange(0, N_trunc, dt)  # ensemble de discrétisation
        N_sampled = len(t_grid)

        # Construction de l'échantillonnage aléatoire selon loi normal tronquée (bornée entre [0, dt] )
        start_trunc = 0
        end_trunc = dt
        mean = end_trunc / 2
        a, b = (start_trunc - mean) / std, (end_trunc - mean) / std

        # Génération des valeurs aléatoires entre [0, DT]
        random_instants = np.ceil(truncnorm.rvs(a, b, loc=mean, scale=std, size=N_sampled-1))
        random_instants = np.hstack((random_instants, [0]))

        # Vecteur des instants de l'échantillonnage aléatoire
        sampling_instants = (t_grid + random_instants).astype('int')

        # temporal signal sampled
        self.temporal_sampled = self.temporal[sampling_instants]
        self.freq_sampled = rfft(self.temporal_sampled)/N_sampled

        # Sampling matrix
        phi = np.zeros((N_sampled, N_trunc), dtype=int)

        for i in range(N_sampled):
            phi[i, sampling_instants[i]] = 1

        self.phi = csr_matrix(phi)

        print('\nSampling process: \n'
              '=================\n'
              f'Law : Truncated gaussian\n'
              f'Mean : centred ; Variance : {std}\n'
              f'Lenght of initial signal : {N_trunc} \n'
              f'Lenght of sampled signal: {N_sampled}\n'
              f'Sampling rate : {rate:.3f}')

        if verbose:
            # Curve of sampling law
            x_range = np.linspace(0, end_trunc, 10000)
            plt.plot(x_range, truncnorm.pdf(x_range, a, b, loc=mean, scale=std))

            # Checking if sampling is relevant
            print(f'Is all values generated between [{start_trunc},{end_trunc}] ?\n'
                  f'>= {start_trunc} : {np.all(random_instants >= 0)}\n'
                  f'<= {end_trunc} : {np.all(random_instants <= end_trunc)}\n')

        if plot:
            plt.figure(figsize=(14, 6))
            plt.title('Sampling')
            plt.plot(np.arange(0, N_trunc), self.temporal)
            plt.plot(sampling_instants, self.temporal_sampled, 'ro',  mfc='none', markersize=10)
            plt.xlabel('Times [s]')
            plt.ylabel('Amplitude')
            plt.xlim((0, 200))
            plt.show()
