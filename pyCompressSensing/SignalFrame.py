import wave
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft, fftshift, ifftshift, rfft, irfft
from scipy.stats import truncnorm, uniform
from scipy.sparse import csr_matrix, coo_matrix
from scipy.signal import detrend


class SignalFrame:
    """ SignalFrame class provides tools to read wave signal, generate periodic signal with or without noise
    and perform a random sampling

    Attributes
    ----------

    time : bytearray, shape = [len]
        Signal in time basis.

    time_sampled : bytearray, shape = [len]
        Sampled signal in time basis.

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
        self.time = np.array(0)
        self.time_sampled = np.array(0)
        self.freq = np.array(0)
        self.freq_sampled = np.array(0)
        self.obs_time = 'NA'
        self.fundamental = 'NA'
        self.amplitude = 'NA'

        self.len = 0
        self.phi = np.array(0)

    def read_wave(self, filename, coeff_amplitude=1, trunc=1):
        """ Convert wav file in numpy array.

            Parameters
            ----------
            filename : basestring
                Path to input wav

            coeff_amplitude : float
                Coefficient to apply to amplitude if needed to rescale

            trunc : float (default=1)
                Lenght of input signal considered.
                    1 = all instants considered
                    0 = no time considered

            Returns
            -------
            time : SignalFrame
                Return signal expressed in time basis, in numpy array format

        """

        wf = wave.open(filename)
        self.len = wf.getnframes()  # Returns number of audio frames
        frames = wf.readframes(self.len)  # Read len frame, as a string of bytes
        self.time = coeff_amplitude * np.frombuffer(frames, dtype=np.int16)  # 1-D numpy array
        self.time = self.time[:int(trunc * len(self.time))]
        self.detrend().rfft()
        wf.close()

        return self

    def detrend(self):
        """
        Detrend input signal, i.e. substracts the mean, in order to avoid DC bias.

        Returns
        -------
            self
        """
        self.time = detrend(self.time)

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

            t_grid : ndarray
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
        Generate nosie from truncated normal distribution

        Parameters
        ----------
        noise_level : float
            Noise level applied to amplitude of truncated normal distribution as follow:
                noise_level * max(a)
            In case of signal composed of several frequencies,
            noise level is computed from the maximum amplitude.

        std : float
            Standard deviation of truncated normal distribution

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
            Because of Fast Fourier Transform applied to time signal,
             this function is most efficient when n is a power of two, and least efficient when n is prime.

            Noise generated from truncated normal distribution may be add.

            Parameters
            ----------
            a : bytearray, default = [2,15]
                Signal amplitude

            f : bytearray, default = [50,100]
                Signal frequency [Hz]

            observation_time : float, default = 1
                Observation time of signal

            noise_level : float
                Noise level applied to amplitude of truncated normal distribution as follow:
                    noise_level * max(a)
                In case of signal composed of several frequencies, noise level is computed from the maximum amplitude.

            std : float
                Standard deviation of truncated normal distribution

            plot : bool
                If True, plot periodic signal(s) in time basis, gaussian noise, superposition of both
                and signal in frequency basis

            Returns
            -------
        """

        # TODO : warning take real of part fft

        self.amplitude = a
        self.fundamental = f
        self.obs_time = observation_time

        n_freq = len(f)         # Number of frequency of the signal

        sampling_def = 20       # sampling definition : Number of points describing the highest frequency
        f_sampling = np.max(self.fundamental) * sampling_def
        N = int(f_sampling * observation_time + 1)

        self.len = N
        signal_noiseless = np.zeros(N).astype(float)

        t_grid = np.linspace(0, self.obs_time, self.len)
        f_grid = np.linspace(0., f_sampling / 2, self.len)

        for i in range(n_freq):
            signal_noiseless += self.cos_gen(a[i], f[i], t_grid)
        self.time = signal_noiseless

        if noise_level != 0:
            noise = self.noise(noise_level, a, std)
            self.time += signal_noiseless

        self.rfft()

        if plot and noise_level != 0:

            plt.figure(figsize=(10, 10))

            plt.subplot(411)
            plt.plot(t_grid, self.time)
            plt.xlim((0, 5/np.min(f)))
            plt.title(f'Signal y(t) in time basis. Amplitude = {a}, Frequency = {f}, \n  '
                      f'Sampling frequency = {f_sampling}, Observation time = {self.obs_time} s', fontsize=12)
            plt.xlabel('Time t', fontsize=12)
            plt.ylabel('Amplitude y(t)', fontsize=12)

            plt.subplot(412)
            plt.title(f'Signal y(t) in time basis.\n'
                      f'Mean of gaussian noise = 0 and standard deviation = {std}', fontsize=12)
            plt.plot(t_grid, noise)
            plt.xlim((0, 5/np.min(f)))
            plt.xlabel('Time t', fontsize=12)
            plt.ylabel('Amplitude y(t)', fontsize=12)

            plt.subplot(413)
            plt.title('Noisy signal y_s(t) = y(t) + s(t)', fontsize=12)
            plt.plot(t_grid, self.time)
            plt.xlim((0, 5/np.min(f)))
            plt.xlabel('Time t', fontsize=12)
            plt.ylabel('Amplitude y_s(t)', fontsize=12)

            plt.subplot(414)
            plt.plot(f_grid, abs(self.freq))
            plt.xlim(0, np.max(f) * 1.1)
            plt.title('Signal Y(w) in frequency basis', fontsize=12)
            plt.xlabel('Frequency w', fontsize=12)
            plt.ylabel('Amplitude Y(w)', fontsize=12)

            plt.subplots_adjust(hspace=0.5)

        else:
            plt.figure(figsize=(10, 6))

            plt.subplot(211)
            plt.plot(t_grid, self.time)
            plt.xlim((0, 5/np.min(f)))
            plt.title(f'Signal y(t) in time basis. Amplitude = {a}, Frequency = {f}, \n  '
                      f'Sampling frequency = {f_sampling}, Observation time = {self.obs_time} s', fontsize=12)
            plt.xlabel('Time t', fontsize=12)
            plt.ylabel('Amplitude y(t)', fontsize=12)

            plt.subplot(212)
            plt.plot(f_grid, abs(self.freq))
            plt.xlim(0, np.max(f) * 1.1)
            plt.title('Signal Y(w) in frequency basis', fontsize=12)
            plt.xlabel('Frequency w', fontsize=12)
            plt.ylabel('Amplitude Y(w)', fontsize=12)

            plt.subplots_adjust(hspace=0.3)

            plt.show()

        return self

    def fft(self):
        """ Compute the normalized Fast Fourier Transformation of time signal.
        """

        self.freq = fft(self.time)/self.len

    def rfft(self):
        """ Compute the normalized Real Fast Fourier tranform of time signal.
        """

        self.freq = rfft(self.time)/self.len

        return self

    def ifft(self):
        """ Compute inverse fast Fourier transformation of frequency signal.
        """

        self.time = ifft(ifftshift(self.freq))

    def irfft(self):
        """ Compute inverse fast Fourier transformation of frequency signal.
        """

        self.time = irfft(self.freq*self.len)

    def describe(self):
        """ Properties of time/frequency signal
        """

        print(f'Fundamental frequencies : {self.fundamental} Hz\n'
              f'Signal amplitudes : {self.amplitude}\n'
              f'Observation time : {self.obs_time} s\n'
              f'Signal definition : {self.len} instants')
        try:
            print(f'Nyquist–Shannon criterion (rate) : {2*(self.obs_time*np.max(self.fundamental))/(self.len-1)}\n')
        except TypeError:
            print('Nyquist–Shannon criterion (rate) : \'NA\' \n')

    def plot(self, basis, obs_time=1, f_sampling=1):
        """ Plot signal in time or frequency basis.

            Parameters
            ----------
            basis : string
                Define support of signal:
                    time = time
                    freq = frequency

            obs_time : float (default = 1)
                Observation time to consider for x axis.

            f_sampling : float (default = 1)
                Sampling frequecy

            Returns
            -------
            Plot : matplotlib.pyplot.plt
                Plot of input signal with matplotlib.

        """
        plt.figure(figsize=(10, 4))

        if basis == 'time':

            t_grid = np.linspace(0, obs_time, self.len)
            plt.plot(t_grid, self.time)
            plt.title('Signal in time basis', fontsize=12)
            plt.xlabel('Time [s]', fontsize=12)

        if basis == 'freq':
            f_grid = np.linspace(0, f_sampling/2, self.len)
            signal_f_std = self.freq
            plt.plot(f_grid, abs(signal_f_std))
            plt.title('Signal in frequency basis', fontsize=12)
            plt.xlabel('Frequency [Hz]', fontsize=12)

        plt.ylabel('Amplitude', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    def sampler_regular(self, rate=0.5, trunc=1, plot=True):
        """ Method to compute the regular sampled times and measurement matrix associated.

            Parameters
            ----------
                trunc : float (default=1)
                    Signal lenght which have to be considered.
                        1 = all instants considered
                        0 = no time considered

                rate : float (default=0.5)
                    Rate sampling computed as : considered_instants / initial_instants
                        1 = without sampling
                        0 = no time kept

                plot : boolean (default=True)
                   Plot sampling instants on input signal

            Return
            ------
                sampling_instants : array of shape = [rate*len]
                    Sampling instants of input signal.

        """

        signal_trunc = self.time[:int(trunc * len(self.time))]
        N_trunc = len(signal_trunc)  # number of sampling points

        dt = int(1/rate)  # time step size
        t_grid = np.arange(0, N_trunc, dt)  # grid space of discretization
        N_sampled = len(t_grid)

        # Vecteur des instants de l'échantillonnage aléatoire
        sampling_instants = t_grid.astype('int')

        # time signal sampled
        self.time_sampled = self.time[sampling_instants]
        self.freq_sampled = rfft(self.time_sampled)/N_sampled

        # Sampling matrix
        phi = coo_matrix(([True]*N_sampled, (range(N_sampled), sampling_instants)),
                         shape=(N_sampled, N_trunc), dtype=bool)

        self.phi = csr_matrix(phi)

        print('\nSampling process: \n'
              '=================\n'              
              f'Lenght of initial signal : {N_trunc} \n'
              f'Lenght of sampled signal: {N_sampled}\n'
              f'Sampling rate : {rate:.3f}')

        if plot:
            plt.figure(figsize=(10, 4))
            plt.title('Sampling')
            plt.plot(np.arange(0, N_trunc), self.time)
            plt.plot(sampling_instants, self.time_sampled, 'ro',  mfc='none', markersize=5)
            plt.xlabel('Times [s]')
            plt.ylabel('Amplitude')
            plt.xlim((0, 200))
            plt.grid(True)
            plt.show()

    def sampler_uniform(self, rate=0.5, trunc=1, verbose=True, plot=True):
        """ Method to compute the non-regular sampled times according uniform distribution and measurement matrix.
            Value of uniform distribution is limited to [0, dt] with dt = time step.

            Parameters
            ----------
                trunc : float (default=1)
                    Signal lenght which have to be considered.
                        1 = all instants considered
                        0 = no time considered

                rate : float (default=0.5)
                    Rate sampling computed as : considered_instants / initial_instants.
                        1 = without sampling
                        0 = no time kept

                verbose : boolean (default=False)
                    Display information about the sampling performed.

                plot : boolean (default=True)
                    Plot sampling instants on input signal.

            Return
            ------
                sampling_instants : array of shape = [rate*len]
                    Sampling instants of input signal.

        """

        signal_trunc = self.time[:int(trunc * len(self.time))]
        N_trunc = len(signal_trunc)  # number of samplint points

        dt = int(1/rate)  # time step size
        t_grid = np.arange(0, N_trunc, dt)  # grid space of discretization
        N_sampled = len(t_grid)

        # Computation of random variable between [0, DT]
        random_instants = np.rint(uniform.rvs(0, dt, size=N_sampled-1))
        random_instants = np.hstack((random_instants, [0]))

        # Vecteur des instants de l'échantillonnage aléatoire
        sampling_instants = (t_grid + random_instants).astype('int')

        # time signal sampled
        self.time_sampled = self.time[sampling_instants]
        self.freq_sampled = rfft(self.time_sampled)/N_sampled

        # Sampling matrix
        phi = coo_matrix(([True]*N_sampled, (range(N_sampled), sampling_instants)),
                         shape=(N_sampled, N_trunc), dtype=bool)
        self.phi = csr_matrix(phi, dtype=bool)

        if verbose:

            print('\nSampling process: \n'
                  '=================\n'
                  f'Distribution : Uniform\n'              
                  f'Lenght of initial signal : {N_trunc} \n'
                  f'Lenght of sampled signal: {N_sampled}\n'
                  f'Sampling rate : {rate:.3f}')

        if plot:
            plt.figure(figsize=(10, 4))
            plt.title('Sampling')
            plt.plot(np.arange(0, N_trunc), self.time)
            plt.plot(sampling_instants, self.time_sampled, 'ro',  mfc='none', markersize=5)
            plt.xlabel('Times [s]')
            plt.ylabel('Amplitude')
            plt.xlim((0, 200))
            plt.grid(True)
            plt.show()

    def sampler_gauss(self, rate=0.5, trunc=1, std=1, verbose=False, plot=True):
        """ Method to compute the list of sampling times according truncated normal distribution and measurement matrix.
            Value of truncated normal distribution is restricted to [0, dt] with dt = time step. Mean is fixed at
            middle step time.

            Parameters
            ----------
                trunc : float (default=1)
                    Signal lenght which have to be considered.
                        1 = all instants considered
                        0 = no time considered

                rate : float (default=0.5)
                    Rate sampling computed as : considered_instants / initial_instants
                        1 = without sampling
                        0 = no time kept

                std : float (default=1)
                    Standard deviation of truncated normal distribution.

                verbose : boolean (default=0)
                    If verbose=1, displays parameters of sampling and truncated normal distribution.
                    Plot the curve of truncated normal distribution.

                plot : boolean (default=True)
                   Plot sampling instants on input signal

            Return
            ------
                sampling_instants : array of shape = [rate*len]
                    Sampling instants of input signal.


        """

        signal_trunc = self.time[:int(trunc * len(self.time))]
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

        # time signal sampled
        self.time_sampled = self.time[sampling_instants]
        self.freq_sampled = rfft(self.time_sampled)/N_sampled

        # Sampling matrix
        phi = coo_matrix(([True]*N_sampled, (range(N_sampled), sampling_instants)),
                         shape=(N_sampled, N_trunc), dtype=bool)
        self.phi = csr_matrix(phi)

        print('\nSampling process: \n'
              '=================\n'
              f'Distribution : Truncated gaussian\n'
              f'Mean : centred ; Variance : {std}\n'
              f'Lenght of initial signal : {N_trunc} \n'
              f'Lenght of sampled signal: {N_sampled}\n'
              f'Sampling rate : {rate:.3f}')

        if verbose:
            # Curve of sampling distribution
            x_range = np.linspace(0, end_trunc, 10000)
            plt.plot(x_range, truncnorm.pdf(x_range, a, b, loc=mean, scale=std))

            # Checking if sampling is relevant
            print(f'Is all values generated between [{start_trunc},{end_trunc}] ?\n'
                  f'>= {start_trunc} : {np.all(random_instants >= 0)}\n'
                  f'<= {end_trunc} : {np.all(random_instants <= end_trunc)}\n')

        if plot:
            plt.figure(figsize=(10, 4))
            plt.title('Sampling')
            plt.plot(np.arange(0, N_trunc), self.time)
            plt.plot(sampling_instants, self.time_sampled, 'ro',  mfc='none', markersize=5)
            plt.xlabel('Times [s]')
            plt.ylabel('Amplitude')
            plt.xlim((0, 200))
            plt.grid(True)
            plt.show()

    def max_amplitude(self, threshold):
        """

        Parameters
        ----------
        signal : ndarray
            Signal input

        threshold : float
            Only displays frequencies over this threshold.

        Returns
        -------
        result : ndarray
            Column on the left is frequency and column on the right is amplitude.

        """
        freq_filtered = np.argwhere(np.abs(self.freq) > threshold)
        result = np.concatenate((freq_filtered, abs(self.freq[freq_filtered])), axis=1)
        return result
