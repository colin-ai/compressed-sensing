import wave
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft
from scipy.stats import truncnorm
from scipy.sparse import csr_matrix


class SignalFrame:
    """ SignalFrame class provides tools to read wave signal, generate periodic signal with or without noise
    and perform a random sampling

    Attributes
    ----------

    temporal : bytearray, shape = [signal_length]
        Signal in temporal basis.

    temporal_sampled : bytearray, shape = [signal_length]
        Sampled signal in temporal basis.

    freq : bytearray
        Signal in frequency basis.

    freq_sampled : bytearray
        Sampled signal in frequency basis.

    signal_length : int
        Number of frames on source signal.

    phi : scipy.sparse.csr_matrix object with rate*signal_length elements
        Sampling matrix in compressed sparse row matrix format.

    """

    def __init__(self):
        self.temporal = np.array(0)
        self.temporal_sampled = np.array(0)
        self.freq = np.array(0)
        self.freq_sampled = np.array(0)
        self.signal_length = 0
        self.phi = np.array(0)

    def read_wave(self, filename):
        """ Convert wav file in numpy array.

            Parameters
            ----------
            filename : basestring
                Path to input wav

            Returns
            -------
            temporal : array, shape = [signal_length]
                Return signal expressed in temporal basis, in numpy array format

        """

        wf = wave.open(filename)
        self.signal_length = wf.getnframes()  # Returns number of audio frames
        frames = wf.readframes(self.signal_length)  # Reads and returns at most signal_length of audio, as a string of bytes
        self.temporal = np.frombuffer(frames, dtype=np.int16)  # 1-D numpy array
        wf.close()

        return self

    @staticmethod
    def cos_gen(a0, f0, fe, signal_length):
        """ Generate a cosinus signal of one harmonic  :

            Parameters
            ----------
            a0 : float
                Signal amplitude

            f0 : float
                Signal frequency [Hz]

            fe : int
                Sampling frequency [Hz]

            signal_length : int
                Number of points of signal

            Returns
            -------
            signal_cos : bytearray, shape = [signal_length]
                signal periodic built from cos function

        """
        t = np.arange(signal_length) / fe  # temporal discretization
        cos = a0 * np.cos(2 * np.pi * f0 * t)

        return cos

    # @staticmethod
    # def verbose_plot(*args):
    #     """
    #
    #     Parameters
    #     ----------
    #     y_data :
    #     figsize : tuple of int, shape = (width, height)
    #         size of matplotlib figure object
    #     titles : list of str
    #     x_axis :
    #     xlabel :
    #     ylabel :
    #     xlim :
    #     ylim :
    #
    #     Returns
    #     -------
    #     plot :
    #
    #     """
    #
    #
    #     n_plot = len(y_data)
    #     plt.figure(figsize)
    #
    #     if not x_axis:
    #         x_axis = range(y_data)
    #
    #     for i, y in enumerate(y_data):
    #         plt.subplot(n_plot+1+i)
    #         plt.plot(x_axis[i], y)
    #         plt.title(titles[i])
    #         plt.ylim(ylim)
    #         plt.xlim(xlim)
    #         plt.xlabel(xlabel[i])
    #         plt.ylabel(ylabel[i])

    def signal_gen(self, a0=[1, 10], f0=[10, 100], fe=1000, t=1, noise_level=0, mean=0, std=1,
                   plot=False):
        """ Create a periodic signal with 1 or more cosinus.
            Noise generated from truncated normal law may be add.

            Parameters
            ----------
            a0 : bytearray, default = [2,15]
                Signal amplitude

            f0 : bytearray, default = [50,100]
                Signal frequency [Hz]

            fe : float, default = 1000
                Sampling frequency [Hz]

            t : float, default = 1
                length of signal [s]

            noise_level : float
                Noise level applied to amplitude of truncated normal law as follow:
                    noise_level * max(A0)
                In case of signal composed of several frequencies, noise level is computed from the maximum amplitude.

            mean : float
                Mean of truncated normal law

            std : float
                Standard deviation of truncated normal law

            plot : bool
                If True, plot periodic signal(s) in temporal basis, gaussian noise, superposition of both
                and signal in frequency basis

            Returns
            -------
        """

        self.signal_length = int(t*fe)
        signal_noiseless = np.zeros(self.signal_length).astype(float)
        n_cos = len(a0)

        for i in range(n_cos):
            signal_noiseless += self.cos_gen(a0[i], f0[i], fe, self.signal_length)

        if noise_level != 0:
            start_trunc = -np.max(a0) * noise_level
            end_trunc = np.max(a0) * noise_level
            mean = 0
            std = 4
            a, b = (start_trunc - mean) / std, (end_trunc - mean) / std
            gaussian_noise = truncnorm.rvs(a, b, loc=mean, scale=std, size=self.signal_length)

            self.temporal = signal_noiseless + gaussian_noise

        else:
            self.temporal = signal_noiseless

        if plot and noise_level != 0:

            plt.figure(figsize=(14, 9))

            plt.subplot(411)
            plt.plot(signal_noiseless)
            plt.xlim((0, 250))
            plt.title(f'Signal avec amplitude = {a0}, fréquence = {f0}, \n  fe = {fe}, '
                      f'points d\'échantillonage = {self.signal_length}')
            plt.xlabel('Temps [s]')
            plt.ylabel('Amplitude')

            plt.subplot(412)
            plt.title(f'Bruit gaussien de moyenne = {mean} et écart-type = {std}')
            plt.plot(gaussian_noise)
            plt.xlim((0, 250))
            plt.xlabel('Temps [s]')
            plt.ylabel('Amplitude')

            plt.subplot(413)
            plt.title('Signal bruité')
            plt.plot(self.temporal)
            plt.xlim((0, 250))
            plt.xlabel('Temps [s]')
            plt.ylabel('Amplitude')

            plt.subplot(414)
            plt.title(f'Base de Fourier')
            plt.plot(fft(self.temporal)/self.signal_length)
            plt.xlabel('Fréquence [Hz]')
            plt.ylabel('Amplitude')

        else:
            plt.figure(figsize=(14, 9))

            plt.subplot(211)
            plt.plot(signal_noiseless)
            plt.xlim((0, 250))
            plt.title(f'Signal avec amplitude = {A0}, fréquence = {f0}, \n  fe = {fe}, points d\'échantillonage = {self.signal_length}')
            plt.xlabel('Temps [s]')
            plt.ylabel('Amplitude')

            plt.subplot(212)
            plt.title(f'Base de Fourier')
            plt.plot(fft(self.temporal)/self.signal_length)
            plt.xlabel('Fréquence [Hz]')
            plt.ylabel('Amplitude')

        return self

    def fft(self):
        """ Compute fast Fourier transformation of temporal signal.

        """

        self.freq = fft(self.temporal)

    def ifft(self):
        """ Compute inverse fast Fourier transformation of frequency signal.

        """

        self.temporal = ifft(self.freq)

    def describe(self):
        """ Properties of temporal/frequency signal

            Returns
            -------

        """

        print(f'Longueur du signal : {self.signal_length} points')

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

        if basis == 'temporal':
            plt.figure(figsize=(10, 6))
            plt.plot(self.temporal)
            plt.title('Signal en base temporelle', fontsize=15)
            plt.xlabel('Fréquence [Hz]', fontsize=15)
            plt.ylabel('Amplitude', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()

        if basis == 'freq':
            signal_f_std = self.freq / self.signal_length
            X_frequencies = np.arange(-self.signal_length / 2, self.signal_length / 2)

            plt.figure(figsize=(10, 6))
            plt.plot(X_frequencies, abs(signal_f_std))
            plt.title('Signal en base fréquentielle', fontsize=15)
            plt.xlabel('Fréquence [Hz]', fontsize=15)
            plt.ylabel('Amplitude', fontsize=15)
            plt.show()

    def sampler_gauss(self, rate=0.5, signal_length=1, std=1, verbose=0):
        """ Method to compute the list of sampling instants according truncated normal law and sampling matrix.
            Value of truncated normal law is restricted to [0, dt] with dt = step-temporal. Mean is fixed at
            middle step temporal.

            Parameters
            ----------
                signal_length : float (default=1)
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

            Return
            ------
                sampling_instants : array of shape = [rate*signal_length]
                    Sampling instants of input signal.


        """

        signal_t_trunc = self.temporal[:int(signal_length * len(self.temporal))]
        n_signal_t_trunc = len(signal_t_trunc)  # nombre d'échantillonnage total du signal d'entrée
        dt = int(1/rate)  # pas de temps
        x_sampled = np.arange(0, n_signal_t_trunc, dt)  # ensemble de discrétisation
        n_signal_t_sampled = len(x_sampled)

        # Construction de l'échantillonnage aléatoire selon loi normal tronquée (bornée entre [0, dt] )
        start_trunc = 0
        end_trunc = dt
        mean = end_trunc / 2
        a, b = (start_trunc - mean) / std, (end_trunc - mean) / std

        # Génération des valeurs aléatoires entre [0, DT]
        random_instants = np.ceil(truncnorm.rvs(a, b, loc=mean, scale=std, size=n_signal_t_sampled-1))
        random_instants = np.hstack((random_instants, [0]))

        # Vecteur des instants de l'échantillonnage aléatoire
        sampling_instants = (x_sampled + random_instants).astype('int')

        # temporal signal sampled
        self.temporal_sampled = self.temporal[sampling_instants]

        # Sampling matrix
        phi = np.zeros((n_signal_t_sampled, n_signal_t_trunc), dtype=int)

        for i in range(n_signal_t_sampled):
            phi[i, sampling_instants[i]] = 1

        self.phi = csr_matrix(phi)

        if verbose:
            # Représentation graphique
            x_range = np.linspace(0, end_trunc, 10000)
            plt.plot(x_range, truncnorm.pdf(x_range, a, b, loc=mean, scale=std))

            # Vérification cohérence de l'échantillonnage
            print(f'Est-ce que toutes les valeurs générées sont comprises dans [{start_trunc},{end_trunc}] ?\n'
                  f'>= {start_trunc} : {np.all(random_instants >= 0)}\n'
                  f'<= {end_trunc} : {np.all(random_instants <= end_trunc)}\n')

            print('\nCas d\'étude : \n'
                  '=============\n'
                  f'Echantillonnage gaussien, moyenne centrée sur l\'intervalle, variance {std}\n'
                  f'Definition du signal initial : {n_signal_t_trunc} \n'
                  f'Définition du signal échantilloné : {n_signal_t_sampled}\n'
                  f'Taux d\'échantillonnage : {rate:.3f}')
