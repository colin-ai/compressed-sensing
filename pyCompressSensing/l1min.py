import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft


class L1min:
    """
    Algorithm for signal recovering with L1-minimimization solver as follows :

    x^hat = argmin_x |y-Ax|^2 + |x|_1

    """
    def __init__(self):
        self._objective_fct = list()
        self._l1_norm = list()

    def solver(self, signal_sampled, phi, lambda_n=1, gamma=0.5, w=100, max_iter=100,
               cv_criterium=1e-3, verbose=0):
        """ Algorithm for OLS minimization with L1-constraint

            x^hat = argmin_x |y-Ax|^2 + |x|_1

            Parameters
            ----------
            signal_sampled : bytearray
                Sampled signal in temporal basis

            phi : scipy.sparse.csr_matrix
                Random sampling matrix

            lambda_n : float (default=1)
                relaxation, applied to the proxima term

            gamma : float (default=0.5)
                learning right, applied to descent gradient term

            w : float (default=100)
                regularization value, applied to L1 norm.

            max_iter : int (default=100)
                Number max of iteration

            cv_criterium : float (default 1e-3)
                Convergence criterium on target fonction

            verbose : boolean (default=0)
                If verbose=1, displays parameters of sampling and truncated normal law.
                Plot the curve of truncated normal law.

            Return
            ------
            signal_t_recovered : array of shape = []
                Signal recovered expressed in temporal basis

        """

        # Initialisation du x_hat = signal à reconstruire en frequentiel
        signal_length = phi.get_shape()[1]
        x_hat = np.full(signal_length, 1e3)
        y = signal_sampled

        # Initialisation values
        iteration = 0
        self._objective_fct.append(cv_criterium + 1)
        c = 1 / signal_length

        while iteration < max_iter and self._objective_fct[iteration] > cv_criterium:

            f1 = phi @ ifft(x_hat)
            grad = fft(phi.T @ f1) * c - np.real(fft(phi.T @ y) * c)
            z = x_hat - gamma * grad
            x_hat = x_hat + lambda_n * \
                    (self.soft_threshold(np.real(z), w) + 1j * self.soft_threshold(np.imag(z), w) - x_hat)

            self._l1_norm.append(np.linalg.norm(x_hat, 1))
            self._objective_fct.append(np.linalg.norm(f1 - y, 2) + lambda_n * np.linalg.norm(x_hat, 1))

            if verbose:
                print('Iteration : ', iteration)
                #   print(f'Data fidelity: {data_fidel[iteration]}')
                #   print(f'Norme1 de X_hat: {Xhat_L1[iteration]}\n')
                print(f'{iteration} : {self._objective_fct[iteration]:}')
                #    print('grad :',np.linalg.norm(grad,2))
                #    print('Z :',np.linalg.norm(Z,2))
                print('\n')

            iteration += 1

        signal_freq_recovered = x_hat

        return signal_freq_recovered

    @staticmethod
    def soft_threshold(x, w):
        """ Threshold function
        """
        return np.sign(x) * np.maximum(np.abs(x) - w, 0.)

    def plot_recovery(self, signal_freq_recovered):
        """ Plot curves of signal recovered and intermediate values.

            Parameters
            ----------

                signal_freq_recovered : bytearray
                    Signal recovered in temporal basis

            Returns
            -------
            Plot curves of signal recovered and intermediate values with matplotlib

        """

        plt.figure(figsize=(14, 10))

        plt.subplot(221)
        plt.plot(self._objective_fct, label='Fonction objective')
        plt.xticks(label='Iteration')

        plt.subplot(222)
        plt.hist(np.abs(signal_freq_recovered.real), bins=50, label='Histogramme du signal reconstruit')

        plt.subplot(223)
        plt.plot(self._l1_norm, label='Norme L1 de X_hat')
        plt.xticks(label='Iteration')

        plt.subplot(224)
        plt.plot(signal_freq_recovered, label='Signal reconstruit')

        plt.show()

    @staticmethod
    def plot_score(signal_temporal, signal_freq_recovered):
        """ Plot curves of signal recovered on signal to recover.

                    Parameters
                    ----------
                    signal_temporal : bytearray
                        Initial signal in freqency basis

                    signal_freq_recovered : bytearray
                        Recovered signal in frequency basis

                    Returns
                    -------
                    Plot curves of signal recovered on signal to recover with matplotlib

        """

        signal_length = len(signal_freq_recovered)

        signal_temporal = signal_temporal[:len(signal_freq_recovered)]
        signal_freq_trunc = fft(signal_temporal)

        signal_freq_std = signal_freq_trunc/signal_length
        x_frequencies = np.arange(-signal_length/2, signal_length/2)

        plt.figure(figsize=(14, 7))

        plt.subplot(211)
        plt.plot(x_frequencies, abs(signal_freq_std), label='Signal to recover')
        plt.xlabel('Frequency [Hz]', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.legend()

        plt.plot(x_frequencies, abs(signal_freq_recovered.real),
                 label=f'Signal recovered, with {np.linalg.norm(signal_freq_recovered, 0)} values', linestyle=':')
        plt.legend()

        plt.subplot(212)
        plt.plot(x_frequencies, signal_temporal, label='Signal to recover')
        plt.xlabel('Time [s]', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.legend()

        plt.plot(x_frequencies, ifft(signal_freq_recovered), label='Signal recovered', linestyle=':')
        plt.legend()

        plt.show()
