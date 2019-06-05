import pyCompressSensing as cs

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft, fftshift, ifftshift


class L1min:
    """
    Algorithm for signal recovering with L1-minimimization solver as follows :

    x^hat = argmin_x |y-Ax|^2 + |x|_1
    """

    def solver(self, signal_sampled, phi, lambda_n=1, gamma=0.5, w=100, max_iter=100,
               cv_criterium=1e-3, observation_time=1, verbose=0, tol_null=1e-16, plot=False):
        """ Algorithm for OLS minimization with L1-constraint

            x^hat = argmin_x |y-Ax|^2 + |x|_1

            Parameters
            ----------
            signal_sampled : ndarray
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

            tol_null : float (default 1e-16)
                Tolerance to consider value as 0.
                Tolerance = 0 has no effect

            observation_time : float
                Obervation time of signal, used for x axis.
                If unknown, time and frequencies axis will not be revelant

            verbose : boolean (default=0)
                If verbose=1, displays parameters of sampling and truncated normal law.
                Plot the curve of truncated normal law.

            plot :
                Plot recovered signal, histogram of recovered signal, values of objective function and norm l1

            Return
            ------
            signal_t_recovered : array of shape = []
                Signal recovered expressed in temporal basis

        """

        # TODO : add Lipshitz to regularization w such as w = lambda / L

        # Initialisation du x_hat = signal à reconstruire en frequentiel
        signal_length = phi.get_shape()[1]
        y = signal_sampled
        #x_hat = np.full(signal_length, 1e5)
        x_hat = np.random.uniform(low=0.0, high=np.max(signal_sampled), size=signal_length)

        # Initialisation values
        iteration = 0
        data_fidelity = list()
        objective_fct = list()
        l1_norm = list()

        while iteration < max_iter :

            f1 = phi @ ifft(x_hat)
            grad = fft(phi.T @ f1) - np.real(fft(phi.T @ y))
            z = x_hat - gamma * grad
            x_hat = x_hat + lambda_n * \
                    (self.soft_threshold(np.real(z), w) + 1j * self.soft_threshold(np.imag(z), w) - x_hat)

            l1_norm.append(np.linalg.norm(x_hat, 1))
            data_fidelity.append(np.linalg.norm(y - f1, 2))
            objective_fct.append(data_fidelity[iteration] + lambda_n * l1_norm[iteration])

            if verbose:
                print('Iteration : ', iteration)
                print(f'Data fidelity : {data_fidelity[iteration]:.0f}')
                print(f'L1-norm of X_hat : {l1_norm[iteration]:.0f}')
                print(f'Objective function : {objective_fct[iteration]:.0f}\n')

            if iteration > 0 and cv_criterium != 0:
                if (np.abs(objective_fct[iteration]-objective_fct[iteration-1]))/objective_fct[iteration] \
                        < cv_criterium:
                    print(f'Convergence has been reached at iteration = {iteration}')
                    break

            iteration += 1

        # Set small values to 0
        x_hat.real[abs(x_hat.real) < tol_null] = 0.0
        x_hat.imag[abs(x_hat.imag) < tol_null] = 0.0

        sf = cs.SignalFrame()

        signal_freq_recovered = np.abs(x_hat)/len(x_hat)

        if plot:

            fig = plt.figure(figsize=(8, 8), constrained_layout=True)
            gs = fig.add_gridspec(3, 2)

            ax_df = fig.add_subplot(gs[0, 0], title='Data fidelity', xlabel='Iteration')
            self.my_plotter(ax_df, data_fidelity)

            ax_l1 = fig.add_subplot(gs[0, 1], title='L1-norm of recovered signal', xlabel='Iteration')
            self.my_plotter(ax_l1, l1_norm)

            ax_of = fig.add_subplot(gs[1, 0], title='Objective function', xlabel='Iteration')
            self.my_plotter(ax_of, objective_fct)

            ax_hist = fig.add_subplot(gs[1, 1], title='Histogram of recovered signal')
            ax_hist.hist(signal_freq_recovered.real, bins=50)

            ax_res = fig.add_subplot(gs[2, :], title='Recovered signal in frequency basis', xlabel='Frequency')
            f_grid = np.arange(0, signal_length/2)/observation_time
            signal = signal_freq_recovered[0:int(signal_length/2)].real
            self.my_plotter(ax_res, y=signal, x=f_grid)

            plt.show()

        return signal_freq_recovered

    @staticmethod
    def soft_threshold(x, w):
        """ Threshold function
        """
        return np.sign(x) * np.maximum(np.abs(x) - w, 0.)

    @staticmethod
    def plot_score(signal_temporal, signal_freq_recovered, observation_time=1):
        """ Plot curves of signal recovered on signal to recover.

                    Parameters
                    ----------
                    signal_temporal : bytearray
                        Initial signal in freqency basis

                    signal_freq_recovered : bytearray
                        Recovered signal in frequency basis

                    observation_time : float, default = 1
                        Obervation time of signal, used for x axis.
                        If unknown use default value, time and frequencies axis will not be revelant.

                    Returns
                    -------
                    Plot curves of signal recovered on signal to recover with matplotlib

        """

        signal_length = len(signal_freq_recovered)

        observation_time = len(signal_freq_recovered) / len(signal_temporal)*observation_time
        signal_temporal = signal_temporal[:len(signal_freq_recovered)]

        t_grid = np.linspace(0, observation_time, signal_length)

        plt.figure(figsize=(10, 6))

        plt.subplot(211)
        plt.plot(t_grid, signal_temporal, label='Signal to recover', linewidth=3)
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.xlim((t_grid[0], t_grid[200]))
        plt.legend()

        plt.plot(t_grid, ifft(ifftshift(signal_freq_recovered)), label='Signal recovered', linestyle='--', linewidth=3)
        plt.legend()

        f_grid = np.arange(0, int(signal_length/2+1))/observation_time
        signal_freq_trunc = fftshift(fft(signal_temporal))
        signal_freq_std_ref = signal_freq_trunc/signal_length

        signal_freq_trunc = signal_freq_recovered
        signal_freq_std = signal_freq_trunc/signal_length

        plt.subplot(212)
        plt.plot(f_grid, abs(signal_freq_std_ref[int(signal_length/2):signal_length]),
                 label='Signal to recover', linewidth=3)

        plt.plot(f_grid, abs(signal_freq_std[int(signal_length/2):signal_length]),
                 label=f'Signal recovered, with {np.linalg.norm(signal_freq_recovered, 0)} values',
                 linestyle=':', linewidth=3)
        plt.xlabel('Frequency [Hz]', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.legend()
        plt.show()

    @staticmethod
    def my_plotter(ax,  y, x=False, **param_dict):
        """
        A helper function to make a graph

        Parameters
        ----------
        ax : Axes
            The axes to draw to

        x : array
           The x data

        y : array
           The y data

        param_dict : dict
           Dictionary of kwargs to pass to ax.plot

        Returns
        -------
        out : list
            list of artists added
        """
        if x is False:
            axes = ax.plot(y, **param_dict)
        else:
            axes = ax.plot(x, y, **param_dict)

        return axes
