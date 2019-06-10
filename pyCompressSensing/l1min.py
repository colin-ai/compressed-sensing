from pyCompressSensing.SignalFrame import SignalFrame

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft, irfft, rfft


class L1min:
    """
    Algorithm for signal recovering with L1-minimimization solver as follows :

    x^hat = argmin_x |y-Ax|^2 + |x|_1
    """

    def solver(self, signal_sampled, phi, lambda_n=1, gamma=0.5, w=100, max_iter=100,
               cv_criterium=1e-3, obs_time=1, verbose=0, tol_null=1e-16, plot=False):
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

            obs_time : float
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

        while iteration < max_iter:

            f1 = phi @ irfft(x_hat)
            grad = rfft(phi.T @ f1) - np.real(rfft(phi.T @ y))
            z = x_hat - gamma * grad
            x_hat = x_hat + lambda_n * \
                    (self.soft_threshold(np.real(z), w)  - x_hat)

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
#        x_hat.imag[abs(x_hat.imag) < tol_null] = 0.0

        s_recovered = SignalFrame()
        s_recovered.freq = x_hat/len(x_hat)
        s_recovered.len = int(len(s_recovered.freq))
        s_recovered.temporal = irfft(x_hat)

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
            ax_hist.hist(x_hat.real, bins=50)

            ax_res = fig.add_subplot(gs[2, :], title='Recovered signal in frequency basis', xlabel='Frequency')
            f_grid = np.arange(0, s_recovered.len/2)/obs_time
            signal = x_hat[0:int(s_recovered.len/2)].real
            self.my_plotter(ax_res, y=np.abs(signal), x=f_grid)

            plt.show()

        return s_recovered

    @staticmethod
    def soft_threshold(x, w):
        """ Threshold function
        """
        return np.sign(x) * np.maximum(np.abs(x) - w, 0.)

    @staticmethod
    def plot_score(real_signal, recovered_signal, obs_time=1, f_sampling=1):
        """ Plot curves of signal recovered on signal to recover.

                    Parameters
                    ----------
                    real_signal : SignalFrame
                        Input signal

                    recovered_signal : SignalFrame
                        Recovered signal

                    obs_time : float, default = 1
                        Obervation time of signal, used for x axis.
                        If unknown use default value but time and frequencies axis will not be revelant.

                    f_sampling : float
                        Sampling frequency

                    Returns
                    -------
                    Plot curves of signal recovered on signal to recover with matplotlib

        """

        # Plot 1 : temporal basis
        real_signal_t_trunc = real_signal.temporal[:recovered_signal.len]  # To avoid mismatch dimension with truncated input signal
        t_grid = np.linspace(0, obs_time, recovered_signal.len)

        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        gs = fig.add_gridspec(2, 1)

        ax_t = fig.add_subplot(gs[0, :], xlabel='Time t', ylabel='Amplitude y(t)')
        ax_t.plot(t_grid, real_signal_t_trunc,
                  label='Real signal', alpha=0.7)
        ax_t.plot(t_grid, recovered_signal.temporal,
                  label='Signal recovered', alpha=0.7)
        ax_t.legend()

        # Plot 2 : frequency basis
        real_signal_f_trunc = real_signal.freq[:recovered_signal.len]

        ax_f = fig.add_subplot(gs[1, :], xlabel='Frequency w', ylabel='Amplitude Y(w)')
        f_grid = np.linspace(0, f_sampling/2, recovered_signal.len)

        ax_f.plot(f_grid, abs(real_signal_f_trunc.real),
                  label='Signal to recover', linewidth=1, alpha=0.8)

        ax_f.plot(f_grid, abs(recovered_signal.freq),
                  label=f'Signal recovered, with ' f'{np.linalg.norm(recovered_signal.freq, 0)} values',
                  alpha=0.8)
        ax_f.legend()

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

    @staticmethod
    def indicator_gear(signal, fe, f_gears):
        indic = np.sum(signal[f_gears] ** 2) / signal[fe] ** 2
        return indic