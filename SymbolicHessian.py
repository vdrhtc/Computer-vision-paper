from sympy import lambdify, Symbol
from sympy.matrices.dense import hessian
from numpy import linspace, ones, ones_like, where, logical_and, zeros, concatenate

from matplotlib import pyplot as plt

class FisherMatrix():

    def __init__(self, X, mus, theta):
        self._X = X
        self._mus = mus
        self._theta = theta
        self._sigma = Symbol("sigma")

    def symbolic_expectation_of_hessian_of_log_likelihood_sum_ith_element(self):
        expects = []
        for mu in self._mus:
            expects.append(-1 / 2 / self._sigma * (hessian(mu ** 2, self._theta) -
                                    2 * mu * hessian(mu, self._theta)))
        return expects

    def numeric_expectation_of_hessian_of_log_likelihood_sum_ith_element(self):
        numeric_funcs = []
        symbolic_exprs = self.symbolic_expectation_of_hessian_of_log_likelihood_sum_ith_element()
        for symbolic_expr in symbolic_exprs:
            args = self._theta + (self._sigma, self._X)
            numeric_funcs.append(lambdify(args, symbolic_expr))
        return numeric_funcs

    def numeric_mus(self):
        Nmus = []
        for mu in self._mus:
            args = self._theta + (self._X,)
            Nmus.append(lambdify(args, mu))
        return Nmus

    def calculate_numeric(self, X_vals, theta_vals, sigma_vals, mu_chooser = None):
        if len(theta_vals) != len(self._theta):
            raise ValueError("Theta vals number invalid")
        if mu_chooser is None:
            if len(self._mus) > 1:
                print("Calculating the Hessian for a multivalued function!")

            NEHs = self.numeric_expectation_of_hessian_of_log_likelihood_sum_ith_element()
            args = theta_vals + (sigma_vals, X_vals)

            hess_expect_list = NEHs[0](*args)
            for NEH in NEHs[1:]:
                hess_expect_list = concatenate((hess_expect_list, NEH(*args)))

        else:
            if len(self._mus) > 1:
                print("Calculating the Hessian for an N-branch function")

            Nmus = self.numeric_mus()
            X_partitioning = mu_chooser(X_vals, Nmus, theta_vals)

            NEHs = self.numeric_expectation_of_hessian_of_log_likelihood_sum_ith_element()

            hess_expect_list = zeros((len(self._theta), len(self._theta), len(X_vals)))
            for NEHith, part in zip(NEHs, X_partitioning):
                args = theta_vals + (sigma_vals, X_vals)
                branch_hessian_expectation = NEHith(*args)
                hess_expect_list[:, :, part] = branch_hessian_expectation[:, :, part]

        for i in range(len(self._theta)):
            plt.plot(X_vals, hess_expect_list[i, i, :])

        return -hess_expect_list.sum(axis=-1)

    def example_mu_chooser_for_anticrossings(self, X_vals, mus, theta_vals, freq_window):
        f_minus_func, f_plus_func = mus
        Is = X_vals
        f_c, g, Pi, I_ss, f_qmax, d = theta_vals

        f_plus_vals = f_plus_func(f_c, g, Pi, I_ss, f_qmax, d, Is)
        f_minus_vals = f_minus_func(f_c, g, Pi, I_ss, f_qmax, d, Is)

        upper_limit = f_c + freq_window
        lower_limit = f_c - freq_window

        res_freqs_model = ones_like(X_vals) * f_c
        idcs1 = where(logical_and(lower_limit < f_minus_vals,
                                  f_minus_vals < upper_limit))
        idcs2 = where(logical_and(lower_limit < f_plus_vals,
                                  f_plus_vals < upper_limit))

        return idcs1[0], idcs2[0]


