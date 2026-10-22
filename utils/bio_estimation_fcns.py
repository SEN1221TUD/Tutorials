import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import log, exp, Variable, MonteCarlo, exp, bioMultSum
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta


# This function estimates the MNL model and returns the estimation results
# input values: utilities for all three alternatives, the choices, the database, and the model name

def estimate_mnl(V, AV, CHOICE, database, model_name):

    # Define the choice model: The function models.logit() computes the MNL choice probabilities of the chosen alternative given the V. 
    prob = models.logit(V, AV, CHOICE)

    # Define the log-likelihood by taking the log of the choice probabilities of the chosen alternative
    LL = log(prob)
   
    # Create the Biogeme object containing the object database and the formula for the contribution to the log-likelihood of each row using the following syntax:
    biogeme = bio.BIOGEME(database, LL)
    
    # The following syntax passes the name of the model:
    biogeme.modelName = model_name

    # Some object settings regaridng whether to save the results and outputs 
    biogeme.generate_pickle = False
    biogeme.generate_html = False
    biogeme.save_iterations = False

    # Syntax to calculate the null log-likelihood. The null-log-likelihood is used to compute the rho-square 
    biogeme.calculate_null_loglikelihood(AV)

    # This line starts the estimation and returns the results object.
    results = biogeme.estimate()
    return results

def simulate_mnl(V, AV, CHOICE, database, betas, model_name):

    # # Simulate the model on the test database

    # Compute the denominator: sum of exp(V[i]) for all available alternatives
    denominator = sum(exp(V[i]) * AV[i] for i in V)

    # Compute the choice probabilities for each alternative
    simulate = {}
    for i in V:
        simulate[f'P{i}'] = (exp(V[i]) / denominator) * AV[i]
        simulate[f'V{i}'] = V[i]

    # Compute the probability of the chosen alternative
    simulate['Pchosen'] = sum((CHOICE == i) * (exp(V[i]) / denominator) for i in V)

    # Run the simulation
    biogeme_sim = bio.BIOGEME(database, simulate)
    Psim = biogeme_sim.simulate(betas)

    # Compute the log-likelihood of the simulated data
    LLsim = np.log(Psim['Pchosen']).sum()

    LLnull = biogeme_sim.calculate_null_loglikelihood(AV)

    rho_square = 1 - (LLsim / LLnull)

    # Add the results to a dictionary
    simresults ={"V1": Psim['V1'], "V2": Psim['V2'],"P1": Psim['P1'], "P2": Psim['P2'],"Pchosen": Psim['Pchosen'], "LL": LLsim, "rho_square": rho_square,"model_name": model_name}
    
    return simresults

# This function estimates the cross-sectional MXL model and returns the estimation results
def estimate_mxl(V,AV,CHOICE,biodata,model_name, num_draws = 100):
    
    # The conditional probability of the chosen alternative is a logit
    condProb = models.logit(V, AV ,CHOICE)

    # The unconditional probability is obtained by simulation
    uncondProb = MonteCarlo(condProb)

    # The Log-likelihood is the log of the unconditional probability
    LL = log(uncondProb)

    # Create the Biogeme estimation object containing the data and the model
    biogeme = bio.BIOGEME(biodata , LL, number_of_draws=num_draws)

    # Set reporting levels
    biogeme.generate_pickle = False
    biogeme.generate_html = False
    biogeme.save_iterations = False
    biogeme.modelName = model_name

    # Estimate the parameters and print the results
    results = biogeme.estimate()
    return results

# This function estimates the panel MXL model and returns the estimation results
def estimate_panel_mxl(V,AV,CHOICE,obs_per_ind,biodata_wide,model_name, num_draws = 100):

    # The conditional probability of the chosen alternative is a logit
    condProb = [models.loglogit(V[q], AV, CHOICE[q]) for q in range(obs_per_ind)] 

    # Take the product of the conditional probabilities
    condprobIndiv = exp(bioMultSum(condProb))   # exp to convert from logP to P again

    # The unconditional probability is obtained by simulation
    uncondProb = MonteCarlo(condprobIndiv)

    # The Log-likelihood is the log of the unconditional probability
    LL = log(uncondProb)

    # Create the Biogeme estimation object containing the data and the model
    num_draws = num_draws
    biogeme = bio.BIOGEME(biodata_wide , LL, number_of_draws=num_draws)

    # Compute the null loglikelihood for reporting
    # Note that we need to compute it manually, as biogeme does not do this for panel data
    biogeme.nullLogLike = biodata_wide.get_sample_size() * obs_per_ind * np.log(1 / len(V[0]))

    # Set reporting levels
    biogeme.generate_pickle = False
    biogeme.generate_html = False
    biogeme.save_iterations = False
    biogeme.modelName = model_name    

    # Estimate the parameters and print the results
    results = biogeme.estimate()
    return results

def is_mxl(stats) -> bool:
    return "Types of draws" in stats


def get_stat(stats, key):
    item = stats[key]
    return item.value if hasattr(item, "value") else item[0]


def make_stat_rows(stats, row_specs):
    return [
        (label, get_stat(stats, key), fmt)
        for label, key, fmt in row_specs
        if key in stats]


def format_value(value, fmt, width=12):
    if fmt == "left_s":
        return str(value)

    if fmt == "s":
        return f"{str(value):>{width}}"

    if fmt == "d":
        return f"{int(value):>{width}d}"

    return f"{float(value):>{width}{fmt}}"


def print_results(results):
    
    # Print the estimation statistics
    opt = results.data.optimizationMessages
    stats = results.get_general_statistics()

    summary_rows = [
        ("Cause of termination", opt["Cause of termination"], "left_s"),
        ("Number of iterations", opt["Number of iterations"], "d"),
        ("Estimation time", timedelta(seconds=int(opt["Optimization time"].total_seconds())),"s",)]

    if not is_mxl(stats):
            mnl_row_specs = [
                ("Number of estimated parameters", "Number of estimated parameters", "d"),
                ("Sample size",                    "Sample size", "d"),
                ("Excluded observations",          "Excluded observations", "d"),
                ("Null log likelihood",            "Null log likelihood", ".2f"),
                ("Init log likelihood",            "Init log likelihood", ".2f"),
                ("Final log likelihood",           "Final log likelihood", ".2f"),
                ("Rho-square for the null model",  "Rho-square for the null model", ".3f"),
                ("Bayesian Information Criterion", "Bayesian Information Criterion", ".2f"),
                ("Number of threads",              "Nbr of threads", "d"),
            ]
            stat_rows = make_stat_rows(stats, mnl_row_specs)

    else:
        mxl_row_specs = [
            ("Number of estimated parameters", "Number of estimated parameters", "d"),
            ("Number of individuals",          "Sample size", "d"),
            ("Excluded observations",          "Excluded observations", "d"),
            ("Null log likelihood",            "Null log likelihood", ".2f"),
            ("Init log likelihood",            "Init log likelihood", ".2f"),
            ("Final log likelihood",           "Final log likelihood", ".2f"),
            ("Rho-square for the null model",  "Rho-square for the null model", ".3f"),
            ("Bayesian Information Criterion", "Bayesian Information Criterion", ".2f"),
            ("Number of threads",              "Nbr of threads", "d"),
            ("Number of draws",                "Number of draws", "d"),
            ("Types of draws",                 "Types of draws", "left_s"),
        ]
        stat_rows = make_stat_rows(stats, mxl_row_specs)

    rows = summary_rows + stat_rows
    label_width = max(len(label) for label, _, _ in rows)

    print()
    print("Estimation finished")

    for label, value, fmt in rows:
        print(f"{label:<{label_width}} : {format_value(value, fmt)}")
        
    # Get the model parameters in a pandas table and  print it
    beta_hat = results.get_estimated_parameters()
    
    # Round the results to suitable decimal places
    beta_hat = beta_hat.round(4)
    beta_hat['Rob. t-test']  = beta_hat['Rob. t-test'].round(2)
    beta_hat['Rob. p-value'] = beta_hat['Rob. p-value'].round(2)
    print()
    print(beta_hat)

def get_beta_param(params, name):
    """Get parameter value ignoring case and allowing B_, b_, or beta_ as prefixes."""
    name = name.lower()
    for key in params:
        if key.lower() == name or (key.lower().endswith(name) and key.lower().startswith(('b_','b_', 'beta_', 'mu_'))):
            return float(params[key])
    raise KeyError(f"Parameter for '{name}' not found.")

def get_sigma_param(params, name):
    """Get parameter value ignoring case and allowing sigma_, S_, or sigma_ as prefixes."""
    name = name.lower()
    for key in params:
        if key.lower().endswith(name) and key.lower().startswith(('sigma_', 'S_')):
            return float(params[key])
    raise KeyError(f"Parameter for '{name}' not found.")

def plot_distributions(mu, sigma, dist, xmin, xmax, xlabel="Marginal utility", ax=None):
    """Plot one normal, lognormal, or negative_lognormal distribution.

    With Z ~ N(0, 1), the plotted taste parameter is:
        normal:             beta = mu + abs(sigma) * Z
        lognormal:          beta = exp(mu + abs(sigma) * Z)
        negative_lognormal: beta = -exp(mu + abs(sigma) * Z)

    For lognormal types, mu and sigma describe the underlying normal,
    not the mean and standard deviation of beta. A signed estimated
    scale is accepted; sigma must be nonzero. xmin and xmax only set
    the viewing window: the density is not truncated or renormalised.

    Returns (fig, ax). In a script, call plt.show() to display the plot.
    """
    valid_types = ("normal", "lognormal", "negative_lognormal")
    if not isinstance(dist, str):
        raise ValueError(f"dist must be one of {valid_types}.")
    dist = dist.strip().lower().replace("-", "_").replace(" ", "_")
    if dist not in valid_types:
        raise ValueError(f"Unknown distribution {dist!r}. Choose from {valid_types}.")

    try:
        mu, sigma, xmin, xmax = map(float, (mu, sigma, xmin, xmax))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mu, sigma, xmin and xmax must be finite numbers.") from exc
    if not np.all(np.isfinite([mu, sigma, xmin, xmax])):
        raise ValueError("mu, sigma, xmin and xmax must be finite numbers.")
    sigma = abs(sigma)
    if sigma == 0:
        raise ValueError("sigma must be nonzero; a fixed parameter has no continuous PDF.")
    if xmin >= xmax:
        raise ValueError("xmin must be smaller than xmax.")
    if dist == "lognormal" and xmax <= 0:
        raise ValueError("A lognormal distribution needs a range with xmax > 0.")
    if dist == "negative_lognormal" and xmin >= 0:
        raise ValueError("A negative lognormal distribution needs a range with xmin < 0.")

    x = np.linspace(xmin, xmax, 1000)
    if xmin < 0 < xmax:
        x = np.sort(np.append(x, 0.0))

    if dist == "normal":
        mean = mu
        pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma / np.sqrt(2 * np.pi)
    else:
        sign = -1 if dist == "negative_lognormal" else 1
        mean = sign * np.exp(mu + 0.5 * sigma ** 2)
        pdf = np.zeros_like(x)
        on_support = sign * x > 0
        magnitude = sign * x[on_support]
        # Evaluate logs only on the support; the PDF is zero elsewhere,
        # including at x = 0. Log space avoids multiplying small factors.
        log_pdf = (
            -0.5 * ((np.log(magnitude) - mu) / sigma) ** 2
            - np.log(magnitude) - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        )
        pdf[on_support] = np.exp(log_pdf)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    ax.plot(x, pdf)
    if xmin <= 0 <= xmax:
        ax.axvline(0, ls="--", c="black", lw=1)
    ax.set(
        xlim=(xmin, xmax), ylim=(0, None),
        xlabel="Marginal utility" if xlabel is None else xlabel, ylabel="PDF",
        title=f"{dist.replace('_', ' ')}\nmean = {mean:.3g}",
    )
    fig.tight_layout()
    return fig, ax
