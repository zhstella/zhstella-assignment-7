from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "zhstella"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)  # Replace with code to generate random values for X

    # TODO 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    error = np.random.normal(mu, np.sqrt(sigma2), N)
    # Y = beta0 + beta1 * X + mu + error term
    Y = beta0 + beta1 * X + error  # Replace with code to generate Y

    # TODO 3: Fit a linear regression model to X and Y
    model = LinearRegression().fit(X.reshape(-1, 1), Y)  # Initialize the LinearRegression model
    # None  # Fit the model to X and Y
    slope = model.coef_[0]  # Extract the slope (coefficient) from the fitted model
    intercept = model.intercept_  # Extract the intercept from the fitted model
    # TODO 4: Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    # Replace with code to generate and save the scatter plot
    plt.figure()
    plt.scatter(X, Y, color="blue", label="Data points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Regression line")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # TODO 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N)
        error_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + error_sim

        # TODO 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim) # Replace with code to fit the model
        sim_slope = sim_model.coef_[0]  # Extract slope from sim_model
        sim_intercept = sim_model.intercept_  # Extract intercept from sim_model

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # TODO 8: Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    # Replace with code to generate and save the histogram plot
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.5, label="Simulated Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, label="Simulated Intercepts")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # TODO 9: Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slopes"] = slopes
        session["slope"] = slope
        session["intercepts"] = intercepts
        session["intercept"] = intercept
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slopes = np.array(session.get("slopes"))
    intercepts = np.array(session.get("intercepts"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    beta1 = float(session.get("beta1"))
    beta0 = float(session.get("beta0"))

    parameter = request.form.get("parameter")  # 'slope' or 'intercept'
    test_type = request.form.get("test_type")  # 'greater', 'less', or 'not_equal'

    # Set observed statistic and hypothesized value based on user selection
    if parameter == "slope":
        simulated_stats = slopes
        observed_stat = slope
        hypothesized_value = beta1
        parameter_name = "Slope"
    else:
        simulated_stats = intercepts
        observed_stat = intercept
        hypothesized_value = beta0
        parameter_name = "Intercept"

    # TODO 10: Calculate p-value based on test type
    if test_type == "greater":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "less":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:  # two-tailed
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))

    # TODO 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    fun_message = "Wow, that's rare!" if p_value <= 0.0001 else None

    # TODO 12: Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    # Replace with code to generate and save the plot
    plt.figure()
    plt.hist(simulated_stats, bins=20, alpha=0.5, label="Simulated Statistics")
    plt.axvline(observed_stat, color="red", linestyle="dashed", linewidth=2, label=f"Observed {parameter_name}: {observed_stat:.4f}")
    plt.axvline(hypothesized_value, color="blue", linestyle="solid", linewidth=2, label=f"Hypothesized {parameter_name} (H₀): {hypothesized_value:.4f}")
    plt.legend()
    plt.title(f"Hypothesis Test for {parameter_name}")
    plt.xlabel(parameter_name)
    plt.ylabel("Frequency")
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()


    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        # TODO 13: Uncomment the following lines when implemented
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
        parameter_name = "Slope"
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0
        parameter_name = "Intercept"

    # TODO 14: Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    # TODO 15: Calculate confidence interval for the parameter estimate
    # Use the t-distribution and confidence_level

    dof = len(estimates) - 1  # degrees of freedom
    t_value = t.ppf(1 - (1 - confidence_level/100) / 2, dof)
    margin_of_error = t_value * std_estimate / np.sqrt(len(estimates))

    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error

    # TODO 16: Check if confidence interval includes true parameter
    includes_true = (ci_lower <= true_param) and (true_param <= ci_upper)

    # TODO 17: Plot the individual estimates as gray points and confidence interval
    # Plot the mean estimate as a colored point which changes if the true parameter is included
    # Plot the confidence interval as a horizontal line
    # Plot the true parameter value
    plot4_path = "static/plot4.png"
    # Write code here to generate and save the plot
    plt.figure()
    plt.plot(estimates, np.zeros_like(estimates), 'o', color="gray", alpha=0.5, label="Simulated Estimates")
    plt.axvline(mean_estimate, color="green", linestyle="dashed", linewidth=2, label="Mean Estimate")
    plt.hlines(0, ci_lower, ci_upper, color="blue", linewidth=4, label=f"{confidence_level}% Confidence Interval")
    plt.axvline(true_param, color="red", linestyle="dotted", linewidth=2, label="True Parameter")
    plt.legend()
    plt.title(f"{confidence_level}% Confidence Interval for {parameter_name} (Mean Estimate)")
    plt.xlabel(f"{parameter_name} Estimate")
    plt.savefig("static/plot4.png")
    plt.close()
    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
