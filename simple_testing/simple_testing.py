import numpy as np
import matplotlib.pyplot as plt


def run(f, epochs, optimizer, epsilon, is_verbose):
    print(f.as_string() + "\n")

    steps_to_fosp = None
    steps_to_sosp = None

    x = f.random_init()
    f_x_vector = [f.eval(x)]
    if is_verbose:
        print_epoch(0, f, x, epsilon)

    for e in range(1, epochs + 1):
        x = optimizer.step_not_ml(f, x, is_verbose)
        f_x_vector.append(f.eval(x))
        if is_verbose:
            print_epoch(e, f, x, epsilon)

        if steps_to_fosp == None and is_first_order_stationary_point(f, x, epsilon):
            steps_to_fosp = e
        if steps_to_sosp == None and is_second_order_stationary_point(f, x, epsilon):
            steps_to_sosp = e

    if steps_to_fosp == None:
        print("Did not converge to a " + str(epsilon) + "-first-order stationary-point.")
    else:
        print(
            "Converged to a " + str(epsilon) + "-first-order stationary-point after " + str(steps_to_fosp) + " steps.")
    if steps_to_sosp == None:
        print("Did not converge to a " + str(epsilon) + "-second-order stationary-point.")
    else:
        print(
            "Converged to a " + str(epsilon) + "-second-order stationary-point after " + str(steps_to_sosp) + " steps.")

    plot_results(epochs, f_x_vector)


def plot_results(epochs, f_x_vector):
    fig = plt.figure()
    plt.plot(range(epochs + 1), f_x_vector)
    plt.xlabel('epoch')
    plt.ylabel('f(x)')
    plt.show()


def print_epoch(e, f, x, epsilon):
    print("x" + str(e) + " = " + str(x))
    print("f(x" + str(e) + ") = " + str(f.eval(x)))
    print("grad f(x" + str(e) + ") = " + str(f.grad(x)))
    evals, _ = np.linalg.eig(f.hessian(x))
    print("evals(grad^2 f(x" + str(e) + ")) = " + str(evals))
    if (is_second_order_stationary_point(f, x, epsilon)):
        print("Second-order stationary point for epsilon = " + str(epsilon) + "\n")
    elif (is_first_order_stationary_point(f, x, epsilon)):
        print("First-order stationary point for epsilon = " + str(epsilon) + "\n")
    else:
        print("Not a stationary point for epsilon = " + str(epsilon) + "\n")


def is_first_order_stationary_point(f, x, epsilon):
    return np.linalg.norm(f.grad(x), ord=2) < epsilon


def is_second_order_stationary_point(f, x, epsilon):
    evals, _ = np.linalg.eig(f.hessian(x))
    return is_first_order_stationary_point(f, x, epsilon) and np.min(evals) > -1 * epsilon
