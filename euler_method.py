import math
import numpy as np
import matplotlib.pyplot as plt


def euler_method(w, t, f, h):
    return w + h*f(t, w)


def get_w_list(y, f, h, interval):
    a, b = interval
    t_list = np.linspace(a, b, round((b-a) / h) + 1)
    w_list = [y(t_list[0])]
    for i in range(1, len(t_list)):
        w = euler_method(w_list[i-1], t_list[i-1], f, h)
        w_list.append(w)
    return w_list


def plot_euler_method(y, f, h, interval):
    a, b = interval
    t_list = np.linspace(a, b, round((b-a) / h) + 1)
    w_list = get_w_list(y, f, h, interval)
    y_true = [y(t) for t in t_list]

    t_smooth = np.linspace(a, b, 1000)
    y_smooth = [y(t) for t in t_smooth]

    plt.plot(t_list, w_list, 'o-', color='red', label='approximation', zorder=2)
    plt.plot(t_smooth, y_smooth, color='blue', label='true', zorder=1)
    plt.scatter(t_list, y_true, color='blue', zorder=0)

    plt.title(f"Euler's Method on [{a}, {b}], h={h}")
    plt.legend()
    plt.show()


def main():
    print('Testing code:')

    e = math.e

    h_list = [0.1, 0.05, 0.025]
    y = lambda t: t ** 2 * (e ** t - e)
    f = lambda t, y_: (2 / t) * y_ + t ** 2 * e ** t

    for h in h_list:
        plot_euler_method(y, f, h, (1, 2))


if __name__ == '__main__':
    main()
