import math
import numpy as np
import matplotlib.pyplot as plt


def product(iterable):
    prod = 1
    for e in iterable:
        prod *= e
    return prod


def newtonDD(points):
    """
    Recursively computes Newton's Divided Differences given a list of points to interpolate.
    :param points: a list of (x,y) pairs
    :return: the result of Newton's DD formula, a real number
    """

    if len(points) == 1:
        # If only one point is passed, return its corresponding y-value
        return points[0][1]
    else:
        # Otherwise, compute Newton's DD recursive formula
        return (newtonDD(points[1:]) - newtonDD(points[:-1])) / (points[-1][0] - points[0][0])


def print_newtonDD_triangle(points, precision=None):
    """
    Neatly prints Newton's DD triangle given a list of points to interpolate.
    :param points: an iterable of (x,y) pairs
    """

    # 1) Compute values in triangle

    n = len(points)

    # Generate and store the columns of the triangle as lists in a dict
    cols = dict()
    for k in range(1, n+1):
        col = []
        for j in range(n-k + 1):
            dd = newtonDD(points[j: j + k])
            col.append(dd if precision is None else round(dd, precision))
        cols[k] = col

    # 2) Format output

    x_list = [p[0] for p in points]

    # Set width of the column of x-values
    x_width = 1
    for x in x_list:
        width = len(str(x))
        if width > x_width:
            x_width = width
    x_width += 1

    # Set width of columns in triangle
    col_width = 1
    for col in range(1, n+1):
        for num in cols[col]:
            width = len(str(num))
            if width > col_width:
                col_width = width
    col_width += 1

    padding = ' ' * col_width  # padding between values in a row

    # 3) Print triangle

    # Print row by row; we will have 2*n - 1 rows where odd rows (indexed from 1) start with an x value
    for i in range(1, 2*n):
        num_values = ((n - abs(i-n)) + (i % 2)) // 2  # number of values in row

        # If row # is odd
        if i % 2 == 1:
            x = x_list[(i + 1)//2 - 1]
            row = str(x).ljust(x_width) + '| '

            for j in range(num_values):
                col = 2*j + 1  # column number from left to right (only odd columns here)
                idx = (i+1)//2 - j - 1  # index in column list
                row += str(cols[col][idx]).ljust(col_width) + padding

            row = row.rstrip()

        # If row # is even
        else:
            row = (' ' * x_width) + '| '

            for j in range(num_values):
                col = 2*j + 2  # column number from left to right (only even columns here)
                idx = i//2 - j - 1  # index in column list
                row += padding + str(cols[col][idx]).ljust(col_width)

        print(row)


def get_coefs(points):
    return [newtonDD(points[:k]) for k in range(1, len(points) + 1)]


def evaluate_interpolant(x, points):
    x_list = [p[0] for p in points]
    coefs = get_coefs(points)
    return coefs[0] + sum(c * product(x - x_i for x_i in x_list[:i]) for i, c in enumerate(coefs[1:], 1))


class InterpolatingPolynomial:
    def __init__(self, points):
        self.points = points
        self.degree = len(points) - 1
        self.coefs = get_coefs(points)

        x_list = [p[0] for p in self.points]
        coefs = self.coefs
        func_str = str(coefs[0]) + ' + '
        func_str += ' + '.join(
            f'{c}*{"*".join(f"(x - {x_i})" for x_i in x_list[:i])}' for i, c in enumerate(coefs[1:], 1)
        )
        self.formula = func_str

    def __call__(self, x):
        return evaluate_interpolant(x, self.points)

    def __str__(self):
        return self.formula

    def eval(self, x):
        return self.__call__(x)

    def error_at(self, x, fxn):
        return abs(fxn(x) - self(x))

    def plot(self,
             x_min=None, x_max=None, y_min=None, y_max=None,
             fxn=None, fxn_label=None,
             title=None, x_label='x', y_label='y'
             ):

        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]

        # Determine appropriate x and y bounds for plot window
        # Design decision (implemented below):
        #  - extend plot window horizontally (on both sides) by 1/3 of the points' x-range
        #  - extend plot window vertically (on both sides) by 1/3 of the points' y-range

        min_x = min(x_coords)
        max_x = max(x_coords)
        x_range = max_x - min_x

        min_y = min(y_coords)
        max_y = max(y_coords)
        y_range = max_y - min_y

        if x_min is None:
            x_min = min_x - (x_range / 3)

        if x_max is None:
            x_max = max_x + (x_range / 3)

        if y_min is None:
            y_min = min_y - (y_range / 3)

        if y_max is None:
            y_max = max_y + (y_range / 3)

        # Generate many (x,y) points to plot a seemingly smooth curve for the interpolant
        x = np.linspace(x_min, x_max, 10000)
        y = np.array([self(i) for i in x])

        # Plot
        plt.plot(x, y, color='red', linewidth=1, zorder=3, label='interpolant')  # plot interpolant
        plt.scatter(x=x_coords, y=y_coords, color='blue', s=15, zorder=4)  # plot initial points
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))

        # If a function is given, plot it to compare with the interpolant
        if fxn is not None:
            plt.plot(x, [fxn(i) for i in x], color='royalblue', linewidth=1, zorder=2, label=(fxn_label or 'function'))
            plt.legend()

        # Title, labels, and x/y axes
        if title is None:
            title = f'Interpolating Polynomial (degree={self.degree})'
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.axvline(x=0, color='black', linewidth=0.5, zorder=0)  # plot x-axis
        plt.axhline(y=0, color='black', linewidth=0.5, zorder=1)  # plot y-axis

        plt.show()

    def plot_error(self, fxn,
                   x_min=None, x_max=None, y_min=None, y_max=None,
                   title=None, x_label='x', y_label='error'
                   ):

        x_coords = [p[0] for p in self.points]

        # Determine appropriate x-bounds for plot window
        # Design decision (implemented below):
        #  - extend plot window horizontally (on both sides) by 1/3 of the points' x-range

        min_x = min(x_coords)
        max_x = max(x_coords)
        x_range = max_x - min_x

        if x_min is None:
            x_min = min_x - (x_range / 3)

        if x_max is None:
            x_max = max_x + (x_range / 3)

        # Determine appropriate y-bounds for plot window
        # Design decision (implemented below):
        #  - set initial y-bounds of plot window to max/min of the error inside the domain of the points
        #  - then extend plot window vertically (on both sides) by 1/3 of the error range

        x_coords_smooth = np.linspace(x_coords[0], x_coords[-1], 1000)
        error_in_interval = [self.error_at(i, fxn) for i in x_coords_smooth]

        min_error = min(error_in_interval)
        max_error = max(error_in_interval)
        error_range = max_error - min_error

        if y_min is None:
            y_min = min_error - (error_range / 3)

        if y_max is None:
            y_max = max_error + (error_range / 3)

        # Generate many (x,y) points to plot a seemingly smooth curve for the error
        x = np.linspace(x_min, x_max, 10000)
        error = np.array([self.error_at(i, fxn) for i in x])

        # Plot
        plt.plot(x, error, color='magenta', linewidth=1, zorder=2)
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))

        # Title, labels, and x/y axes
        if title is None:
            title = f'Interpolant error'
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.axvline(x=0, color='black', linewidth=0.5, zorder=0)  # plot x-axis
        plt.axhline(y=0, color='black', linewidth=0.5, zorder=1)  # plot y-axis

        plt.show()


def main():
    print('Code demo:\n')

    # points = [(0, 1), (2, 2), (3, 4), (1, 0)]

    f = lambda t: math.cos(t)
    points = [(i, f(i)) for i in np.linspace(0, math.pi, 4)]

    print('Points:')
    print(*points, sep=', ')
    print()

    print("Newton's DD Triangle:")
    print_newtonDD_triangle(points)
    print()

    print("Evaluate interpolant:")
    P = InterpolatingPolynomial(points)
    print(f'P(x) = {P}')

    for p in points:
        x = p[0]
        print(f'P({x}) = {P(x)}')
    print()

    print('Plot interpolant and initial points:')
    print('(plot opened in new window)')
    P.plot(fxn=f, fxn_label='cos(x)', x_min=-math.pi, x_max=2*math.pi)
    print()

    print('Plot error:')
    print('(plot opened in new window)')
    P.plot_error(f)


if __name__ == '__main__':
    main()
