import math


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


def print_newtonDD_triangle(points):
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
            col.append(dd)
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


def main():
    print('Code demo:\n')

    points = [(0, 1), (2, 2), (3, 4), (1, 0)]

    print('Points:')
    print(*points, sep=', ')
    print()

    print("Newton's DD Triangle:")
    print_newtonDD_triangle(points)
    print()

    print("Evaluating interpolant:")
    p = InterpolatingPolynomial(points)
    print(f'p(x) = {p}\n')

    for x in [0, 2, 3, 1, math.pi]:
        print(f'x = {x}, p({x}) = {p(x)}')
    print()


if __name__ == '__main__':
    main()
