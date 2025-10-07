import numpy as np
import pandas as pd

import Math.Vector as v
import Math.TridiagonalMatrix as tm

class DataGenerator:
    def __init__(self,
                 matr_low = -1.0, matr_high = 1.0,
                 x_low = -1.0, x_high = 1.0, size = 5):
        self.matrix = tm.TridiagonalMatrix(size=size)
        self.matrix.fill_random(matr_low, matr_high)

        self.vector = v.Vector(size=size)
        self.vector.fill_random(x_low, x_high)

    def get_data(self):
        return self.matrix, self.vector

    def __str__(self):
        print(self.matrix)
        print(self.vector)


class ExperimentUtils:
    @staticmethod
    def progonka(matrix, exact_x):
        d = matrix @ exact_x    # right hand side
        l = v.Vector(np.zeros(matrix.size + 1))
        m = v.Vector(np.zeros(matrix.size + 1))

        # straight
        l[2] = matrix.c[1] / matrix.b[1]
        m[2] = d[1] / matrix.b[1]

        for i in range(2, matrix.size + 1):
            if matrix.b[i] != matrix.a[i] * l[i]:
                l[i + 1] = matrix.c[i] / (matrix.b[i] - matrix.a[i] * l[i])
                m[i + 1] = (d[i] - matrix.a[i] * m[i]) / (matrix.b[i] - matrix.a[i] * l[i])
            else:
                raise ZeroDivisionError("b[i] must not be equal to a[i] * L[i]")

        # reverse
        approximate_x = v.Vector(np.zeros(matrix.size))
        approximate_x[matrix.size] = m[matrix.size + 1]
        for i in range(matrix.size - 1, 0, -1):
            approximate_x[i] = m[i + 1] - l[i + 1] * approximate_x[i + 1]

        return approximate_x


    @staticmethod
    def unstable(matrix, exact_x):
        d = matrix @ exact_x    # right hand side
        n = matrix.size
        y = v.Vector(np.zeros(n + 1))
        z = v.Vector(np.zeros(n + 1))

        y[1] = 0.0
        y[2] = d[1] / matrix.c[1]

        z[1] = 1.0
        z[2] = -(matrix.b[1]) / matrix.c[1]

        for i in range(2, n):
            y[i + 1] = (d[i] - matrix.a[i] * y[i - 1] - matrix.b[i] * y[i]) / matrix.c[i]
            z[i + 1] = -(matrix.a[i] * z[i - 1] + matrix.b[i] * z[i]) / matrix.c[i]

        const_k = ((d[n] - matrix.a[n] * y[n - 1] - matrix.b[n] * y[n]) /
                   (matrix.a[n] * z[n - 1] + matrix.b[n] * z[n]))

        approximate_x = v.Vector(np.zeros(matrix.size))

        for i in range(1, n + 1):
            approximate_x[i] = y[i] + const_k * z[i]

        return approximate_x


    @staticmethod
    def calculate_error(exact_x, approximate_x):
        absolute_errors = v.Vector(size=exact_x.size)
        relative_errors = v.Vector(size=exact_x.size)

        for i in range(1, exact_x.size + 1):
            abs_err = abs(exact_x[i] - approximate_x[i])
            absolute_errors[i] = abs_err

            # Защита от деления на ноль
            if abs(exact_x[i]) > np.finfo(float).eps:
                relative_errors[i] = abs_err / abs(exact_x[i])
            else:
                relative_errors[i] = abs_err  # если точное значение близко к 0

        absolute_error = absolute_errors.norm()
        relative_error = relative_errors.norm()

        return absolute_error, relative_error


def main_experiment(exp_count: int,
                    matr_low = -1.0, matr_high = 1.0,
                    x_low = -1.0, x_high = 1.0):
    data_dict = {
        'system size': [],
        'filling range': [],
        'absolute error (A)': [],
        'relative error (A)': [],
        'absolute error (B)': [],
        'relative error (B)': [],
        'conditioning': []
    }

    for size in np.logspace(1, 6, 6, base=2).astype(int):
        for i in range(exp_count):
            data = DataGenerator(matr_low, matr_high, x_low, x_high, size)
            matrix, exact_x = data.get_data()

            approx_x_a = ExperimentUtils.progonka(matrix, exact_x)
            approx_x_b = ExperimentUtils.unstable(matrix, exact_x)

            absolute_error_a, relative_error_a = ExperimentUtils.calculate_error(exact_x, approx_x_a)
            absolute_error_b, relative_error_b = ExperimentUtils.calculate_error(exact_x, approx_x_b)

            cond = matrix.get_cond()

            data_dict['system size'].append(size)
            data_dict['filling range'].append([matr_low, matr_high])
            data_dict['absolute error (A)'].append(absolute_error_a)
            data_dict['relative error (A)'].append(relative_error_a)
            data_dict['absolute error (B)'].append(absolute_error_b)
            data_dict['relative error (B)'].append(relative_error_b)
            data_dict['conditioning'].append(cond)

    df = pd.DataFrame(data_dict)
    return df


def print_test():
    matrix = tm.TridiagonalMatrix(np.array([0.0, 2.0, 5.0, 2.0, 5.0]),
                                  np.array([4.0, 6.0, 5.0, 11.0, 8.0]),
                                  np.array([2.0, 1.0, 2.0, 2.0, 0.0]), size=5)
    
    exact_x = v.Vector(np.array([1.0, -1.0, 2.0, -1.0, 1.0]), size=5)

    approx_x_a = ExperimentUtils.progonka(matrix, exact_x)
    approx_x_b = ExperimentUtils.unstable(matrix, exact_x)

    absolute_error_a, relative_error_a = ExperimentUtils.calculate_error(exact_x, approx_x_a)
    absolute_error_b, relative_error_b = ExperimentUtils.calculate_error(exact_x, approx_x_b)

    print(f"Exact x:         {exact_x}")
    print(f"Calculated by A: {approx_x_a}")
    print(f"Calculated by B: {approx_x_b}")
    print(f"absolute_error_a: {absolute_error_a:10e}, relative_error_a: {relative_error_a:10e}")
    print(f"absolute_error_b: {absolute_error_b:10e}, relative_error_a: {relative_error_b:10e}")



if __name__ == '__main__':
    # print_test()
    df1 = main_experiment(3, -1.0, 1.0, -1.0, 1.0)
    df10 = main_experiment(3, -10.0, 10.0, -10.0, 10.0)
    df100 = main_experiment(3, -100.0, 100.0, -100.0, 100.0)
    df1000 = main_experiment(3, -1000.0, 1000.0, -1000.0, 1000.0)

    df = pd.concat([df1, df10, df100, df1000], ignore_index=True)

    df.to_csv('results/results.csv', index=False)
    print('"results.csv" created')
