import random
import numpy as np
import Math.Vector as v

class TridiagonalMatrix:
    def __init__(self, a=None, b=None, c=None, size=0):
        """
        Initialize tridiagonal matrix
        a, b, c - vectors of lower, main and upper diagonals
        Element indexing starts from 1
        a[1] = 0, c[n] = 0
        """
        if a is not None and b is not None and c is not None:
            self.a = v.Vector(a)  # lower diagonal
            self.b = v.Vector(b)  # main diagonal
            self.c = v.Vector(c)  # upper diagonal
            self.size = len(b)

            # Check boundary conditions
            if abs(self.a[1]) > 1e-12:  # a[1] should be 0
                print("Warning: a[1] should be 0")
            if abs(self.c[self.size]) > 1e-12:  # c[n] should be 0
                print("Warning: c[n] should be 0")
        else:
            self.size = size
            self.a = v.Vector(size=size)
            self.b = v.Vector(size=size)
            self.c = v.Vector(size=size)

            # Set boundary conditions
            if size > 0:
                self.a[1] = 0.0
                self.c[size] = 0.0


    def __add__(self, other):
        """Tridiagonal matrix addition"""
        if self.size != other.size:
            raise ValueError("Matrix sizes don't match for addition")

        result = TridiagonalMatrix(size=self.size)
        for i in range(1, self.size + 1):
            result.a[i] = self.a[i] + other.a[i]
            result.b[i] = self.b[i] + other.b[i]
            result.c[i] = self.c[i] + other.c[i]

        return result


    def __sub__(self, other):
        """Tridiagonal matrix subtraction"""
        if self.size != other.size:
            raise ValueError("Matrix sizes don't match for subtraction")

        result = TridiagonalMatrix(size=self.size)
        for i in range(1, self.size + 1):
            result.a[i] = self.a[i] - other.a[i]
            result.b[i] = self.b[i] - other.b[i]
            result.c[i] = self.c[i] - other.c[i]

        return result


    def multiply_vector(self, vector):
        """Multiply matrix by vector"""
        if self.size != len(vector):
            raise ValueError("Matrix and vector sizes don't match")

        result = v.Vector(size=self.size)

        # First row: b1*x1 + c1*x2
        result[1] = self.b[1] * vector[1] + self.c[1] * vector[2]

        # Last row: an*x_{n-1} + bn*xn
        result[self.size] = self.a[self.size] * vector[self.size - 1] + self.b[self.size] * vector[self.size]

        # Middle rows: ai*x_{i-1} + bi*xi + ci*x_{i+1}
        for i in range(2, self.size):
            result[i] = (self.a[i] * vector[i - 1] +
                         self.b[i] * vector[i] +
                         self.c[i] * vector[i + 1])

        return result


    def __matmul__(self, other):
        """Overload @ operator for vector multiplication"""
        return self.multiply_vector(other)


    def fill_random(self, low=-5.0, high=5.0, diagonally_dominant=False):
        """Fill matrix with random numbers - OPTIMIZED VERSION"""

        if diagonally_dominant:
            self._fill_random_diagonally_dominant(low, high)
        else:
            self._fill_random_regular(low, high)

        # Boundary conditions
        self.a[1] = 0.0
        self.c[self.size] = 0.0

        return self


    def _fill_random_regular(self, low, high):
        """Regular random generation without conditions in loop"""
        # Main diagonal (all elements)
        for i in range(1, self.size + 1):
            self.b[i] = random.uniform(low, high)

        # Lower diagonal (except first element)
        for i in range(2, self.size + 1):
            self.a[i] = random.uniform(low, high)

        # Upper diagonal (except last element)
        for i in range(1, self.size):
            self.c[i] = random.uniform(low, high)


    def _fill_random_diagonally_dominant(self, low, high):
        """Diagonally dominant generation"""
        # Generate base random values
        for i in range(1, self.size + 1):
            self.b[i] = random.uniform(abs(high), 2 * abs(high))

        for i in range(2, self.size + 1):
            self.a[i] = random.uniform(low, high)

        for i in range(1, self.size):
            self.c[i] = random.uniform(low, high)

        # Enhance diagonal dominance
        # First row
        if self.size >= 1:
            self.b[1] = abs(self.b[1]) + abs(self.c[1])

        # Last row
        if self.size >= 2:
            self.b[self.size] = abs(self.b[self.size]) + abs(self.a[self.size])

        # Middle rows
        for i in range(2, self.size):  # i = 2, 3, ..., n-1
            self.b[i] = abs(self.b[i]) + abs(self.a[i]) + abs(self.c[i])


    def read_from_console(self):
        """Read matrix from console"""
        print(f"Enter elements of tridiagonal matrix {self.size}x{self.size}:")

        for i in range(1, self.size + 1):
            if i > 1:
                self.a[i] = float(input(f"a[{i}] (element [{i},{i - 1}]): "))
            self.b[i] = float(input(f"b[{i}] (element [{i},{i}]): "))
            if i < self.size:
                self.c[i] = float(input(f"c[{i}] (element [{i},{i + 1}]): "))

        # Boundary conditions
        self.a[1] = 0.0
        self.c[self.size] = 0.0

        return self


    def read_from_file(self, filename):
        """Read matrix from file"""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()

                a_data = []
                b_data = []
                c_data = []

                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            a_data.append(float(parts[0]))
                            b_data.append(float(parts[1]))
                            c_data.append(float(parts[2]))

                if len(b_data) == 0:
                    print("File contains no data")
                    return None

                self.size = len(b_data)
                self.a = v.Vector(a_data)
                self.b = v.Vector(b_data)
                self.c = v.Vector(c_data)

                # Boundary conditions
                self.a[1] = 0.0
                self.c[self.size] = 0.0

                return self

        except FileNotFoundError:
            print(f"File {filename} not found")
            return None


    def write_to_console(self):
        """Print matrix to console"""
        print(f"Tridiagonal matrix {self.size}x{self.size}:")
        print(f"|{'i':^3}|{'a':^12}|{'b':^12}|{'c':^12}|")

        for i in range(1, self.size + 1):
            a_val = self.a[i] if i > 1 else 0.0
            c_val = self.c[i] if i < self.size else 0.0
            print(f"|{i:^3}|{a_val:^12.6f}|{self.b[i]:^12.6f}|{c_val:^12.6f}|")

        return self


    def write_to_file(self, filename):
        """Write matrix to file"""
        with open(filename, 'w') as f:
            f.write("# Tridiagonal matrix\n")
            f.write("# Format: a[i] b[i] c[i]\n")
            f.write("# a[1] = 0, c[n] = 0\n")

            for i in range(1, self.size + 1):
                f.write(f"{self.a[i]:.6f} {self.b[i]:.6f} {self.c[i]:.6f}\n")

        return self

    def to_full_matrix(self):
        """Преобразование в полную матрицу (для проверки)"""
        full_matrix = np.zeros((self.size, self.size))

        for i in range(1, self.size + 1):
            if i > 1:
                full_matrix[i - 1, i - 2] = self.a[i]  # a[i] -> (i, i-1)
            full_matrix[i - 1, i - 1] = self.b[i]  # b[i] -> (i, i)
            if i < self.size:
                full_matrix[i - 1, i] = self.c[i]  # c[i] -> (i, i+1)

        return full_matrix

    def get_cond(self):
        """"Вычисляет число обусловленности det(A) * det(A^-1)"""
        full_matrix = self.to_full_matrix()
        return np.linalg.cond(full_matrix)

    def __str__(self):
        """String representation of matrix"""
        result = ""
        for i in range(1, self.size + 1):
            a_val = self.a[i] if i > 1 else 0.0
            c_val = self.c[i] if i < self.size else 0.0
            result += f"{a_val:10.6f} {self.b[i]:10.6f} {c_val:10.6f}\n"
        return result
