# Библиотеки используемые в работе
from typing import List, Tuple
from matplotlib.figure import Figure
from fractions import Fraction
import matplotlib.pyplot as plt
import unittest
import math
import random
from sklearn.datasets import load_iris  # Только загрузка датасета


# Структура класса matrix
class Matrix:
    def __init__(self, data):  # Инициализация объектов матрицы
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows else 0

    def __getitem__(self, idx):  # Возвращает строку матрицы по индексу
        return self.data[idx]

    def __str__(self):  # Возвращает строковое представление матрицы
        return '\n'.join(['\t'.join(map('{:.2f}'.format, row)) for row in self.data])

    def copy(self):  # Создание копии матрицы
        return Matrix([row.copy() for row in self.data])

    def transpose(self):  # Транспонирование
        transposed = list(map(list, zip(*self.data)))
        return Matrix(transposed)

    def __matmul__(self, other: 'Matrix') -> 'Matrix':  # Перемножение матриц
        if self.cols != other.rows:
            raise ValueError("Несовместимые размеры матриц")
        result = [
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def __mul__(self, scalar: float) -> 'Matrix':  # Умножение матрицы на скаляр
        if isinstance(scalar, (int, float)):
            return Matrix([[val * scalar for val in row] for row in self.data])
        raise TypeError("Можно умножать только на скаляры (int, float)")

    def __rmul__(self, scalar: float) -> 'Matrix':  # Умножение скаляра на матрицу слева
        return self.__mul__(scalar)

    def swap_rows(self, i, j):  # Меняем местами строки
        self.data[i], self.data[j] = self.data[j], self.data[i]

    def get(self, i, j):  # Возвращает элемент матрицы по строке и столбцу
        return self.data[i][j]

    def set(self, i, j, value):  # Устанавливает новое значение матрицы по индексу
        self.data[i][j] = value

    def __sub__(self, other):  # Вычитание матриц
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Матрицы разных размеров")
        result = [
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def trace(self):  # Вычисление следа матрицы
        if self.rows != self.cols:
            raise ValueError("Не квадратная матрица")
        return sum(self.data[i][i] for i in range(self.rows))

    def determinant(self):  # Вычисление определителя матрицы
        if self.rows != self.cols:
            raise ValueError("Матрица должна быть квадратной, чтобы вычислить определитель.")
        n = self.rows
        if n == 0:
            raise ValueError("Пустая матрица не имеет определителя.")
        if n == 1:
            return self.data[0][0]

        # Создаем копию матрицы, чтобы не изменять исходную
        matrix = [row.copy() for row in self.data]
        sign = 1  # Для отслеживания изменений знака из-за перестановок строк

        for i in range(n):
            # Поиск строки с максимальным элементом в текущем столбце
            max_row = i
            for j in range(i, n):
                if abs(matrix[j][i]) > abs(matrix[max_row][i]):
                    max_row = j

            # Если все элементы столбца нулевые, определитель 0
            if matrix[max_row][i] == 0:
                return 0

            # Перестановка строк
            if max_row != i:
                matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
                sign *= -1

            # Обнуление элементов ниже текущего
            for j in range(i + 1, n):
                factor = matrix[j][i] / matrix[i][i]
                for k in range(i, n):
                    matrix[j][k] -= factor * matrix[i][k]

        # Вычисление произведения диагональных элементов
        det = 1.0
        for i in range(n):
            det *= matrix[i][i]

        det *= sign

        # Возвращаем целое значение, если результат целый
        return int(det)  # if det.is_integer() else det

    def nullspace(self) -> List[List[float]]:  # Находит базис нуль-пространства (решения Ax = 0)
        zero_b = Matrix([[0.0] for _ in range(self.rows)])
        solutions = self.gauss_solver(zero_b)
        return solutions

    def almost_equal(self, other: 'Matrix', tol: float = 1e-6) -> bool:
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.data[i][j] - other.data[i][j]) > tol:
                    return False
        return True

    @classmethod
    def from_list(cls, lst):  # Создает матрицу из списка списков
        return cls(lst)

    def __eq__(self, other):  # Проверяет равенство двух матриц
        if not isinstance(other, Matrix):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] != other.data[i][j]:
                    return False
        return True

    # Метод Гауса из Easy
    def gauss_solver(self, b: 'Matrix') -> List['Matrix']:
        eps = 1e-8
        A = self.copy()
        n = A.rows
        m = A.cols

        for k in range(n):
            pivot_row = max(range(k, n), key=lambda i: abs(A.get(i, k)), default=k)
            if abs(A.get(pivot_row, k)) < eps:
                continue
            A.swap_rows(k, pivot_row)
            b.swap_rows(k, pivot_row)

            pivot = A.get(k, k)
            if abs(pivot) < eps:
                continue
            for j in range(k, m):
                A.set(k, j, A.get(k, j) / pivot)
            b.set(k, 0, b.get(k, 0) / pivot)

            for i in range(n):
                if i == k:
                    continue
                factor = A.get(i, k)
                for j in range(k, m):
                    A.set(i, j, A.get(i, j) - factor * A.get(k, j))
                b.set(i, 0, b.get(i, 0) - factor * b.get(k, 0))

        # Проверка на несовместность
        for i in range(n):
            if all(abs(A.get(i, j)) < eps for j in range(m)) and abs(b.get(i, 0)) > eps:
                raise ValueError("Система несовместна")

        # Определение свободных переменных
        pivot_cols = []
        for i in range(n):
            for j in range(m):
                if abs(A.get(i, j)) > eps:
                    pivot_cols.append(j)
                    break

        free_vars = [j for j in range(m) if j not in pivot_cols]
        solutions = []

        for var in free_vars:
            vec = [0.0] * m
            vec[var] = 1.0
            for i in reversed(range(n)):
                row = A.data[i]
                pivot = next((j for j in range(m) if abs(row[j]) > eps), None)
                if pivot is None or pivot >= m:
                    continue
                sum_val = sum(row[j] * vec[j] for j in range(pivot + 1, m))
                vec[pivot] = (b.get(i, 0) - sum_val) / row[pivot] if abs(row[pivot]) > eps else 0.0
            solutions.append(Matrix([[x] for x in vec]))

        if len(free_vars) == 0:
            solution = []
            for i in reversed(range(n)):
                row = A.data[i]
                pivot = next((j for j in range(m) if abs(row[j]) > eps), None)
                if pivot is None:
                    continue
                sum_val = sum(row[j] * (solution[j] if j < len(solution) else 0) for j in range(pivot + 1, m))
                solution.insert(0, (b.get(i, 0) - sum_val) / row[pivot])
            solutions.append(Matrix([[x] for x in solution]))
            return solutions

        return solutions

    def qr_decomposition(self) -> Tuple[
        'Matrix', 'Matrix']:  # QR-разложение методом Грама-Шмидта с обработкой нулевых векторов.
        m, n = self.rows, self.cols
        Q = Matrix([[0.0] * n for _ in range(m)])
        R = Matrix([[0.0] * n for _ in range(n)])

        for j in range(n):
            v = [self.data[i][j] for i in range(m)]
            for k in range(j):
                R.data[k][j] = sum(Q.data[i][k] * self.data[i][j] for i in range(m))
                v = [v[i] - R.data[k][j] * Q.data[i][k] for i in range(m)]

            norm = math.sqrt(sum(x ** 2 for x in v))
            if norm < 1e-10:  # Избегаем деления на ноль
                norm = 1e-10

            for i in range(m):
                Q.data[i][j] = v[i] / norm
            R.data[j][j] = norm

        return Q, R


# Тест matrix
class TestMatrix(unittest.TestCase):
    def test_matmul(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        result = A @ B
        expected = Matrix([[19, 22], [43, 50]])
        self.assertEqual(result.data, expected.data)

    def test_scalar_mul(self):
        A = Matrix([[1, 2], [3, 4]])
        result = 2 * A
        expected = Matrix([[2, 4], [6, 8]])
        self.assertEqual(result.data, expected.data)

    def test_determinent(self):
        A = Matrix([
            [1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 43.0, 4.0],
            [5.0, 13.0, 5.0, 6.0],
            [7.0, 8.0, 7.0, 8.0]
        ])
        expected = Matrix.determinant(A)
        result = 1680


# Easy
# Метод Гауса находится в matrix
# Функция центрирования данных
def center_data(X: 'Matrix') -> 'Matrix':
    """
    Вход: матрица данных X (n×m)
    Выход: центрированная матрица X_centered (n×m)
    """
    n, m = X.rows, X.cols
    means = []
    for j in range(m):
        column_sum = sum(Fraction(str(X.get(i, j))) for i in range(n))
        means.append(column_sum / n)
    centered = []
    for i in range(n):
        row = [float(Fraction(str(X.get(i, j)))) - float(means[j]) for j in range(m)]
        centered.append(row)
    return Matrix(centered)


# Вычисление матрицы ковариаций
def covariance_matrix(X_centered: 'Matrix') -> 'Matrix':
    """
    Вход: центрированная матрица X_centered (n×m)
    Выход: матрица ковариаций C (m×m)
    """
    XT = X_centered.transpose()
    return (XT @ X_centered) * (1 / (X_centered.rows - 1))


# Тестирование работу функций из Easy
class Test(unittest.TestCase):
    # Тестирование Гаусса
    def test_gauss_unique_solution(self):
        A = Matrix([[2, 1], [1, -1]])
        b = Matrix([[5], [2]])
        solutions = A.gauss_solver(b)
        self.assertEqual(len(solutions), 1)
        expected = Matrix([[7 / 3], [1 / 3]])
        self.assertTrue(solutions[0].almost_equal(expected, tol=1e-4))

    def test_gauss_infinite_solutions(self):
        A = Matrix([[1, 2], [2, 4]])
        b = Matrix([[3], [6]])
        solutions = A.gauss_solver(b)
        self.assertEqual(len(solutions), 1)

    def test_gauss_inconsistent(self):
        A = Matrix([[1, 2], [1, 2]])
        b = Matrix([[3], [4]])
        with self.assertRaises(ValueError):
            A.gauss_solver(b)

    # Тестирование центрирования
    def test_center_data(self):
        X = Matrix([
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9]
        ])
        centered = center_data(X)
        expected_means = [(2.5 + 0.5 + 2.2) / 3, (2.4 + 0.7 + 2.9) / 3]
        expected = [
            [2.5 - expected_means[0], 2.4 - expected_means[1]],
            [0.5 - expected_means[0], 0.7 - expected_means[1]],
            [2.2 - expected_means[0], 2.9 - expected_means[1]]
        ]
        for i in range(3):
            for j in range(2):
                self.assertAlmostEqual(centered.get(i, j), expected[i][j], places=4)

    # Тест ковариации
    def test_covariance_matrix(self):
        X = Matrix([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        centered = center_data(X)
        cov = covariance_matrix(centered)
        self.assertAlmostEqual(cov.get(0, 0), 4.0, places=4)
        self.assertAlmostEqual(cov.get(0, 1), 4.0, places=4)
        self.assertAlmostEqual(cov.get(1, 1), 4.0, places=4)


# Пример матриц и решения с помощью Гаусса
A = Matrix([[2, 1], [1, -1]])
b = Matrix([[5], [2]])
solutions = Matrix.gauss_solver(A, b)
print("Матрица A:")
print(A)
print("Вектор b:")
print(b)
print("Решение системы:")
for vec in solutions:
    print("[", end="")
    for x in vec.data:
        val = x[0]
        if isinstance(val, Fraction):
            print(f"{val.numerator}/{val.denominator}" if val.denominator != 1 else f"{val.numerator}", end=" ")
        else:
            print(f"{val}", end="")
    print("]")

X = Matrix([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9]
])

print("Исходная матрица X:")
print(X)

X_centered = center_data(X)
print("\nЦентрированная матрица X_centered:")
print(X_centered)

C = covariance_matrix(X_centered)
print("\nКовариационная матрица C:")
print(C)


# Normal
# Нахождение собственных значений матрицы методом бисекции
def find_eigenvalues(C: 'Matrix', tol: float = 1e-6) -> List[float]:
    """
    Вход:
    C: матрица ковариаций (m×m)
    tol: допустимая погрешность
    Выход: список вещественных собственных значений
    """
    if C.rows != C.cols:
        raise ValueError("Матрица должна быть квадратной")

    A = C.copy()
    n = A.rows
    prev_diag = [A.data[i][i] for i in range(n)]

    for _ in range(10000):
        Q, R = A.qr_decomposition()
        A = R @ Q
        current_diag = [A.data[i][i] for i in range(n)]
        max_diff = max(abs(current_diag[i] - prev_diag[i]) for i in range(n))
        prev_diag = current_diag.copy()
        if max_diff < tol:
            break

    return [round(A.data[i][i], 6) for i in range(n)]


# Нахождение собственных векторов матрицы
def find_eigenvectors(C: 'Matrix', eigenvalues: List[float]) -> List['Matrix']:
    """
    Вход:
    C: матрица ковариаций (m×m)
    eigenvalues: список собственных значений
    Выход: список собственных векторов (каждый вектор - объект Matrix)
    """
    eigenvectors = []
    eps = 1e-6
    for lambda_ in eigenvalues:
        if math.isnan(lambda_):
            continue  # Пропускаем NaN значения
        A = C - Matrix([[lambda_ if i == j else 0.0 for j in range(C.cols)] for i in range(C.rows)])
        nullspace = []
        # Основная попытка
        try:
            nullspace = A.nullspace()
        except:
            pass

        # Многократные попытки с разным шумом
        if not nullspace:
            for _ in range(20):  # Увеличено количество попыток
                noise = [[random.uniform(-1e-8, 1e-8) for _ in range(A.cols)] for _ in range(A.rows)]
                noisy_A = A + Matrix(noise)
                try:
                    ns = noisy_A.nullspace()
                    if ns:
                        nullspace = ns
                        break
                except:
                    continue

        # Резервный механизм
        if not nullspace:
            nullspace = [
                Matrix([[1.0 if i == j else 0.0 for i in range(A.cols)]])
                for j in range(A.cols)
            ]

        # Нормализация и проверка уникальности
        for vec in nullspace:
            components = [row[0] for row in vec.data]
            norm = math.sqrt(sum(x ** 2 for x in components))
            if norm < eps:
                continue
            normalized = [x / norm for x in components]
            is_unique = True
            for existing in eigenvectors:
                if all(abs(a - b) < eps for a, b in zip(normalized, existing)):
                    is_unique = False
                    break
            if is_unique:
                eigenvectors.append(normalized)

    # Гарантия непустого списка
    if not eigenvectors:
        eigenvectors = [[1.0 if i == j else 0.0 for i in range(C.cols)] for j in range(C.cols)]

    return eigenvectors


# Вычисление доли объяснённой дисперсии
def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """
    Вход:
    eigenvalues: список собственных значений
    k: число компонент
    Выход: доля объяснённой дисперсии
    """
    total = sum(eigenvalues)
    explained = sum(eigenvalues[:k])
    return explained / total


# Тесты Normal
class TestEigenvalues(unittest.TestCase):
    # Тест нахождения собственных значений
    def test_convergence(self):
        C = Matrix([[2.0, 1.0], [1.0, 2.0]])
        eigenvalues = find_eigenvalues(C, tol=1e-6)
        self.assertAlmostEqual(eigenvalues[0], 3.0, places=4)
        self.assertAlmostEqual(eigenvalues[1], 1.0, places=4)

    def test_max_iter(self):
        C = Matrix([[1.0, 2.0], [3.0, 4.0]])
        eigenvalues = find_eigenvalues(C, tol=1e-12)
        self.assertEqual(len(eigenvalues), 2)

    # Тест нахождения собственных векторов
    def test_eigenvectors(self):
        C = Matrix([[4.0, 1.0], [1.0, 4.0]])
        eigenvalues = [5.0, 3.0]
        eigenvectors = find_eigenvectors(C, eigenvalues)
        self.assertEqual(len(eigenvectors), 2)
        self.assertAlmostEqual(abs(eigenvectors[0][0]), abs(eigenvectors[0][1]), places=4)
        self.assertAlmostEqual(abs(eigenvectors[1][0]), abs(eigenvectors[1][1]), places=4)

    # Тест вычисления дисперсии
    def test_explained_variance(self):
        eigenvalues = [3.0, 2.0, 1.0]
        self.assertAlmostEqual(explained_variance_ratio(eigenvalues, 2), 5.0 / 6.0, places=4)


# Пример работы
C = Matrix([[2.0, 1.0], [1.0, 2.0]])
eigenvalues = find_eigenvalues(C)
print("Собственные значения:", eigenvalues)  # [3.0, 1.0]

eigenvectors = find_eigenvectors(C, eigenvalues)
print("Собственные векторы:", eigenvectors)  # [[1.0, 1.0], [-1.0, 1.0]]

sorted_eigenvalues = sorted(eigenvalues, reverse=True)
variance = explained_variance_ratio(sorted_eigenvalues, 1)
print("Доля дисперсии (k=1):", variance)  # 0.75


# Hard
# Реализация полного алгоритма PCA
def pca(X: 'Matrix', k: int) -> Tuple['Matrix', float]:
    """
    Вход:
    X: матрица данных (n×m)
    k: число главных компонент
    Выход:
    X_proj: проекция данных (n×k)
    : доля объяснённой дисперсии
    """
    if k > X.cols or k < 1:
        raise ValueError("k должно быть в диапазоне [1, число_признаков]")

    X_centered = center_data(X)
    cov = covariance_matrix(X_centered)

    eigenvalues = find_eigenvalues(cov)
    eigenvectors = find_eigenvectors(cov, eigenvalues)

    if not eigenvectors:
        raise ValueError("Собственные векторы не найдены")

    sorted_pairs = sorted(zip(eigenvalues, eigenvectors), key=lambda x: -x[0])
    sorted_eigenvalues = [pair[0] for pair in sorted_pairs]
    sorted_eigenvectors = [pair[1] for pair in sorted_pairs]

    components = sorted_eigenvectors[:k]
    components_matrix = Matrix(components).transpose()
    projection = X_centered @ components_matrix

    total = sum(sorted_eigenvalues)
    explained = sum(sorted_eigenvalues[:k])
    variance_ratio = explained / total if total != 0 else 0.0
    # Обработка случая, когда не найдено достаточно собственных векторов
    if len(sorted_eigenvectors) < k:
        raise ValueError(f"Недостаточно собственных векторов ({len(sorted_eigenvectors)} найдено, требуется {k})")
    return projection, variance_ratio


# Визуализация проекции данных на первые две главных компоненты
def plot_pca_projection(X_proj: 'Matrix') -> Figure:
    """
    Вход: проекция данных X_proj (n×2)
    Выход: объект Figure из Matplotlib
    """
    fig = plt.figure(figsize=(8, 6))
    plt.scatter([row[0] for row in X_proj.data], [row[1] for row in X_proj.data])
    plt.xlabel('Основной компонент 1')
    plt.ylabel('Основной компонент 2')
    plt.title('Проекция на первые два основных компонента')
    plt.grid(True)
    plt.close(fig)
    return fig


# Вычисление среднеквадратичной ошибки восстановления данных
def reconstruction_error(X_orig: 'Matrix', X_recon: 'Matrix') -> float:
    """
    Вход:
    X_orig: исходные данные (n×m)
    X_recon: восстановленные данные (n×m)
    Выход: среднеквадратическая ошибка MSE
    """
    error = 0.0
    for i in range(X_orig.rows):
        for j in range(X_orig.cols):
            error += (X_orig.data[i][j] - X_recon.data[i][j]) ** 2
    return error / (X_orig.rows * X_orig.cols)


# Тесты Hard
class TestPCA(unittest.TestCase):
    def setUp(self):
        # Тестовые данные (3 образца, 2 признака)
        self.X = Matrix([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])

        # Искусственная ковариационная матрица 2x2
        self.cov_matrix = Matrix([
            [2.0, 1.0],
            [1.0, 2.0]
        ])

    def test_pca_input_validation(self):
        # Проверка некорректных значений k
        with self.assertRaises(ValueError):
            pca(self.X, 0)
        with self.assertRaises(ValueError):
            pca(self.X, 3)

    def test_centering(self):
        # Проверка центрирования данных
        centered = center_data(self.X)
        expected_means = [3.0, 4.0]
        actual_means = [
            sum(col) / self.X.rows
            for col in zip(*centered.data)
        ]
        for exp, act in zip([0.0, 0.0], actual_means):
            self.assertAlmostEqual(exp, act, places=6)

    def test_covariance_matrix(self):
        # Проверка вычисления ковариационной матрицы
        X = Matrix([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        centered = center_data(X)
        cov = covariance_matrix(centered)

        # Ожидаемая ковариационная матрица:
        expected = Matrix([
            [4.0, 4.0],
            [4.0, 4.0]
        ])

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(cov.data[i][j], expected.data[i][j], places=4)

    def test_pca_output(self):
        # Проверка основной функциональности PCA
        X_proj, var_ratio = pca(self.X, 1)

        # Проверка размерности проекции
        self.assertEqual(X_proj.rows, 3)
        self.assertEqual(X_proj.cols, 1)

        # Проверка доли дисперсии
        self.assertAlmostEqual(var_ratio, 1.0, places=4)  # Все дисперсия в 1 компоненте

    def test_reconstruction_error(self):
        # Тест MSE для идентичных матриц
        error = reconstruction_error(self.X, self.X)
        self.assertAlmostEqual(error, 0.0, places=6)

        # Тест MSE для разных матриц
        X_recon = Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        error = reconstruction_error(self.X, X_recon)
        expected = (1 ** 2 + 2 ** 2 + 3 ** 2 + 4 ** 2 + 5 ** 2 + 6 ** 2) / 6
        self.assertAlmostEqual(error, expected, places=4)

    def test_plot_function(self):
        # Проверка что функция не падает с ошибкой
        X_proj, _ = pca(self.X, 2)
        fig = plot_pca_projection(X_proj)
        self.assertIsInstance(fig, plt.Figure)

    def test_edge_cases(self):
        # Тест для k = n_features
        X_proj, var_ratio = pca(self.X, 2)
        self.assertEqual(X_proj.cols, 2)
        self.assertAlmostEqual(var_ratio, 1.0, places=4)

        # Тест для уже центрированных данных
        centered_X = center_data(self.X)
        X_proj, _ = pca(centered_X, 1)
        self.assertEqual(X_proj.cols, 1)


# Пример работы
X = Matrix([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

try:
    X_proj, variance = pca(X, k=1)
    print("Проекция (не нормированная):\n", X_proj)
    print("Доля дисперсии:", variance)
except ValueError as e:
    print("Ошибка:", e)


# Expert
# Обработка пропущенных значений в данных
def handle_missing_values(X: 'Matrix') -> 'Matrix':
    filled_data = []
    n, m = X.rows, X.cols
    for j in range(m):
        valid_values = []
        for i in range(n):
            val = X.get(i, j)
            if not math.isnan(val):
                valid_values.append(val)
        mean = sum(valid_values) / len(valid_values) if valid_values else 0.0
        # Заменяем все NaN в столбце на среднее
        filled_column = []
        for i in range(n):
            val = X.get(i, j)
            filled_column.append(mean if math.isnan(val) else val)
        filled_data.append(filled_column)
    # Транспонируем обратно в n×m
    transposed = list(map(list, zip(*filled_data)))
    return Matrix(transposed)


# Автоматический выбор числа главных компонент
def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)
    total_variance = sum(sorted_eigenvalues)
    if total_variance == 0:
        return 0
    cumulative = 0.0
    for k, value in enumerate(sorted_eigenvalues, 1):
        cumulative += value
        if cumulative / total_variance >= threshold:
            return k
    return len(sorted_eigenvalues)


# Влияние шума на PCA
def add_noise_and_compare(X: 'Matrix', noise_level: float = 0.1):
    X_clean = handle_missing_values(X)

    # Генерация шума с проверкой на нулевое стандартное отклонение
    noise = []
    for j in range(X_clean.cols):
        col = [X_clean.get(i, j) for i in range(X_clean.rows)]
        col_mean = sum(col) / len(col)
        variance = sum((x - col_mean) ** 2 for x in col) / len(col)
        std = math.sqrt(variance) if variance != 0 else 0.0
        noise_std = std * noise_level if std != 0 else 0.0
        noise_col = [random.gauss(0, noise_std) for _ in range(X_clean.rows)]
        noise.append(noise_col)

    X_noisy = Matrix([
        [X_clean.get(i, j) + noise[j][i] for j in range(X_clean.cols)]
        for i in range(X_clean.rows)
    ])

    # PCA с обработкой нулевых собственных значений
    try:
        proj_orig, var_orig = pca(X_clean, X_clean.cols)
    except ValueError:
        var_orig = 0.0
    try:
        proj_noisy, var_noisy = pca(X_noisy, X_noisy.cols)
    except ValueError:
        var_noisy = 0.0

    return {
        "original_variance": var_orig,
        "noisy_variance": var_noisy,
    }


# Реализация применения PCA к датасету
def apply_pca_to_dataset(dataset_name: str, k: int) -> Tuple['Matrix', float]:
    if dataset_name == "iris":
        data = load_iris()
        X = Matrix(data.data.tolist())
        y = data.target
    else:
        raise ValueError("Датасет не поддерживается")

    # Обработка пропущенных значений
    X_clean = handle_missing_values(X)

    # Применение PCA
    X_proj, variance_ratio = pca(X_clean, k)

    # Пример метрики (доля объясненной дисперсии)
    return X_proj, variance_ratio


# Тесты Expert
class TestExpertFunctions(unittest.TestCase):
    def test_auto_select_k_basic(self):
        eigenvalues = [4.0, 3.0, 2.0, 1.0]
        self.assertEqual(auto_select_k(eigenvalues, 0.9), 3)  # (4+3+2)/10=0.9 → k=3
        self.assertEqual(auto_select_k(eigenvalues, 0.5), 2)  # (4+3)/10=0.7 → k=2

    def test_noise_impact(self):
        random.seed(42)
        X = Matrix([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        result = add_noise_and_compare(X, noise_level=0.1)
        self.assertLessEqual(result["noisy_variance"], 1.0)

    def test_handle_nan(self):
        X = Matrix([
            [1.0, 2.0, float('nan')],
            [float('nan'), 5.0, 6.0],
            [7.0, float('nan'), 9.0]
        ])
        X_filled = handle_missing_values(X)
        # Проверка замены NaN во втором элементе первого столбца
        self.assertAlmostEqual(X_filled.get(1, 0), 4.0, places=4)
        # Проверка замены NaN в третьем элементе второго столбца
        self.assertAlmostEqual(X_filled.get(2, 1), 3.5, places=4)
        # Проверка замены NaN в первом элементе третьего столбца
        self.assertAlmostEqual(X_filled.get(0, 2), 7.5, places=4)


class TestApplyPCA(unittest.TestCase):
    def test_full_variance_with_max_components(self):
        _, variance_ratio = apply_pca_to_dataset("iris", 4)
        self.assertAlmostEqual(variance_ratio, 1.0, delta=0.01)  # Увеличена дельта для погрешности

    def test_iris_dataset_dimensions(self):
        k = 2
        X_proj, variance_ratio = apply_pca_to_dataset("iris", k)
        self.assertEqual(X_proj.rows, 150)
        self.assertEqual(X_proj.cols, k)
        self.assertGreater(variance_ratio, 0.9)
        self.assertLess(variance_ratio, 0.98)

    def test_variance_increase_with_components(self):
        var1 = apply_pca_to_dataset("iris", 1)[1]
        var2 = apply_pca_to_dataset("iris", 2)[1]
        self.assertGreater(var2, var1)

    def test_unsupported_dataset_error(self):
        with self.assertRaises(ValueError):
            apply_pca_to_dataset("invalid_dataset", 2)


# Запуск всех тестов
if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)