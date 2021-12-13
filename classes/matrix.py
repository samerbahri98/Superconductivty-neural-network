class Matrix:
    def __init__(self, data):
        self.data = data

    def multiply(self, b):
        result = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(b.data[0])):
                current = 0
                for k in range(len(b.data)):
                    current += self.data[i][k] * b.data[k][j]

                row.append(current)
            result.append(row)
        return Matrix(result)

    # Leaky Relu
    def activate(self, beta=0.1):
        current_data = self.data
        for i in range(len(current_data)):
            for j in range(len(current_data[0])):
                if current_data[i][j] <= 0:
                    current_data[i][j] *= beta
        return Matrix(current_data)

    def derivative(self, beta=0.1):
        current_data = self.data
        for i in range(len(current_data)):
            for j in range(len(current_data[0])):
                if current_data[i][j] < 0:
                    current_data[i][j] = -beta
                else:
                    current_data[i][j] = 1
        return Matrix(current_data)

    def add(self, b):
        result = self.data
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                result[i][j] += b.data[i][j]
        return Matrix(result)

    def scale(self, b):
        result = self.data
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                result[i][j] *= b
        return Matrix(result)

    def hadamard(self, b):
        result = self.data
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                result[i][j] *= b.data[i][j]
        return Matrix(result)

    def transpose(self):
        result = [[] for i in range(len(self.data[0]))]
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                result[j].append(self.data[i][j])
        return Matrix(result)