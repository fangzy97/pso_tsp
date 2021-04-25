import numpy as np
from utils import read_data


class PSO:
    def __init__(self, path, max_iter, population_num, w, alpha, beta):
        self.path = path
        self.max_iter = max_iter
        self.population_num = population_num
        self.w = w
        self.alpha = alpha
        self.beta = beta

        # 读取城市数据
        self.cities = read_data(self.path)
        self.city_num = len(self.cities)
        # 计算各个城市之间的距离
        self.dist = self.cal_dist()
        # 初始化粒子群
        self.population, self.list_v = self.init_population()

    def solve(self):
        pbest = np.empty(self.population_num)
        for i in range(self.population_num):
            pbest[i] = self.evaluate(self.population[i])
        gbest = np.min(pbest)
        gv = self.population[np.argmin(pbest)]

        for i in range(self.max_iter):
            for j in range(self.population_num):
                x, y = np.random.choice(self.city_num, 2, replace=False)
                temp = self.population[j]
                temp[x], temp[y] = temp[y], temp[x]
                dist = self.evaluate(temp)

                if dist < pbest[j]:
                    pbest[j] = dist
                    self.population[j] = temp


    def cal_dist(self):
        mat1 = np.expand_dims(self.cities, 0)
        mat2 = np.expand_dims(self.cities, 1)
        dist = np.linalg.norm(mat1 - mat2, ord=2, axis=-1)
        return dist

    def init_population(self):
        origin = np.arange(self.city_num)
        populations = np.zeros((self.population_num, self.city_num))
        list_v = np.zeros((self.population_num, self.city_num))
        for i in range(self.population_num):
            populations[i] = np.random.choice(self.city_num)
            for j in range(self.city_num):
                if origin[j] != populations[i, j]:
                    list_v[i, j] = j
        return populations, list_v

    def evaluate(self, sequence):
        dist = 0.
        for i in range(1, self.city_num):
            dist += self.dist[sequence[i - 1]][sequence[i]]
        dist += self.dist[sequence[0]][sequence[self.city_num - 1]]
        return dist


if __name__ == '__main__':
    pso = PSO('dataset.txt', 100, 50, 0.9, 0.5, 0.5)
    pso.solve()
