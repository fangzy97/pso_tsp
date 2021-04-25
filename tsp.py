import numpy as np
from utils import read_data


class PSO:
    def __init__(self, path, max_iter, population_num, w):
        self.path = path
        self.max_iter = max_iter
        self.population_num = population_num
        self.w = w

        # 读取城市数据
        self.cities = read_data(self.path)
        self.city_num = len(self.cities)
        # 计算各个城市之间的距离
        self.dist = self.cal_dist()
        # 初始化粒子群
        self.population, self.list_v = self.init_population()

    def solve(self):
        pass

    def cal_dist(self):
        mat1 = np.expand_dims(self.cities, 0)
        mat2 = np.expand_dims(self.cities, 1)
        dist = np.linalg.norm(mat1 - mat2, ord=2, axis=-1)
        return dist

    def init_population(self):
        populations = np.zeros((self.population_num, self.city_num))
        list_v = np.zeros((self.population_num, 2))
        for i in range(self.population_num):
            populations[i] = np.random.choice(self.city_num)
            list_v[i] = np.random.choice(self.city_num, 2, replace=False)
        return populations, list_v

    def evaluate(self, sequence):
        dist = 0.
        for i in range(1, self.city_num):
            dist += self.dist[sequence[i - 1]][sequence[i]]
        dist += self.dist[sequence[0]][sequence[self.city_num - 1]]
        return dist


if __name__ == '__main__':
    pso = PSO('dataset.txt', 100, 50, 0.9)
    pso.solve()
