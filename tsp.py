import numpy as np
from tqdm import tqdm

from utils import read_data


class PSO:
    def __init__(self, path, max_iter, population_num, alpha, beta):
        self.path = path
        self.max_iter = max_iter
        self.population_num = population_num
        self.alpha = alpha
        self.beta = beta

        # 读取城市数据
        self.cities = read_data(self.path)
        self.city_num = len(self.cities)
        # 计算各个城市之间的距离
        self.dist_mat = self.cal_dist()
        # 初始化粒子群
        self.population= self.init_population()
        self.dists = np.zeros(self.population_num)
        for i in range(self.population_num):
            self.dists[i] = self.evaluate(self.population[i])
        # 记录当前个体的最优解
        self.local_best = self.population
        self.local_dist = self.dists
        # 记录全局最优解
        global_index = np.argmin(self.dists)
        self.global_best = self.population[global_index]
        self.global_dist = self.dists[global_index]

    def solve(self):
        for i in tqdm(range(self.max_iter)):
            for idx, population in enumerate(self.population):
                dist = self.dists[idx]
                new_population = self.swap(population, self.local_best[idx])
                new_dist = self.evaluate(new_population)
                if new_dist < dist or np.random.rand() < self.alpha:
                    population = new_population
                    dist = new_dist

                new_population = self.swap(population, self.global_best)
                new_dist = self.evaluate(new_population)
                if new_dist < dist or np.random.rand() < self.beta:
                    population = new_population
                    dist = new_dist

                self.population[idx] = population
                self.dists[idx] = dist

            min_idx = np.argmin(self.dists)
            min_dist = self.dists[min_idx]
            min_path = self.population[min_idx]
            if min_dist < self.global_dist:
                self.global_dist = min_dist
                self.global_best = min_path
            for idx, dist in enumerate(self.dists):
                if dist < self.local_dist[idx]:
                    self.local_dist[idx] = dist
                    self.local_best[idx] = self.population[idx]

        return self.global_best, self.global_dist

    def cal_dist(self):
        mat1 = np.expand_dims(self.cities, 0)
        mat2 = np.expand_dims(self.cities, 1)
        dist = np.linalg.norm(mat1 - mat2, ord=2, axis=-1)
        return dist

    def init_population(self):
        populations = np.zeros((self.population_num, self.city_num), dtype=np.int)
        for i in range(self.population_num):
            populations[i] = np.random.choice(self.city_num, self.city_num, replace=False)

        return populations

    def swap(self, cur, best):
        x, y = np.random.choice(self.city_num, 2)
        if x > y:
            x, y = y, x
        swap_part = best[x : y]
        no_swap_part = []
        for city in cur:
            if city not in swap_part:
                no_swap_part.append(city)
        no_swap_part = np.array(no_swap_part)
        out = np.concatenate((swap_part, no_swap_part))
        return out

    def evaluate(self, sequence):
        dist = self.dist_mat[sequence[0]][sequence[self.city_num - 1]]
        for i in range(1, self.city_num):
            dist += self.dist_mat[sequence[i - 1]][sequence[i]]
        return dist


if __name__ == '__main__':
    np.random.seed(1234)
    pso = PSO('dataset.txt', 100, 1000, 0.1, 0.1)
    best_path, best_dist = pso.solve()
    print(best_dist)
