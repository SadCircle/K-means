import numpy as np
import copy
import matplotlib.pyplot as  plt
import matplotlib.image as img
from PIL import Image 
from collections import Counter


class k_means:
    def __init__(self,n_clstrs,seed=None):
        if seed:
            np.random.seed(seed)
        self.colored_points=[]
        self.n_clstrs=n_clstrs


    def find_nearest_cluster(self,point):
        return np.argmin([np.linalg.norm(point-cluster) for cluster in self.clusters])

    def color_points(self):
        self.colored_points=np.array([self.find_nearest_cluster(point) for point in self.points])
        
    def calc_new_clusters(self):
        old=copy.deepcopy(self.clusters)
        for n_cluster in range(len(self.clusters)):
            points=np.where(self.colored_points == n_cluster)
            if points:
                self.clusters[n_cluster]=np.mean(self.points[points],axis=0)
        if np.array_equal(old,self.clusters):
            return False
        return True

    def calc_claster_distanse(self):
        distanse=[]
        objects_per_cluster=Counter(self.colored_points)
        for cluster in self.clusters:
            distanse.append([np.linalg.norm(cluster_2-cluster) for  cluster_2 in self.clusters])
        print(f'============Расстояние между центроидами=====\n{distanse}')
        print(f'===========Число объектов в кластере================\n{objects_per_cluster} ')
        
    def vizualize(self):
        plt.title('Представление точек')
        colors=['r','green','y','black']
        for n_cluster in range(len(self.clusters)):
            points=self.points[np.where(self.colored_points == n_cluster)]
            x_1 = []
            y_1 = []
            for el in points:
                x_1.append(el[0])
                y_1.append(el[1])
            plt.scatter(x_1,y_1,marker='o',c=colors[n_cluster%len(colors)],edgecolor='b')

        x_1 = []
        y_1 = []
        for n_cluster,el in enumerate(self.clusters):
            plt.scatter(el[0],el[1],marker='o',c=colors[n_cluster%len(colors)],edgecolor='r')
        plt.grid()
        plt.show()
    

    def fit(self,X):
        self.points=X
        #Определяем кластеры случайным образом
        self.clusters=self.points[np.random.choice(self.points.shape[0],self.n_clstrs,replace=False)]
        #Окрашиваем входные точки по близости к цетроидам
        self.color_points()
        # Пересчитываем центтроиды
        while self.calc_new_clusters():
            self.color_points()
            self.calc_claster_distanse()