import numpy as np
import networkx as nx

class GraphCentrality:
    def __init__(self, adjacency_matrix):
        """
        Инициализация объекта GraphCentrality с заданной матрицей смежности.
        
        Параметры:
        adjacency_matrix (numpy.ndarray): Квадратная матрица смежности, описывающая граф.
        """
        self.adjacency_matrix = adjacency_matrix
        self.graph = nx.from_numpy_array(adjacency_matrix)

    def calculate_betweenness_centrality(self):
        """
        Вычисление посреднической центральности (betweenness centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их посредническая центральность.
        """
        return nx.betweenness_centrality(self.graph)

    def calculate_eigenvector_centrality(self):
        """
        Вычисление векторной центральности (eigenvector centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их векторная центральность.
        """
        return nx.eigenvector_centrality(self.graph)

    def calculate_pagerank_centrality(self, alpha=0.85):
        """
        Вычисление центральности на основе PageRank.
        
        Параметры:
        alpha (float): Коэффициент затухания (damping factor) для PageRank (по умолчанию 0.85).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их PageRank-центральность.
        """
        return nx.pagerank(self.graph, alpha=alpha)

    def calculate_flow_coefficient(self):
        """
        Вычисление «коэффициента потока» (flow coefficient) для всех узлов графа.
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их «коэффициент потока».
        """
        flow_coefficient = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) < 2:
                flow_coefficient[node] = 0.0
            else:
                edge_count = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if self.graph.has_edge(neighbors[i], neighbors[j]):
                            edge_count += 1
                flow_coefficient[node] = (2 * edge_count) / (len(neighbors) * (len(neighbors) - 1))
        return flow_coefficient

    def calculate_degree_centrality(self):
        """
        Вычисление степенной центральности (degree centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их степенная центральность.
        """
        return nx.degree_centrality(self.graph)

    def calculate_closeness_centrality(self):
        """
        Вычисление близостной центральности (closeness centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их близостная центральность.
        """
        return nx.closeness_centrality(self.graph)

    def calculate_harmonic_centrality(self):
        """
        Вычисление гармонической центральности (harmonic centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их гармоническая центральность.
        """
        return nx.harmonic_centrality(self.graph)

    def calculate_percolation_centrality(self, attribute=None):
        """
        Вычисление перколяционной центральности (percolation centrality).
        
        Параметры:
        attribute (str или dict): Атрибут узла для моделирования «перколяции» (по умолчанию None).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их перколяционная центральность.
        """
        if attribute is None:
            attribute = {node: 1 for node in self.graph.nodes()}
        return nx.percolation_centrality(self.graph, states=attribute)

    def calculate_cross_clique_centrality(self):
        """
        Вычисление центральности «cross-clique» (количество кликовых покрытий узла).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их центральность по числу клик.
        """
        cross_clique_centrality = {}
        cliques = list(nx.find_cliques(self.graph))
        for node in self.graph.nodes():
            cross_clique_centrality[node] = sum(node in clique for clique in cliques)
        return cross_clique_centrality

    def find_hub_vertices(self, threshold=0.9):
        """
        Определение «hub»-вершин на основе степенной центральности.
        
        Параметры:
        threshold (float): Порог для определения «hub»-вершин (по умолчанию 0.9).
        
        Возвращает:
        list: Список индексов узлов, являющихся «hub»-вершинами.
        """
        degree_centrality = self.calculate_degree_centrality()
        max_centrality = max(degree_centrality.values())
        hub_vertices = [node for node, val in degree_centrality.items()
                        if val >= threshold * max_centrality]
        return hub_vertices