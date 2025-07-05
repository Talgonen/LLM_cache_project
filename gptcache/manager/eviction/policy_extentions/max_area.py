from typing import Any, List, Callable, Optional, Union
import numpy as np
from collections import defaultdict
import logging
import math
from scipy.special import gammaln, betainc

class MaxAreaEvictionPolicy():

    def __init__(
        self,
        maxsize: int,
        data_func: Callable[[Any], np.ndarray],
        distance_func: Callable[[Any, Any], Any],
        **kwargs
    ):
        self._distance_func = distance_func
        self.maxsize = maxsize
        self.embedding_dim = kwargs["embedding_dimension"]
        self.radius = kwargs["radius"]
        similarity_threshold = kwargs["threshold"]
        min_rank, max_rank = self._distance_func.range()
        rank_threshold = (max_rank - min_rank) * similarity_threshold * 1.0
        rank_threshold = (
            max_rank
            if rank_threshold > max_rank
            else min_rank
            if rank_threshold < min_rank
            else rank_threshold
        )
        self.threshold = self._distance_adjust(rank_threshold)
        
        # _cache storage: key -> (item, embedding, coverage_score)
        self._cache = {}
        self._relation_graph = {}
        self.eval_query_data = lambda id: self._eval_query_data(data_func, id)
        self.eval_cache_data = lambda id: self._eval_cache_data(data_func, id)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def _eval_query_data(self, f, id):
        information = f(id)
        return {
            "question": information.question,
            "embedding": information.embedding_data
        }
    
    def _eval_cache_data(self, f, id):
        information = f(id)
        return {
            "question": information.question,
            "answer": information.answers[0].answer,
            "cache_data": information,
            "embedding": information.embedding_data
        }
    
    def _distance_adjust(self, d):
        return self._distance_func.range()[1] - d
    
    def __setitem__(self, key, value):
        if key in self._cache:
            return
        
        current_data = self.eval_query_data(key)
        _key_neighbors = {}

        # Update all graph with new key and calculate intersection areas
        for _cached_key in self._relation_graph.keys():
            cached_data = self.eval_cache_data(_cached_key)
            cached_data["search_result"] = (float(np.linalg.norm(current_data["embedding"] - cached_data["embedding"], ord=2)), _cached_key)
            d = self._distance_adjust(self._distance_func.evaluation(current_data, cached_data))
            if self.threshold <= d:
                intersection_area = self._n_ball_intersection_volume(self.embedding_dim, self.radius, d)
                _key_neighbors[_cached_key] = intersection_area
                self._relation_graph[_cached_key][key] = intersection_area

        self._relation_graph[key] = _key_neighbors
        self._cache[key] = True

        if len(self._cache) > self.maxsize:
            t_inter_area = list(map(lambda key, values: (key, sum(values.values())), self._relation_graph.keys(), self._relation_graph.values()))
            max_key, max_area = max(t_inter_area, key=lambda x: x[1])
            self.popitem(max_key)

    def popitem(self, key: Any):
        for _cached_key in list(self._relation_graph[key].keys()):
            self._relation_graph[_cached_key].pop(key)

        self._relation_graph.pop(key)
        self._cache.pop(key)
        return key,

    def get(self, keys: Any):
        return self._cache.get(keys)

    @property
    def policy(self) -> str:
        return "MAX_AREA"

    @classmethod
    def _n_ball_intersection_volume(cls, n: int, R: float, d: float) -> float:
        # --- 1. Input Validation ---
        if not isinstance(n, int) or n < 0:
            raise ValueError("Dimension 'n' must be a non-negative integer.")
        if R < 0:
            raise ValueError("Radius 'R' must be non-negative.")
        if d < 0:
            raise ValueError("Distance 'd' must be non-negative.")

        # --- 2. Handle Edge Cases ---
        if d >= 2 * R:
            # If the balls are separated or just touching, the intersection volume is 0.
            return 0.0

        # --- 3. Main Calculation ---

        # Volume of a full n-dimensional ball of radius R
        # V_n(R) = (pi^(n/2) / Gamma(n/2 + 1)) * R^n
        volume_n_ball = cls._n_ball_volume(n, R)

        if d == 0:
            # If the balls are concentric, the intersection is the volume of one ball.
            return volume_n_ball

        # Calculate the regularized incomplete beta function I_x(a, b)
        incomplete_beta_value = betainc(n / 2.0, 3/2, 1 - (d**2) / (4 * R**2))
        
        # The volume of the intersection is V_n(R) * I_x(a, b)
        return volume_n_ball * incomplete_beta_value
    
    @classmethod
    def _n_ball_volume(cls, n, R):
        if not isinstance(n, int) or n < 0:
            raise ValueError("Dimension 'n' must be a non-negative integer.")
        if R < 0:
            raise ValueError("Radius 'R' must be non-negative.")
        if R == 0:
            return 0.0

        # Calculate the natural logarithm of the volume
        log_volume = (n / 2) * np.log(np.pi) + n * np.log(R) - gammaln(n / 2 + 1)
        
        return np.exp(log_volume)