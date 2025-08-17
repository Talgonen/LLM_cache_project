from gptcache.manager.eviction.policy_extentions.max_area import MaxAreaEvictionPolicy
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager.scalar_data.base import CacheData
import numpy as np
import unittest

class TestMaxAreaEviction(unittest.TestCase):

    def setUp(self):
        # Setup embedding, data_func and distance_func
        ## 2 similiar sentences
        self.sent_1 = "The cat sat on the mat."
        self.sent_2 = "The cat is sitting on the mat."

        ## 2 different sentences
        self.sent_3 = "The dog barked at the cat."
        self.sent_4 = "The sun is shining brightly."
        self.questions = [self.sent_1, self.sent_2, self.sent_3, self.sent_4]
        mock_embeddings = [np.array([1,2]), np.array([1.1,2]), np.array([-3,10]), np.array([7,9])] # Mock embeddings for the sentences

        def data_func(id):
            return CacheData(**{
                "question": self.questions[id],
                "answers": [{"answer": f"Answer for {self.questions[id]}"}],
                "embedding_data": mock_embeddings[id]
            })

        self.data_func = data_func
        self.distance_func = SearchDistanceEvaluation(max_distance=1)
        self.embedding_dimension = 2
        self.radius = 1.0
        self.max_size = 3

        # Initialize MaxAreaEvictionPolicy
        self.max_area_policy = MaxAreaEvictionPolicy(
            maxsize=self.max_size,
            data_func=self.data_func,
            distance_func=self.distance_func,
            embedding_dimension=self.embedding_dimension,
            radius=self.radius,
            threshold=0.99
        )

    def test_max_size(self):
        # Initialize the MaxAreaEvictionPolicy with a max size
        for i in range(4):
            self.max_area_policy[i] = True
        
        # Check if the size of the cache is equal to maxsize
        self.assertEqual(len(self.max_area_policy._cache), self.max_size)

    def test_eviction_policy(self):
        # Initialize the cache with the sentences
        for i in range(4):
            self.max_area_policy[i] = True

        # Sentences 1 (index 0) and 2 (index 1) are similar, so one of them should be evicted
        self.assertIn(2, self.max_area_policy._cache)
        self.assertIn(3, self.max_area_policy._cache)
        # Sen 3 and 4 on the cache, and the cache size is 3 - meaning one of the similar sentences (1 or 2) was evicted
        self.assertEqual(len(self.max_area_policy._cache), self.max_size) 


if __name__ == "__main__":
    unittest.main()