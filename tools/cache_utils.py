import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cache_tools.cache as cache

semi_cache = cache.InMemorySemanticCache()

def get_cache():
    return semi_cache