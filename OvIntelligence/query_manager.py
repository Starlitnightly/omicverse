# query_manager.py
import os

class QueryManager:
    @staticmethod
    def validate_query(query):
        if not query or len(query.strip()) < 3:
            return False, "Query must be at least 3 characters long"
        if len(query) > 1000:
            return False, "Query must be less than 1000 characters"
        return True, ""
