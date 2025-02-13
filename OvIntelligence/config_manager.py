# config_manager.py

import json
from pathlib import Path
import os

class ConfigManager:
    CONFIG_PATH = Path('config.json')

    @staticmethod
    def load_config():
        if ConfigManager.CONFIG_PATH.exists():
            with open(ConfigManager.CONFIG_PATH, 'r') as f:
                return json.load(f)
        else:
            return {
                'file_selection_model': 'qwen2.5-coder:3b',
                'query_processing_model': 'qwen2.5-coder:7b',
                'rate_limit': 5,
                'gemini_api_key': "Put-Your-Key"
            }

    @staticmethod
    def save_config(config):
        with open(ConfigManager.CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
