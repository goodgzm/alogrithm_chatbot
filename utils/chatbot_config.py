import yaml
import os

class ChatBotConfig:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatBotConfig, cls).__new__(cls)
            cls._instance._config = None
        return cls._instance
    
    def init(self, dir_path):
        with open(os.path.join(dir_path, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)

        self._instance._config = config

    def __getattr__(self, name):
        if self._instance._config and name in self._instance._config:
            return self._instance._config[name]
        raise AttributeError(f"ChatBotConfig 对象没有属性'{name}'")