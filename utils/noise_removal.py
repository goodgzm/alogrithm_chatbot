import re
import jieba
from utils.data_path import get_data_file_path
class NoiseRemoval:
    def __init__(self):
        with open(get_data_file_path("stopwords.txt"), 'r', encoding="UTF-8") as f:
            stop_words = set(word.strip() for word in f if word.strip())
        self.stop_words = stop_words

    def Do_NoiseRemoval(self, text : str):
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = jieba.lcut(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ''.join(filtered_words)

