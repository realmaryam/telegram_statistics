from typing import Union
from pathlib import Path
import json
from src.data import DATA_DIR
from hazm import word_tokenize, Normalizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from loguru import logger


class Chatstatistics:
    """Generates chat statistics from telegram chat json file
    """
    def __init__(self, chat_json: Union[str, Path]):
        """
        :param chat_json: path to telegram export json file
        """   
        # load chat data
        logger.info(f"Loading chat data from {chat_json}")
        with open(chat_json) as f:
            self.chat_data = json.load(f)
        
        self.normalizer = Normalizer()

        # load stopwords
        logger.info(f"Loading stop words from {DATA_DIR / 'stopwords.txt'}")
        stop_words = open(DATA_DIR / 'stopwords.txt').readlines()
        stop_words = list(map(str.strip, stop_words))
        self.stop_words = list(map(self.normalizer.normalize, stop_words))

    def generate_word_cloud(self, output_dir):
        """Generates a word cloud from the chat data

        :param outpu_dir: path to output directory for word cloud image
        """
        Sib = self.chat_data['chats']['list'][5]

        logger.info("Loading text content")
        text_content = ''
        for msg in Sib['messages']:
            if type(msg['text']) is str:
                tokens = word_tokenize(msg['text'])
                tokens = list(filter(lambda item: item not in self.stop_words, tokens))
                text_content += f"{' '.join(tokens)}"

        # generate wordcloud
        logger.info("Generating word cloud...")
        wordcloud = WordCloud(
            width=1200, height=1200,
            font_path=str(DATA_DIR / './BHoma.ttf'),
            background_color='white'
            ).generate(text_content)

        logger.info(f"Saving word cloud to {output_dir}")
        wordcloud.to_file(str(Path(output_dir) / 'wordcload.png'))


if __name__ == "__main__":
    chat_stats = Chatstatistics(chat_json=DATA_DIR / "chats.json")
    chat_stats.generate_word_cloud(output_dir=DATA_DIR)
