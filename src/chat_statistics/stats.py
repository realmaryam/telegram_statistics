import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from hazm import Normalizer, sent_tokenize, word_tokenize
from loguru import logger
from wordcloud import WordCloud

from src.data import DATA_DIR


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
            self.chat_data = json.load(f)['chats']['list'][1]
        
        self.normalizer = Normalizer()

        # load stopwords
        logger.info(f"Loading stop words from {DATA_DIR / 'stopwords.txt'}")
        stop_words = open(DATA_DIR / 'stopwords.txt').readlines()
        stop_words = map(str.strip, stop_words)
        self.stop_words = set(list(map(self.normalizer.normalize, stop_words)))

    @staticmethod
    def rebuild_msg(sub_messages):
        msg_text = ''
        for sub_msg in sub_messages:
            if isinstance(sub_msg, str):
                msg_text += sub_msg
            elif 'text' in sub_msg:
                msg_text += sub_msg['text']
        return msg_text

    def get_top_users(self, top_n: int = 10) -> dict:
        """
        Gets top n users from the chat, default to 10
        """
        # check messages for questions
        is_question = defaultdict(bool)
        for msg in self.chat_data['messages']:
            if not isinstance(msg['text'], str):
                msg['text'] = self.rebuild_msg(msg['text'])
                
            sentences = sent_tokenize(msg['text'])
            for sentence in sentences:
                if ('?' not in sentence) and ('؟' not in sentence):
                    continue
                is_question[msg['id']] = True
                break

        # get top users based on replying to questions from others
        logger.info("Getting top users...")
        users = []
        for msg in self.chat_data['messages']:
            if not msg.get('reply_to_message_id'):
                continue
            if is_question[msg['reply_to_message_id']] is False:
                continue
            users.append(msg['from'])

        return dict(Counter(users).most_common(top_n))

    def generate_word_cloud(self, output_dir):
        """Generates a word cloud from the chat data

        :param outpu_dir: path to output directory for word cloud image
        """
        # Sib = self.chat_data['chats']['list'][6]

        logger.info("Loading text content")
        text_content = ''
        for msg in self.chat_data['messages']:
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
    chat_stats = Chatstatistics(chat_json=DATA_DIR / "group_chats.json")
    chat_stats.generate_word_cloud(output_dir=DATA_DIR)
    top_users = chat_stats.get_top_users()
    print(top_users)
