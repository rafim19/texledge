import nltk
import emoji
import re
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import html
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

file = 'cahya/bert-base-indonesian-522M'
bert_tokenizer = BertTokenizer.from_pretrained(file, sep_token='[SEP]')

url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

MAX_SENTLEN = 50
MAX_SENTNUM = 100

def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    return tokens

def get_model_friendly_scores(scores_array):
    low, high = (-4, 4)
    scores_array = (scores_array - low) / (high - low)
    return scores_array

def replace_url(text):
    """
    Replace URLs in text with <url> token
    Handles:
    - HTTP/HTTPS/FTP protocols
    - www prefixes
    - Bare domains (example.com)
    - Various TLDs
    - Paths and parameters
    """
    try:
        url_pattern = (
            r'(?:'
            # URL with protocol
            r'(?:https?://|ftp://)(?:[\w\-]+\.)*[\w\-]+\.[a-zA-Z]{2,63}(?:/[^\s]*)?|'
            # URL with www prefix
            r'www\.(?:[\w\-]+\.)*[\w\-]+\.[a-zA-Z]{2,63}(?:/[^\s]*)?|'
            # URL bare domains
            r'(?:[\w\-]+\.)*[\w\-]+\.(?:com|org|net|edu|gov|mil|biz|info|mobi|name|aero|asia|jobs|museum|id|co\.id|ac\.id|go\.id|web\.id)(?:/[^\s]*)?'
            r')'
        )
        
        replaced_text = re.sub(url_pattern, url_replacer, text, flags=re.IGNORECASE)
        return replaced_text
    except Exception as e:
        print(f"Error in URL replacement: {e}")
        return text

def preprocess_html_text(text):
    # Clean HTML Tags
    text = html.unescape(text)
    
    soup = BeautifulSoup(text, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()

    text = re.sub(r'\n', ' ', text)
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    return text

def normalize_text(text, replace_url_flag):
    """
    Enhanced text normalization with HTML handling.
    This function normalizes the input text by performing several operations:
    - Optionally replaces URLs if `replace_url_flag` is set to True.
    - Removes double quotes.
    - Processes HTML content.
    - Converts numbers to a text representation.
    - Converts emojis to their text representation.
    - Replaces common slang words with their formal equivalents.
    - Removes repeated characters (e.g., 'mantappppp' -> 'mantap').
    - Handles special characters and symbols.
    - Normalizes whitespace.
    Args:
        text (str): The input text to be normalized.
        replace_url_flag (bool): A flag indicating whether to replace URLs in the text.
    Returns:
        str: The normalized text.
    """

    if replace_url_flag:
        text = replace_url(text)
    text = text.replace(u'"', u'')

    text = preprocess_html_text(text)
    
    text = re.sub(r'\d+[\.,]?\d*', ' <NUM> ', text)
    
    text = emoji.demojize(text)
    
    # Remove repeated characters (e.g., 'mantappppp' -> 'mantap')
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    """
    Handle special characters and symbols
    1. \u00A0: Non-breaking space
    2. \u2000-\u200F: Various space characters, including en quad, em quad, en space, em space, three-per-em space, four-per-em space, six-per-em space, figure space, punctuation space, thin space, hair space, zero-width space, and left-to-right mark
    3. \u2028-\u202F: Line separator, paragraph separator, narrow no-break space
    4. \u205F: Medium mathematical space
    5. \u3000: Ideographic space
    """
    text = re.sub(r'[\u00A0\u2000-\u200F\u2028-\u202F\u205F\u3000]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = normalize_text(text, replace_url_flag)

    sent_tokens = text
    return sent_tokens

# def read_dataset(file_path, prompt_id, score_index=7, char_level=False):
def read_dataset(data, purpose, char_level=False):
    print('Reading dataset for: ' + purpose)

    data_x_id, data_y = [], []
    
    for index, item in data.iterrows():
        content = item['isiPengusul'].strip()
        score = float(item['nilai'])
        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        if char_level:
            raise NotImplementedError
        
        tokenized_text = bert_tokenizer.tokenize(sent_tokens)
        max_num = 512
        indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

        data_x_id.append(indexed_tokens)
        data_y.append(score)
    
    return data_x_id, data_y, max_num

# def get_data(paths, prompt_id):
def get_data(train_df, dev_df, test_df):
    train_x, train_y, train_maxnum = read_dataset(train_df, 'train')
    dev_x, dev_y, dev_maxnum = read_dataset(dev_df, 'dev')
    test_x, test_y, test_maxnum = read_dataset(test_df, 'test')
    overal_maxnum = max(train_maxnum,dev_maxnum,test_maxnum)
    print('Max sentence number is %s' % overal_maxnum)

    return (train_x, train_y), (dev_x, dev_y), (test_x, test_y),overal_maxnum

