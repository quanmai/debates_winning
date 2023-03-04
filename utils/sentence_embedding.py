import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from transformers import AutoTokenizer, AutoModel
import torch
from utils.config import config
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation


special_tokens = ['[URL]', '[NUM]', '[QUOTE]']

tokenizer = AutoTokenizer.from_pretrained(config.bert, additional_special_tokens=special_tokens)
model = AutoModel.from_pretrained(config.bert)
# model.to(config.device)

stop_words = set(stopwords.words('english'))

# def preprocess(text):
#     tokens = [token for token in nltk.word_tokenize(text) if token not in punctuation and token not in stop_words]
#     return ' '.join(tokens)

#https://stackoverflow.com/questions/61708486/whats-difference-between-tokenizer-encode-and-tokenizer-encode-plus-in-hugging
def sentence_embedding(sentences: list[str]):
    tokens = {'input_ids': [], 'attention_mask': []}
    # print(sentences)
    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.

        # sent = preprocess(sent)
        # if len(sent.split(' ')) > 3:
            # print('HeyHey')
        # print(sent)

        new_tokens = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = config.max_length,           # Pad & truncate all sentences.
                            truncation=True,
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                        )
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    tokens['input_ids'] = torch.stack(tokens['input_ids']) #torch.Size([#sent,128])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask']) #torch.Size([#sent,128])
    # tokens['input_ids']  = tokens['input_ids'].to(config.device)
    # tokens['attention_mask']  = tokens['attention_mask'].to(config.device)

    # tokens = tokens.to(config.device)
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state # torch.Size([#sent, 128, 768])
    # print(embeddings.shape)

    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float() #shape: [#sent,128,768]
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    # print('Done')

    return mean_pooled.detach().numpy()