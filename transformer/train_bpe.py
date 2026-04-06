import regex as re
from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def read_text(input_path):
    return open(input_path,'r',encoding='utf-8').read()

def init_vocab(special_tokens:list[str]):
    vocab = {idx:bytes([idx]) for idx in range(256)}
    if special_tokens:
        for i in range(len(special_tokens)):
            vocab[256+i] = special_tokens[i].encode('utf-8')
    return vocab

def split_text(text, special_tokens = None, drop_special = True) -> list[str]:
    if special_tokens == None:
        return [text]
    special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
    pattern = "|".join(re.escape(special_token) for special_token in special_tokens)
    if  not drop_special:
        pattern = f"({pattern})"
    compile_pattern = re.compile(pattern)
    text = compile_pattern.split(text)
    text = [c for c in text if c]    # remove empty strings
    return text

#通过PAT分割后统计每个单词的频率
def count_word(words):
    word_stats = defaultdict(int)  
    for word in words:
        if len(word)<2: 
            continue
        word_stats[word] += 1
    return word_stats

def count_pair(word_stats):
    pair_stats = defaultdict(int) 
    for word in word_stats:
        # if(len(word)<2):
        #     continue
        for pair in zip(word[:-1],word[1:]):
            pair_stats[pair] += word_stats[word]
    return pair_stats

def get_pair(pair_stats):
    pair, _ = max(pair_stats.items(), key=lambda x: (x[1], x[0]))  # lexicographic tie-breaker
    return pair
    # return max(pair_stats, key=pair_stats.get)



def merge(ids, pair):
    merged = pair[0] + pair[1]
    newids = []
    i = 0
    while i < len(ids):
        if ids[i]==pair[0] and i<len(ids)-1 and ids[i+1]==pair[1]:
            newids.append(merged)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return tuple(newids)

def update_stats(word_stats, pair_stats, pair):
    
    new_word_stats = defaultdict(int)
    new_pair_stats = defaultdict(int, pair_stats)
    for word in word_stats:
        count = word_stats[word]
        pair_list = list(zip(word[:-1],word[1:]))
        if pair not in pair_list:
            new_word_stats[word] += count
            continue
        new_word = merge(word, pair)
        new_word_stats[new_word] += count

        # new_pair_stats[pair] -= pair_stats[pair]
        for p in pair_list:
            new_pair_stats[p] -= count
            if new_pair_stats[p] == 0:
                del new_pair_stats[p]
        
        for new_pair in zip(new_word[:-1],new_word[1:]):
            new_pair_stats[new_pair] += count
        
    return new_word_stats, new_pair_stats

def word2bytes(ids):
    return tuple(bytes([i]) for i in ids)


def train_bpe(input_path, vocab_size, special_tokens = None):

    text = read_text(input_path)
    text_chunks = split_text(text, special_tokens)

    vocab = init_vocab(special_tokens)
    num_merge = vocab_size - len(vocab) 

    idx = len(vocab)
    pattern = PAT
    compiled_pattern = re.compile(pattern)

    words = [word2bytes(list(ch.group(0).encode('utf-8'))) for text_chunk in text_chunks for ch in re.finditer(compiled_pattern, text_chunk)]
    # words = [bytes([idx]) for word in words for idx in word]
    word_stats = count_word(words)
    pair_stats = count_pair(word_stats)
    merges = []
    for i in range(num_merge):
        
        pair = get_pair(pair_stats)
        vocab[idx+i] = pair[0] + pair[1] 
        merges.append(pair)
        # words = [merge(ids,pair,idx+i) for ids in words]
        word_stats,pair_stats = update_stats(word_stats, pair_stats, pair)
        
    return vocab, merges


 

