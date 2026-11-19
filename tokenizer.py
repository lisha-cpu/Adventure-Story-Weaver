import json
from collections import Counter

class BPETokenizer:
    # Byte-level BPE: the initial vocabulary contains all 256 byte values.
    def __init__(self, merges=None):
        self.merges = merges or []
        self.ranks = {tuple(pair): i for i, pair in enumerate(self.merges)}

    @staticmethod
    def _pairs(tokens):
        return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

    def train(self, text, num_merges=300):
        tokens = list(text.encode("utf-8"))
        merges = []
        for _ in range(num_merges):
            pairs = self._pairs(tokens)
            if not pairs:
                break
            counts = Counter(pairs)
            pair, count = counts.most_common(1)[0]
            if count < 2:
                break
            new_id = 256 + len(merges)
            merged = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
                    merged.append(new_id); i += 2
                else:
                    merged.append(tokens[i]); i += 1
            merges.append(pair)
            tokens = merged
        self.merges = merges
        self.ranks = {tuple(pair): i for i, pair in enumerate(merges)}
        return self

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            pairs = self._pairs(tokens)
            ranked = [(self.ranks[p], p) for p in pairs if p in self.ranks]
            if not ranked:
                break
            _, pair = min(ranked)
            new_id = 256 + self.ranks[pair]
            merged = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
                    merged.append(new_id); i += 2
                else:
                    merged.append(tokens[i]); i += 1
            tokens = merged
        return tokens

    def decode(self, ids):
        pieces = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            pieces[256+i] = pieces[a] + pieces[b]
        raw = b"".join(pieces[int(i)] for i in ids if int(i) in pieces)
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["merges"])
