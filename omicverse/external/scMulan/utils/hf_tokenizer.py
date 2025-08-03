from transformers import PreTrainedTokenizer

class scMulanTokenizer(PreTrainedTokenizer):
    def __init__(self, chars):
        self.chars = chars
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        super().__init__()

    def _tokenize(self, text):
        # Implement your tokenization method here
        return [self.stoi[c] for c in text.split('##')]

    def _convert_token_to_id(self, token):
        return self.stoi[token]

    def _convert_id_to_token(self, index):
        return self.itos[index]

    def convert_tokens_to_string(self, tokens):
        return '##'.join(tokens)
    
    def get_vocab(self):
        import transformers
        if transformers.__version__ <= "4.40":
            return self.stoi
        else:
            try:
                return self.get_stoi()
            except:
                return self.stoi
    
    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
