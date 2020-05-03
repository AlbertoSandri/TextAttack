from textattack.tokenizers import AutoTokenizer

class T5Tokenizer(AutoTokenizer):
    """ Uses the T5 tokenizer to convert an input for processing. 
        
        For more information, please see the T5 paper, "Exploring the Limits of 
            Transfer Learning with a Unified Text-to-Text Transformer".
            Appendix D contains information about the various tasks supported
            by T5.
        
        Supports the following modes:
            'summarization': summarize English text (CNN/Daily Mail dataset)
            'english_to_german': translate English to German (WMT dataset)
            'english_to_french': translate English to French  (WMT dataset)
            'english_to_romanian': translate English to Romanian (WMT dataset)
            
    """
    def __init__(self, mode='english_to_german', max_seq_length=64):
        if mode == 'english_to_german':
            self.tokenization_prefix = 'translate English to German: '
        elif mode == 'english_to_french':
            self.tokenization_prefix = 'translate English to French: '
        elif mode == 'english_to_romanian':
            self.tokenization_prefix = 'translate English to Romanian: '
        elif mode == 'summarization':
            self.tokenization_prefix = 'summarize: '
        else:
            raise ValueError(f'Invalid t5 tokenizer mode {english_to_german}.')

        super().__init__(name='t5-base', max_seq_length=max_seq_length)

    def encode(self, text):
        """ Encodes a string into IDs of tokens. This prepares an input to be
                passed into T5.
        """
        text_to_encode = self.tokenization_prefix + text
        return super().encode(text_to_encode)
    
    def decode(self, ids):
        """ Converts IDs (typically generated by the model) back to a string. 
        """
        return self.tokenizer.decode(ids)
        
