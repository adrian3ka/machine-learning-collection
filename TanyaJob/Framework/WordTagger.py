
# coding: utf-8

# In[79]:


from nltk.corpus import brown 
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize


# In[84]:


class TextTagger:
    bigram_tagger = nltk.tag.sequential.BigramTagger
    def __init__(self, fileName):
        patterns = ""
        sent_tagged = []
        patterns = [
            (r'meng[aiueokghx].+$', 'VB'),
            (r'mem[bpf]([a-df-z][a-qs-z]|er).+$', 'VB'),
            (r'me[lnryw](a-df-z).+$', 'VB'),
            (r'men[dtcjys].+$', 'VB'),
            (r'di.+(kan|i)$', 'VB'),
            (r'per.+(kan|i|.)$', 'VB'),
            (r'ber.+(kan|an|.)$', 'VB'),
            (r'ter.+(kan|i|.)$', 'VB'),
            (r'(meng|me|mem|men).+(kan|i)$', 'VB'),
            (r'ke.+(i|an)$', 'VB'),
            (r'se(baik|benar|tidak|layak|lekas|sungguh|yogya|belum|pantas|balik|lanjut)(nya)$', 'RB'),
            (r'(sekadar|amat|bahkan|cukup|jua|justru|kembali|kurang|malah|mau|nian|niscaya|pasti|patut|perlu|lagi|pernah|pun|sekali|selalu|senantiasa|sering|sungguh|tentu|terus|lebih|hampir|jarang|juga|kerap|makin|memang|nyaris|paling|pula|saja|saling|sangat|segera|semakin|serba|entah|hanya|kadangkala|)$', 'RB'),
            (r'(akan|antara|bagi|buat|dari|dengan|di|ke|kecuali|lepas|oleh|pada|per|peri|seperti|tanpa|tentang|untuk)$', 'IN'),
            (r'(dan|serta|atau|tetapi|melainkan|padahal|sedangkan)$','CC'),
            (r'(sejak|semenjak|sedari|sewaktu|ketika|tatkala|sementara|begitu|seraya|selagi|selama|serta|sambil|demi|setelah|sesudah|sebelum|sehabis|selesai|seusai|hingga|sampai|jika|kalau|jikalau|asal)$','SC'),
        ]
        for f in fileName:
            print "Learning >> " + f
            file = open(f, "r") 
            sent = file.read()
            data_sents = [nltk.tag.str2tuple(t) for t in sent.split()]
            temp_sent = []
            stop_word = [".","!","?"]
            for data in data_sents:
                temp_sent.append(data)
                if data[0] in stop_word:
                    sent_tagged.append(temp_sent)
                    temp_sent = []
                
        default_tagger = nltk.DefaultTagger('NN')
        regexp_tagger = nltk.RegexpTagger(patterns, backoff= default_tagger)
        unigram_tagger = nltk.UnigramTagger(sent_tagged, backoff=regexp_tagger)
        self.bigram_tagger = nltk.BigramTagger(sent_tagged, backoff=unigram_tagger)

    def getTagger(self,text):
        word_tokenize_list = word_tokenize(text)
        return self.bigram_tagger.tag(word_tokenize_list)

