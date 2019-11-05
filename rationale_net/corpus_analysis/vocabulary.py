import math

# vocabulary of exaplanations
# for the beer dataset
class Vocabulary:
    def __init__(self, corpus):
        # dictionary containing a list of documents for each class
        # where a document is a NL text - string
        # class/label: text (list of strings)
        self.class_documents = self._split_corpus(corpus)
        
    def _split_corpus(self, corpus):
        docs = {}
        for inst in corpus:
            docs[inst['y']] = docs.get(inst['y'],[])
            docs[inst['y']].append(inst['text'])
        return docs

    def tokenizer(self, text):
        """
        text(string)->list of tokens
        """
        return text.split()


class TfidfVocabulary(Vocabulary):
    """TF-IDF for words relative to the document of each class
    (each document comprises the collection of text labeled the same)
    """
    def __init__(self, data, vocab_dim=200):
        Vocabulary.__init__(self, data)
        self.max_dim=vocab_dim
        # Number of documents = number of distinc labels
        self.N = len(self.class_documents.keys())
        # tf: {doc_name:{word:freq}}
        # df: word: word_count_in_corpus
        self.tf, self.df = self.t_freq()
        # {doc_name:{word:score}}
        self.tf_idf = self.tf_idf_scores()

    def t_freq(self):
        tf = {}
        term_freq = {}
        for doc, texts in self.class_documents.items():
            doc_freq = {}
            for text in texts:
                for word in self.tokenizer(text):
                    doc_freq[word] = doc_freq.get(word,0)+1
                    term_freq[word] = term_freq.get(word, 0) + 1
            tf[doc] = doc_freq
        return tf, term_freq    

    def _tf_idf(self, t, d_name):
        return self.tf[d_name][t] * math.log(self.N/(self.df[t]+1))

    def tf_idf_scores(self):
        # {doc(label): {word:score}}
        tf_idf = {}
        for doc_name in self.class_documents.keys():
            tf_idf_doc = {}
            for text in self.class_documents[doc_name]:
                for word in self.tokenizer(text):
                    tf_idf_doc[word] = self._tf_idf(word, doc_name)
            tf_idf[doc_name] = tf_idf_doc
        return tf_idf

    def possible_explanations(self):
        """
        Return a list of possible explanations (as text/strings)
        equally distributed among the documents
        """
        explanations = []
        for doc in self.class_documents.keys():
            ranked_words = sorted(self.tf_idf[doc], key=self.tf_idf[doc].get, reverse=True)
            explanations.extend(ranked_words[:self.max_dim//self.N])
        return {i: explanation for i,explanation in enumerate(explanations)}

