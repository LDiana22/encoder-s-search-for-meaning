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
    def __init__(self, args, data, vocab_dim=400):
        Vocabulary.__init__(self, data)
        self.args = args
        self.path = args.vocab_path + "tfidf-400.txt"
        self.max_dim=vocab_dim
        # Number of documents = number of distinc labels
        self.N = len(self.class_documents.keys())
        # tf: {doc_name:{word:freq}}
        # df: word: word_count_in_corpus
        self.tf, self.df = self.t_freq()
        # {doc_name:{word:score}}
        self.tf_idf = self.tf_idf_scores()
        print(f"Corpus of {sum([len(words) for doc, words in self.class_documents.items()])}")
        if self.args.aspect =="aroma":
            self.ignored_words = []#["snowboarding"]
        elif self.args.aspect =="palate":
            self.ignored_words = []#["barbershop"]
        elif self.args.aspect == "appearance":
            self.ignored_words = []#["lodi", "shirley", "duff"]
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

    def ignore_words(self, word_list):
        """
        Returns: 
            list of removed words (that are in self.ignored_words)
            list with the remaining words
        """
        remove = []
        for word in word_list:
            if word.lower().strip() in self.ignored_words:
                remove.append(word)
                word_list.remove(word)
        return remove, word_list
                

    def possible_explanations(self):
        """
        Return a list of possible explanations (as text/strings)
        equally distributed among the documents
        """
        doc_count = {}
        with open(self.args.vocab_path+"doc-distr.txt", "w") as f:
            explanations = []
            for doc in self.class_documents.keys():
                ranked_words = sorted(self.tf_idf[doc], key=self.tf_idf[doc].get, reverse=True)
                removed_words, remaining_words = self.ignore_words(ranked_words)
                explanations.extend(remaining_words[:self.max_dim//self.N])
                print(f"doc: {doc}\n{ranked_words}\nRemoved:\n{removed_words}~~~", file=f)
                doc_count[doc] = len(remaining_words)
            expl = {i: explanation for i,explanation in enumerate(explanations)}
        with open(self.path, 'w') as f:
            f.write(str(expl))
            f.write("Dict len:")
            f.write(str(len(expl)))
            f.write(f"Doc distribution\n{doc_count}")
        return expl

