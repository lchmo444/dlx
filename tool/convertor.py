import numpy as np

class word_one_hot_vector_convertor(object):
    
    def __init__(self, words_list):
        self.words_list = words_list
        dic = {}
        for idx, word in enumerate(words_list):
            dic[word] = idx
        self.dictionary = dic
        self.size = len(self.words_list)

    def word2one_hot_vector(self, word):
        '''
        make sure word is in the words list.
        '''
        one_hot_vec = np.zeros(self.size, 'bool')
        one_hot_vec[self.dictionary[word]] = 1
        return one_hot_vec
    
    def sentence2one_hot_matrix(self, sentence, maxlen = None): 
        if maxlen is None: 
            n = len(sentence)
        else:
            n = maxlen
        one_hot_matrix = np.zeros((n, self.size), 'bool')
        mask = np.zeros(n, 'bool')
        for idx, word in enumerate(sentence):
            one_hot_matrix[idx, self.dictionary[word]] = 1
            mask[idx] = 1
        if maxlen is None:
            return one_hot_matrix
        else:
            return one_hot_matrix, mask
            
    
    def sentences2one_hot_tensor(self, sentences, maxlen):
        n = len(sentences)
        m = maxlen
        k = self.size
        
        one_hot_tensor = np.zeros((n, m, k), 'bool')
        mask = np.zeros((n, m), 'bool')
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                one_hot_tensor[i, j, self.dictionary[word]] = 1
                mask[i, j] = 1
        return one_hot_tensor, mask

    def one_hot_vector2word(self, vec):
        '''
        vec maybe word probability
        '''
        return self.words_list[np.argmax(vec)]
    
    def one_hot_matrix2sentence(self, matrix, mask = None):
        idxs = np.argmax(matrix, 1)
        sentence = []
        if mask is None:
            for idx in idxs:
                sentence.append(self.words_list[idx])
        else:
            for i, idx in enumerate(idxs):
                if mask[i]:
                    sentence.append(self.words_list[idx])
                else:
                    break            
        return sentence
    
    def one_hot_tensor2sentences(self, tensor, mask):
        idxss = np.argmax(tensor, 2)
        sentences = []
        for i, idxs in enumerate(idxss):
            sentence = []
            for j, idx in enumerate(idxs):
                if mask[i, j]:
                    sentence.append(self.words_list[idx])
                else:
                    break
            sentences.append(sentence)
        return sentences        

def test_word_one_hot_vector_convertor():       
    from data.simple_chain_engine import SimpleChainEngine
    engine = SimpleChainEngine('0123456789abcdef')
    s, c = engine.get_data()
    print "%s -> %s" %(s,c)
    ss, cs = engine.get_dataset(5)
    for (s, c) in zip(ss, cs):
        print "%s -> %s" %(s,c)
    print engine.get_dictionary() 
    
    convertor = word_one_hot_vector_convertor(engine.get_dictionary())
    for word in engine.get_dictionary():
        print "%s -> %s" %(word, convertor.word2one_hot_vector(word).astype('int8'))
        
    for word in engine.get_dictionary():
        print "%s -> %s" %(word, convertor.one_hot_vector2word(convertor.word2one_hot_vector(word).astype('int8')))
    
    matrixs = []
    for c in cs:
        matrixs.append(convertor.sentence2one_hot_matrix(c))
    
    for c, matrix in zip(cs, matrixs):
        print "%s -> " %(c)
        print matrix.astype('int8')

    for c, matrix in zip(cs, matrixs):
        print "%s -> %s" %(c, convertor.one_hot_matrix2sentence(matrix))
        
    maxlen = len(engine.get_dictionary()) + 10
    matrixs = []
    masks = []
    for c in cs:
        matrix, mask = convertor.sentence2one_hot_matrix(c, maxlen)
        matrixs.append(matrix)
        masks.append(mask)

    for c, matrix, mask in zip(cs, matrixs, masks):
        print "%s -> %s" %(c, convertor.one_hot_matrix2sentence(matrix, mask))
        
    tensor, mask = convertor.sentences2one_hot_tensor(cs, len(engine.get_dictionary()))
    #print 'tensor:'
    #print tensor
    #print 'mask:'
    #print mask
    recs = convertor.one_hot_tensor2sentences(tensor, mask)
    for c ,rec in zip(cs, recs):
        print "%s -> %s" %(c, rec)
    

'''
class one_hot_embedding_convertor(object):
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        self.pinv_embedding_matrix = np.linalg.pinv(embedding_matrix)
        
    def one_hot2embedding(self, vec):
        return np.dot(vec, self.embedding_matrix)
    
    def embedding2one_hot(self, emb):
        return np.dot(emb, self.pinv_embedding_matrix)
   
        
def test_one_hot_embedding_convertor():       
    from data.simple_chain_engine import SimpleChainEngine
    engine = SimpleChainEngine('0123456789abcdef')
    s, c = engine.get_data()
    print "%s -> %s" %(s,c)
    ss, cs = engine.get_dataset(10)
    for (s, c) in zip(ss, cs):
        print "%s -> %s" %(s,c)
    print engine.get_dictionary() 
    
    convertor1 = word_one_hot_vector_convertor(engine.get_dictionary())  
    convertor2 = one_hot_embedding_convertor(np.random.random((len(engine.get_dictionary()), 11)))

    for word in engine.get_dictionary():
        print "%s -> %s" %(word, convertor2.one_hot2embedding(convertor1.word2one_hot_vector(word)))
        
    for word in engine.get_dictionary():
        print "%s -> %s" %(word, convertor1.one_hot_vector2word(convertor2.embedding2one_hot(convertor2.one_hot2embedding(convertor1.word2one_hot_vector(word)))))
    
    matrixs = []
    for c in cs:
        matrixs.append(convertor1.sentence2one_hot_matrix(c))
    
    embs = []
    for c, matrix in zip(cs, matrixs):
        embs.append(convertor2.one_hot2embedding(matrix))
#        print "%s -> " %(c)
#        print embs

    for c, emb in zip(cs, embs):
        print "%s -> %s" %(c, convertor1.one_hot_matrix2sentence(convertor2.embedding2one_hot(emb)))   
'''
        
if __name__ == '__main__':
    test_word_one_hot_vector_convertor()
    #test_one_hot_embedding_convertor()