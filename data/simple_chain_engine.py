import numpy as np

class SimpleChainEngine(object):
    
    def __init__(self, words):
        self.words = words
        self.dictionary = ['<EOS>']
        for s in self.words:
            self.dictionary.append(s)
        self.cnt = len(words)
            
    def get_start(self):
        return np.random.randint(self.cnt - 1)
    
    def get_chain(self, start):
        chain = [self.words[start]]
        for i in range(start + 1, self.cnt):
            chain.append(self.words[i])
        chain.append('<EOS>')
        return chain
            
    def get_data(self):
        start = self.get_start()
        return self.words[start], self.get_chain(start)
    
    def get_dataset(self, size):
        starts = []
        chains = []
        for _ in range(size):
            q, a = self.get_data()
            starts.append(q)
            chains.append(a)
        return starts, chains
    
    def get_dictionary(self):
        return self.dictionary
    
        
            
if __name__ == '__main__':
    engine = SimpleChainEngine('0123456789abcdef')
    s, c = engine.get_data()
    print "%s -> %s" %(s,c)
    ss, cs = engine.get_dataset(10)
    for (s, c) in zip(ss, cs):
        print "%s -> %s" %(s,c)
    
    print engine.get_dictionary() 