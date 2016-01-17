'''
Created on 2016/01/09

@author: Xiang Long
'''
import numpy as np

class BigNumberDataEngine(object):
    '''
    classdocs
    '''
    num_chars = '0123456789'
    seen = set()

    def __init__(self, min_digits=1, max_digits=3, sort = True):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.sort = sort
            
    def get_number(self):
        n = np.random.randint(self.min_digits, self.max_digits+1)
        s = str(np.random.randint(1, 10))
        for _ in range(n-1):
            s += str(np.random.randint(0, 10))
        return s
    
    def add(self, a, b):
        a = a[::-1]
        b = b[::-1]
        left = 0
        length = max(len(a), len(b))
        c = ''
        for i in range(length):
            if i >= len(a):
                x = 0
            else:
                x = int(a[i])
            if i >= len(b):
                y = 0
            else:
                y = int(b[i])
            z = x + y + left
            if z>=10:
                z = z-10
                left = 1
            else:
                left = 0
            c += str(z)
        if left:
            c += '1'
        return c[::-1]
    
    def get_seperate_data(self):
        while 1:
            a, b = self.get_number(), self.get_number()  
            if self.sort:   
                key = tuple(sorted((a, b))) # Skip any such that A+B == A+B or A+B == B+A (hence the sorting)
            else:
                key = (a,b) # Skip any addition questions we've already seen 
            if key not in self.seen:
                self.seen.add(key)
                answer = self.add(a,b)
                return str(a), str(b), answer
    
    def get_seperate_dataset(self, size):
        As = []
        Bs = []
        answers = []
        for _ in range(size):
            a, b, ans = self.get_seperate_data()
            As.append(a)
            Bs.append(b)
            answers.append(ans)

        return As, Bs, answers
    
    def get_character_set(self):
        return '0123456789+'
        
            
if __name__ == '__main__':
    engine = NumberDataEngine(min_digits=15, max_digits=30)
    print engine.get_number()
    print engine.get_seperate_data()
    As, Bs, Cs = engine.get_seperate_dataset(5)
    for a, b, c in zip(As, Bs, Cs):
        print "%s + %s = %s" %(a,b,c)
    
    print 
    
    
    
    
    