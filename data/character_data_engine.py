import numpy as np

class CharacterDataEngine(object):
    '''
    Given a set of string:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen, soldier = '', mask=False):
        self.mask = mask
        self.soldier = soldier
        if mask:
            self.chars = sorted(set(chars))
            self.dim = len(self.chars);
            self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
            self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
            self.indices_char[-1] = soldier
            self.maxlen = maxlen            
        else:
            self.chars = sorted(set(chars))
            self.dim = len(self.chars) + 1;
            self.char_indices = dict((c, i + 1) for i, c in enumerate(self.chars))
            self.indices_char = dict((i + 1, c) for i, c in enumerate(self.chars))
            self.indices_char[0] = soldier
            self.maxlen = maxlen
    
    def set_maxlen(self, maxlen):
        self.maxlen = maxlen

    def encode(self, string, maxlen=None, invert=False, index=False):
        maxlen = maxlen if maxlen else self.maxlen
        if index:
            vectors = np.zeros((maxlen, self.dim + maxlen), dtype=np.bool)
            if invert:
                offset = maxlen - len(string)
                for i, c in enumerate(string):
                    vectors[offset + i, self.char_indices[c]] = True
                if not self.mask:
                    for i in range(offset):
                        vectors[i, 0] = True
                for i in range(maxlen):
                    vectors[i, self.dim + i] = True
                return vectors[::-1]
            else:
                for i, c in enumerate(string):
                    vectors[i, self.char_indices[c]] = True
                if not self.mask:
                    for i in range(len(string), maxlen):
                        vectors[i, 0] = True
                for i in range(maxlen):
                    vectors[self.dim + i, i] = True                    
                return vectors
        else:
            vectors = np.zeros((maxlen, self.dim), dtype=np.bool)
            if invert:
                offset = maxlen - len(string)
                for i, c in enumerate(string):
                    vectors[offset + i, self.char_indices[c]] = True
                if not self.mask:
                    for i in range(offset):
                        vectors[i, 0] = True
                return vectors[::-1]
            else:
                for i, c in enumerate(string):
                    vectors[i, self.char_indices[c]] = True
                if not self.mask:
                    for i in range(len(string), maxlen):
                        vectors[i, 0] = True
                return vectors

    def decode(self, vectors, calc_argmax=True, invert=False, index=False):
        if index:
            if calc_argmax:
                vectors = vectors[:,:-len(vectors)]
                vs = []
                for i in range(len(vectors)):
                    if vectors[i].max() > 1e-6:
                        vs.append(vectors[i].argmax())
                    else:
                        vs.append(-1);
                vectors = vs
            if invert:
                vectors = vectors[::-1]
            return ''.join(self.indices_char[v] for v in vectors)
        else:
            if calc_argmax:
                vs = []
                for i in range(len(vectors)):
                    if vectors[i].max > 1e-6:
                        vs.append(vectors[i].argmax())
                    else:
                        vs.append(-1);
                vectors = vs
            if invert:
                vectors = vectors[::-1]
            return ''.join(self.indices_char[v] for v in vectors)
    
    def encode_dataset(self, strings, maxlen=None, invert=False, index=False):
        maxlen = maxlen if maxlen else self.maxlen
        if index:
            datas = np.zeros((len(strings), maxlen, self.dim + maxlen), dtype=np.bool)
        else:
            datas = np.zeros((len(strings), maxlen, self.dim), dtype=np.bool)
        for i, sentence in enumerate(strings):
            datas[i] = self.encode(sentence, maxlen, invert, index)
        return datas
    
    def decode_dataset(self, datas, calc_argmax=True, invert=False, index=False):
        strings = []
        for vectors in datas:
            strings.append(self.decode(vectors, calc_argmax, invert, index))
        return strings
    
    def get_dim(self):
        return self.dim
    
if __name__ == '__main__':
    engine = CharacterDataEngine('0123456789+', 10)
    s = '0123+89'
    v = engine.encode(s)
    print v
    r = engine.decode(v)
    print r
    
    ss = ['12+34', '7890+54321', '0+0']
    d = engine.encode_dataset(ss, 12)
    print d
    rs = engine.decode_dataset(d)
    print rs
    
    engine = CharacterDataEngine('0123456789+', 10, soldier='#')
    s = '0123+89'
    v = engine.encode(s, invert=True, index=True)
    print v
    r = engine.decode(v, invert=True, index=True)
    print r
    
    
    engine = CharacterDataEngine('0123456789+', 10, soldier='#', mask=True)
    s = '0123+89'
    v = engine.encode(s, invert=True, index=True)
    print v
    r = engine.decode(v, invert=True, index=True)
    print r
    