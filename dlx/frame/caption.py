import theano.tensor as T
from theano import function, config, shared


def default_encoder(x):
    return x
    
def default_decoder(x):
    return x      

class SequenceGenerator(object):


    def __init__(self, encoder=default_encoder, decoder=default_decoder):
        self.eos = '<EOS>'
        self.encoder = encoder
        self.decoder = decoder
    
    def set_eos(self, eos):
        self.eos = eos
    
    def is_eos(self, state):
        if self.eos == state:
            return True
        else:
            return False
    
    def next(self, state):
        pass
    
    def get_sequence(self, bos):
        seq = [bos]
        state = self.encoder(bos)
        while True:
            state = self.next(state)
            output = self.decoder(state)
            seq = seq + [output]
            if self.is_eos(output):
                break
        return seq
        

    
class NumberSequenceGenerator(SequenceGenerator):
    def digit_encoder(self, ch):
        return int(ch)

    def digit_decoder(self, i):
        if i > 9:
            return self.eos
        else:
            return str(i)
        
    def __init__(self):
        super(NumberSequenceGenerator, self).__init__(self.digit_encoder, self.digit_decoder)
         
    def next(self, state):
        return state + 1

    
        
class RandomSequenceGenerator(SequenceGenerator):
    def __init__(self):
        super(RandomSequenceGenerator, self).__init__()
        self.chars = 'ABCDEFG'
         
    def next(self, state):
        n = len(self.chars)
        import random
        idx = random.randint(0, n)
        if idx == n:
            return self.eos
        else:
            return self.chars[idx]
        
class FunctionSequenceGenerator(SequenceGenerator):
    def __init__(self, function, encoder, decoder):
        super(FunctionSequenceGenerator, self).__init__(encoder, decoder)
        self.function = function
    
    def next(self, state):
        return self.function(state)
        
        
if __name__ == '__main__':
    print 'NumberSequenceGenerator:'
    bos = '1'
    nsg = NumberSequenceGenerator()
    seq = nsg.get_sequence(bos)
    print seq
    bos = '6'
    nsg = NumberSequenceGenerator()
    seq = nsg.get_sequence(bos)
    print seq
    bos = '9'
    nsg = NumberSequenceGenerator()
    seq = nsg.get_sequence(bos)
    print seq
    bos = '0'
    nsg = NumberSequenceGenerator()
    seq = nsg.get_sequence(bos)
    print seq
    bos = '11'
    nsg = NumberSequenceGenerator()
    seq = nsg.get_sequence(bos)
    print seq
    
    print 'RandomSequenceGenerator:'
    bos = '<BOS>'
    rsg = RandomSequenceGenerator()
    seq = rsg.get_sequence(bos)
    print seq
    bos = '<BOS>'
    rsg = RandomSequenceGenerator()
    seq = rsg.get_sequence(bos)
    print seq
    bos = '<BOS>'
    rsg = RandomSequenceGenerator()
    seq = rsg.get_sequence(bos)
    print seq
    bos = '<BOS>'
    rsg = RandomSequenceGenerator()
    seq = rsg.get_sequence(bos)
    print seq
    bos = '<BOS>'
    rsg = RandomSequenceGenerator()
    seq = rsg.get_sequence(bos)
    print seq
    
    print 'FunctionSequenceGenerator:'
    x = T.scalar('x', 'int32')
    f = function([x], x+1)
    fsg = FunctionSequenceGenerator(f, nsg.digit_encoder, nsg.digit_decoder)
    bos = '1'
    seq = fsg.get_sequence(bos)
    print seq
    bos = '6'
    seq = fsg.get_sequence(bos)
    print seq
    bos = '0'
    seq = fsg.get_sequence(bos)
    print seq
    bos = '9'
    seq = fsg.get_sequence(bos)
    print seq
    bos = '11'
    seq = fsg.get_sequence(bos)
    print seq
    