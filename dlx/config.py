import ConfigParser
import theano
import os

usr_home = os.path.expanduser('~')
default_path = os.path.join(usr_home, ".dlx/dlx.default.config.txt")
config_path = os.path.join(usr_home, ".dlx/dlx.config.txt") 

default_config = ConfigParser.ConfigParser()
default_config.read(default_path)
config = ConfigParser.ConfigParser()
config.read(config_path)

def get_config(sec, opt, dtype=''):
    try:
        return eval('config.get' + dtype)(sec, opt)
    except:
        try:
            return eval('config.get' + dtype)('common', opt)
        except:
            return eval('default_config.get' + dtype)(sec, opt)

floatX = get_config('common', 'floatX')
theano.config.floatX = floatX


epsilon = get_config('common', 'epsilon', 'float')

if __name__ == '__main__':
    print 'floatX:', floatX
    print 'epsilon:', epsilon