'''
Created on Jun 13, 2016

@author: hp
'''

from sys import maxsize

class Vars(object):
    '''
    Constant Variables
    '''

    # Default range
    rng = (116.1600, 39.8500, 116.3000, 40.1200)
    # RNG Cache File Name
    RNG_CACHE_FILE = 'mobpf_rng.cache'
    EDGE_DISTANCE_CACHE = 'distance.cache'

    # Earth Radius (Km and m)
    R = 6371.0
    Rm = 6371008.8

    INF = maxsize

    def __init__(self, params):
        '''
        Constructor
        '''
