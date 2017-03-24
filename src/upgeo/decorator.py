'''
Created on Mar 8, 2011

@author: marcel
'''

class singleton(object):
    '''
    classdocs
    '''


    def __init__(self, cls): #on @ decoration
        '''
        Constructor
        '''
        self.__cls = cls
        self.__instance = None
        
    def __call__(self, *args): #on instance creation
        '''
        '''
        if self.__instance == None:
            self.__instance = self.__cls(*args)
        return self.__instance
    
        