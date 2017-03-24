'''
Created on Apr 4, 2011

@author: marcel
'''
from _abcoll import MutableSequence, Container

class _Node(object):
    '''
    classdocs
    '''
    __slots__ = ('cargo', 'next')

    def __init__(self, cargo=None, next=None):
        '''
        Constructor
        '''
        self.cargo = cargo
        self.next = next

    def set_next(self, node):
        self.next = node

    def __str__(self):
        return str(self.cargo)
    
class _DLNode(_Node):
    __slots__ = ('previous')
    
    def __init__(self, cargo=None, next=None, previous=None):
        _Node.__init__(self, cargo, next)
        self.previous = previous
    
class LinkedList(MutableSequence):
    
    '''
    The LinkedList is internally represented by a singly linked list.
      
    @todo: - allow slicing
           - optimize implementation
           - implement an immutable view
    '''
    __slots__ = ('_length', '_head')
    
    def __init__(self, values=None):
        '''
        '''
        self._length = 0
        self._head = _Node() 
        
        self.extend(values)
    
    def insert(self, index, value):
        '''
        '''
        if type(index) != int:
            raise TypeError("list indices must be integers, not %s" % type(index))
        if index < 0:
            index = self._length+1+index
        if index < 0 or index > self._length:
            raise IndexError()
        
        tmp = _Node(value)
        current = self._head
        pos = 0
        while pos<index:
            current = current.next
            pos += 1
        
        tmp.set_next(current.next)
        current.set_next(tmp)
        self._length += 1
    
    def extend(self, values):
        if isinstance(values, Container):
            for v in values:
                self.append(v)
        else:
            self.append(values)
        
    def reverse(self):
        rev_list = LinkedList()
        for v in self:
            rev_list.add_first(v)
        self._head = rev_list._head
    
    def add_first(self, value):
        '''
        '''
        tmp = _Node(value)
        tmp.set_next(self._head.next)
        self._head.set_next(tmp)
    
    def add_last(self, value):
        '''
        '''
        self.insert(self._length, value)
    
    def first(self):
        '''
        '''
        if self.isempty():
            raise AttributeError("list is empty")
        
        return self._head.next.cargo
    
    def last(self):
        '''
        '''
        if self.isempty():
            raise AttributeError("list is empty.")
        return self[self._length-1]
    
    def clear(self):
        '''
        '''
        self._length = 0
        self._head = None
        
    def isempty(self):
        '''
        '''
        return not bool(self)
    
    def __getitem__(self, index):
        '''
        '''
        self._range_check(index)
        
        current = self._head
        pos = -1
        while pos<index:
            current = current.next
            pos += 1
        
        return current.cargo
    
    def __setitem__(self, index, value):
        '''
        '''
        self._range_check(index)
        
        current = self._head
        pos = -1
        while pos<index:
            current = current.next
            pos += 1

        current.cargo = value
    
    def __delitem__(self, index):
        '''
        '''
        self._range_check(index)
        
        if self._length == 1:
            self._head.set_next(None)
        else: 
            current = self._head
            pos = 0
            while pos<index:
                current = current.next
                pos += 1
            current.set_next(current.next.next)

        self.length -= 1
    
    def __iter__(self):
        '''
        '''
        current = self._head
        while current.next != None:
            current = current.next
            yield current.cargo
    
    def __reversed__(self):
        rev_list = self.copy()
        rev_list.reverse()
        return iter(rev_list)
    
    def __len__(self):
        '''
        '''
        return self._length
    
    def __nonzero__(self):
        '''
        '''
        return self._length != 0
    
    def __eq__(self, other):
        '''
        '''
        if id(self) == id(other):
            return True
        
        if isinstance(other, Container) and self._length == len(other):
            it = iter(self)
            other_it = iter(other)
            try:
                while True:
                    if it.next() != other_it.next():
                        return False
            except StopIteration:
                return True
        
        return False
    
    def __ne__(self, other):
        '''
        '''
        return not self.__eq__(other)
    
    def __str__(self):
        tmp = [element for element in self]
        return str(tmp)
    
    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self) + ")"
    
    def copy(self):
        return self.__class__(self)
    
    def _range_check(self, index):
        if type(index) != int:
            raise TypeError("list indices must be integers, not %s" % type(index))
        if index < 0 or index > self._length-1:
            raise IndexError("Index %s out of bound." % index)
    
    
class DLinkedList(LinkedList):
    
    __slots__ = ()
    
    def __init__(self, values=None):
        '''
        '''
        self._length = 0
        self._head = _DLNode()
        self._head.next = self._head.previous = self._head 
        
        self.extend(values)
    
    def insert(self, index, value):
        '''
        '''
        if type(index) != int:
            raise TypeError("list indices must be integers, not %s" % type(index))
        if index < 0:
            index = self._length+1-index
        
        
        node = self._get_node(index) if index != self._length else self._head
        self._add_before(value, node)
    
    def add_first(self, value):
        '''
        '''
        self._add_before(value, self._head.next)
    
    def add_last(self, value):
        '''
        '''
        self._add_before(value, self._head)
       
    def last(self):
        '''
        '''
        if self.isempty():
            raise AttributeError("list is empty.")
        return self._head.previous.cargo
            
    
    def __getitem__(self, index):
        '''
        '''
        node = self._get_node(index)
        return node.cargo
    
    def __setitem__(self, index, value):
        '''
        '''
        node = self._get_node(index)
        node.cargo = value
            
    def __delitem__(self, index):
        '''
        '''
        node = self._get_node(index)
        
        node.previous.next = node.next
        node.next.previous = node.previous
        del node
        self._length -= 1
    
    def __iter__(self):
        '''
        '''
        current = self._head
        while current.next != self._head:
            current = current.next
            yield current.cargo
    
    def __reversed__(self):
        current = self._head
        while current.next != self._head:
            current = current.previous
            yield current.cargo
    
    def _get_node(self, index):
        self._range_check(index)
        
        node = self._head
        if index < (self._length >> 1):
            pos = -1
            while pos < index:
                node = node.next
                pos += 1
        else:
            pos = self._length
            while pos > index:
                node = node.previous
                pos -= 1
        return node
    
    def _add_before(self, value, node):
        new_node = _DLNode(value, node, node.previous)
        new_node.previous.next = new_node
        new_node.next.previous = new_node
        self._length += 1
