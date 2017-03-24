'''
Created on Mar 31, 2011

@author: marcel
'''
from abc import ABCMeta, abstractproperty
from _abcoll import Container

class ITree(object):
    
    __metaclass__ = ABCMeta
    
    __slots__ = ()
    

class AbstractBTree(ITree):
    """The binary tree "interface" class.

    It has three properties: cargo, and the left and right subtrees.
    A terminal node (= atomic tree) is one where the left and right
    subtrees are the empty tree."""
    
    __metaclass__ = ABCMeta
    
    __slots__ = ()
    
    @abstractproperty
    def parent(self):
        pass
    
    @abstractproperty
    def left(self):
        pass
    
    @abstractproperty
    def right(self):
        pass
    
    @abstractproperty
    def cargo(self):
        pass
        
    def isroot(self):
        """Returns 1 if the tree has no parent tree, 0 otherwise."""
        return self.parent == None

    def isatom(self):
        """Returns 1 if the tree has no nonempty subtrees, 0 otherwise."""
        return not (self.left != None or self.right != None)
    
    def __nonzero__(self):
        return self.cargo != None
    
    #The simplest print possible.
    def __str__(self):
        return "(%s, %s, %s)" % (str(self.cargo), str(self.left), str(self.right))

    def root(self):
        root = self
        while root.parent != None:
            root = root.parent
        return root
    
    def leaves(self):
        leaves = []
        for tree in self.subtree():
            if tree.isatom():
                leaves.append(tree)
        return leaves
    
    
   #The BTree iterators.
    def __iter__(self):
        """The standard preorder traversal of a binary tree."""
        if self.cargo != None:
            yield self.cargo
        if self.left != None:
            for element in self.left:
                yield element
        if self.right != None:
            for element in self.right:
                yield element

    def postorder(self):
        """Postorder traversal of a binary tree."""
        if self.left != None:
            for element in self.left.postorder():
                yield element
        if self.right != None:
            for element in self.right.postorder():
                yield element
        if self.cargo != None:
            yield self.cargo
        
    def inorder(self):
        """Inorder traversal of a binary tree."""
        if self.left != None:
            for element in self.left.inorder():
                yield element
        if self.cargo != None:
            yield self.cargo
        if self.right != None:
            for element in self.right.inorder():
                yield element

    #"Inplace" iterators.
    def subtree(self):
        """Preorder iterator over the (nonempty) subtrees.

        Warning: As always, do not use this iterator in a for loop while altering
        the structure of the tree."""
        yield self
        if self.left != None:
            for tree in self.left.subtree():
                yield tree
        if self.right != None:
            for tree in self.right.subtree():
                yield tree

    def postsubtree(self):
        """Postorder iterator over the (nonempty) subtrees.

        Warning: As always, do not use this iterator in a for loop while altering
        the structure of the tree."""
        if self.left != None:        
            for tree in self.left.postsubtree():
                yield tree
        if self.right != None:
            for tree in self.right.postsubtree():
                yield tree
        yield self

    def insubtree(self):
        """Inorder iterator over the (nonempty) subtrees.

        Warning: As always, do not use this iterator in a for loop while altering
        the structure of the tree."""        
        if self.left != None:
            for tree in self.left.insubtree():
                yield tree
        yield self
        if self.right != None:
            for tree in self.right.insubtree():
                yield tree

    #Binary comparisons.
    def __eq__(self, other):
        """Checks for equality of two binary trees."""
        #Both trees not empty.
        if id(self) == id(other):
            return True
        if other == None or self.cargo != other.cargo:
            return False
        return self.left == other.left and self.right == other.right
        
    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, element):
        """Returns 1 if element is in some node of the tree, 0 otherwise."""
        if self.cargo == element:
            return True
        for tree in self.subtree():
            if tree.cargo == element:
                return True
        return False

    def __len__(self):
        """Returns the number of nodes (elements) in the tree."""
        size = 1
        for child in self.subtree():
            size += 1
        return size

#The abstract generalized tree class where most of the methods reside.
class AbstractTree(ITree):
    """The generalized "interface" tree class.

    It has two properties: the cargo and a childs iterator giving the child subtrees.

    The childs property returns a new (reset) iterator each time it is called.
    There is no order of iteration through the nodes (implementation is free to
    swap them around). """
    
    __metaclass__ = ABCMeta
    
    __slots__ = ()
        
    @abstractproperty
    def parent(self):
        pass
    
    @abstractproperty
    def children(self):
        pass
    
    @abstractproperty
    def cargo(self):
        pass

    def isroot(self):
        return self.parent == None    

    def isatom(self):
        """A tree is atomic if it has no subtrees."""
        return self.children == None

    def root(self):
        root = self
        while root.parent != None:
            root = root.parent
        return root
    
    def leaves(self):
        leaves = []
        for tree in self.subtree():
            if tree.isatom():
                leaves.append(tree)
        return leaves

    #The simplest print possible.
    def __str__(self):
        if self.isatom():
            return "(%s)" % str(self.cargo)
        else:
            tmp = [str(subtree) for subtree in self.children]
            return "(%s, %s)" % (str(self.cargo), ", ".join(tmp))

    #The Tree iterators.
    def __iter__(self):
        """The standard preorder traversal iterator."""
        if self.cargo != None:
            yield self.cargo
        if self.children != None:
            for subtree in self.children:
                for element in subtree:
                    yield element

    def postorder(self):
        """Postorder traversal of a tree."""
        if self.children != None:
            for subtree in self.children:
                for element in subtree.postorder():
                    yield element
        if self.cargo != None:
            yield self.cargo

    #The "inplace" iterators.
    def subtree(self):
        """Preorder iterator over the subtrees.

        Warning: As always, do not use this iterator in a for loop while altering
        the structure of the tree."""

        yield self
        if self.children != None:
            for child in self.children:
                for tree in child.subtree():
                    yield tree

    def postsubtree(self):
        """Postorder iterator over the subtrees.

        Warning: As always, do not use this iterator in a for loop while altering
        the structure of the tree."""

        if self.children != None:
            for child in self.children:
                for tree in child.postsubtree():
                    yield tree
        yield self

    #The in protocol.
    def __contains__(self, element):
        """Returns 1 if elem is in the tree, 0 otherwise."""
        if self.cargo == element:
            return True
        for tree in self.subtree():
            if tree.cargo == element:
                return True
        return False

    #Number of elements in the tree.
    def __len__(self):
        """Returns the number of elements (nodes) in the tree."""
        size = 1
        for child in self.subtree():
            size += 1
        return size
    
    def nchildren(self):
        return len

    def copy(self):
        """Shallow copy of a Tree object."""
        
        if self.isatom():
            return self.__class__(self.cargo)
        else:
            tmp = tuple([subtree.copy() for subtree in self.childs])
            return self.__class__(self.cargo, *tmp)

#The two implementations of BTree class.
class MutableBTree(AbstractBTree):
    """A mutable implementation of the binary tree BTree class."""

    __slots__ = ('__cargo', '__parent', '__left', '__right')
    
    def __init__(self, cargo=None, parent=None, left=None, right=None):
        """The initializer."""
        self.__cargo = cargo
        self.__parent = None
        self.__left = None
        self.__right = None
        
        if parent != None: 
            if not isinstance(parent, MutableBTree):
                raise TypeError, "Object %s is not a MutableBTree binary tree." % repr(parent)
            self.parent = parent
        if left != None: 
            if not isinstance(left, MutableBTree):
                raise TypeError, "Object %s is not a MutableBTree binary tree." % repr(left)
            self.left = left
        if right != None: 
            if not isinstance(right, MutableBTree):
                raise TypeError, "Object %s is not a MutableBTree binary tree." % repr(right)
            self.right = right
        
    #Properties.
    def __get_cargo(self):
        return self.__cargo
        
    def __set_cargo(self, cargo):            
        self.__cargo = cargo

    def __del_cargo(self):
        self.__cargo = None

    cargo = property(__get_cargo, __set_cargo, __del_cargo, "The root element of the tree.")

    def __get_parent(self):
        return self.__parent
    
    def __set_parent(self, tree):
        if isinstance(tree, MutableBTree):
            if tree.__left == None:
                tree.__left = self
            elif tree.__right == None:
                tree.__right = self
            else:
                raise ValueError("tree has already a left and right child.")
            self.__left = tree
            self.__parent = tree
        else:
            raise TypeError, "Object %s is not a MutableBTree." % repr(tree)
        
    def __del_parent(self):
        '''
        @todo: - throw an error, if parent doesn't contain the self tree.
        '''
        if self.__parent != None:
            if id(self.__parent.__left) == id(self):
                self.__parent.__left = None
            elif id(self.__parent.__right) == id(self):
                self.__parent.__right = None
            self.__parent = None
    
    parent = property(__get_parent, __set_parent, __del_parent, "The parent")
    
    def __get_left(self):
        return self.__left
        
    def __set_left(self, tree):
        if isinstance(tree, MutableBTree):
            if tree.parent != None:
                raise ValueError("tree is already a child node of another tree.")
            self.__left = tree
            tree.__parent = self
        else:
            raise TypeError, "Object %s is not a MutableBTree." % repr(tree)
        
    def __del_left(self):
        self.__left.__parent = None
        self.__left = None

    left = property(__get_left, __set_left, __del_left, "The left subtree.")

    def __get_right(self):
        return self.__right
        
    def __set_right(self, tree):
        if isinstance(tree, MutableBTree):
            if tree.parent != None:
                raise ValueError("tree is already a child node of another tree.")
            self.__right = tree
            tree.__parent = self
        else:
            raise TypeError, "Object %s is not a MutableBTree." % repr(tree)
        
    def __del_right(self):
        self.__right.__parent = None
        self.__right = None

    right = property(__get_right, __set_right, __del_right, "The right subtree.")

    #General inplace transformations of mutable binary trees.
    def map(self, func):
        """Inplace map transformation of a binary tree."""
        self.cargo = func(self.cargo)
        for tree in self.subtree():
            tree.cargo = func(tree.cargo)

    def make_immutable(self):
        """Returns an ImmutableBTree copy."""
        left = self.__left.make_immutable() if self.__left != None else None
        right = self.__right.make_immutable() if self.__right != None else None
        return ImmutableBTree(self.__cargo, left, right)
    
    def copy(self):
        """Shallow copy of a BTree object."""
        left = self.left.copy() if self.left != None else None
        right = self.right.copy() if self.right != None else None
        return self.__class__(self.cargo, None, left, right)



class ImmutableBTree(AbstractBTree):
    """An implementation of an immutable binary tree using tuples."""

    __slots__ = ('__head', '__parent') 

    def __init__(self, cargo=None, left=None, right=None):
        """The initializer."""
        if left != None and not isinstance(left, ImmutableBTree):
            raise TypeError, "Object %s is not a ImmutableBTree binary tree." % repr(left)
        if right != None and not isinstance(right, ImmutableBTree):
            raise TypeError, "Object %s is not a ImmutableBTree binary tree." % repr(right)
        
        if left != None:
            if left.__head[1] != None:
                raise ValueError("left tree is already a child node of another tree.")
            left.__parent = self
        if right != None: 
            if right.__head[1] != None:
                raise ValueError("left tree is already a child node of another tree.")
            right.__parent = self
        
        self.__head = (cargo, left, right)
        
    #Properties.
    def __get_cargo(self):
        return self.__head[0]

    cargo = property(__get_cargo, None, None, "The root element of the tree.")

    def __get_parent(self):
        return self.__parent
    
    parent = property(__get_parent, None, None, "The parent tree.")

    def __get_left(self):
        return self.__head[1]

    left = property(__get_left, None, None, "The left subtree.")

    def __get_right(self):
        return self.__head[2]
        
    right = property(__get_right, None, None, "The right subtree.")

    #Conversion method.
    def make_mutable(self):
        """Returns a MutableBTree copy."""
        left = self.__left.make_mutable() if self.__left != None else None
        right = self.__right.make_mutable() if self.__right != None else None
        return MutableBTree(self.__cargo, None, left, right)
        
    def copy(self):
        """Shallow copy of a BTree object."""
        left = self.left.copy() if self.left != None else None
        right = self.right.copy() if self.right != None else None
        return self.__class__(self.cargo, None, left, right)
    
#Tree implementations.
class MutableTree(AbstractTree):
    """Class implementing a mutable tree type."""

    __slots__ = ('__parent', '__children', '__cargo')

    def __init__(self, cargo=None, parent=None, *trees):
        """The initializer."""
        
        self.cargo = cargo
        self.parent = parent
        if trees:
            self.children = trees
        
            
    #Properties.
    def __get_cargo(self):
        return self.__cargo

    def __set_cargo(self, cargo):
        self.__cargo = cargo

    def __del_cargo(self):
        self.__cargo = None

    cargo = property(__get_cargo, __set_cargo, __del_cargo, "The element of the tree.")
    
    def __get_parent(self):
        return self.__parent
    
    def __set_parent(self, tree):
        if isinstance(tree, MutableTree):
            tree.graft(self)
        else:
            raise TypeError, "Object %s is not a MutableBTree." % repr(tree)
        
    def __del_parent(self):
        '''
        @todo: - throw an error, if parent doesn't contain the self tree.
        '''
        if self.__parent != None:
            self.__parent.ungraft(self)
            self.__parent = None
    
    parent = property(__get_parent, __set_parent, __del_parent, "The parent of the tree")

    def __get_children(self):
        return iter(self.__children)
    
    def __set_children(self, *trees):
        del self.children
        for tree in trees:
            if not isinstance(tree, MutableTree):
                raise TypeError, "%s is not a MutableTree instance." % repr(tree)
            if tree.__parent != None:
                raise ValueError, ""
            tree.__parent = self
        self.__children = list(trees)
        
    def __del_children(self):
        if self.children != None:
            for child in self.children:
                child.__parent = None
            self.children = None 
    
    children = property(__get_children, __set_children, __del_children, "The iterator over the child subtrees.")

    #Add or delete trees to the root of the tree.
    def graft(self, tree):
        """Graft a tree to the root node."""
        if isinstance(tree, MutableTree):
            if tree.__parent != None:
                raise ValueError("tree is already a child node of another tree.")
            tree.__parent = self
            self.__children.append(tree)
        else:
            raise TypeError, "%s is not a Tree instance." % repr(tree)
        
    def ungraft(self, tree):
        """Ungrafts a subtree from the current node.

        The argument is the subtree to ungraft itself."""
        if self.children != None:
            for index, child in enumerate(self.children):
                if tree == child:
                    del self.__children[index]
                    return None
            raise AttributeError, "Tree %s is not grafted to the root node of this tree." % repr(tree)
        

    #General inplace transformations of trees.
    def map(self, func):
        """Inplace map transformation of a tree."""
        for tree in self.subtree():
            tree.cargo = func(tree.cargo)

    #Conversion methods.
    def make_immutable(self):
        """Convert tree into an immutable tree."""
        if self:
            if self.IsAtom():
                return ImmutableTree(self.cargo)
            else:
                tmp = tuple([subtree.ToImmutableTree() for subtree in self.childs])
                return ImmutableTree(self.cargo, *tmp)
        else:
            return ImmutableTree()


class ImmutableTree(AbstractTree):
    """Class implementing an immutable generalized tree type."""

    __slot__ = ('__head', '__parent')

    def __init__(self, cargo=None, *trees):
        """The initializer."""
        if cargo is not None:
            if trees:
                for tree in trees:
                    if not isinstance(tree, ImmutableTree):
                        raise TypeError, "%s is not a ImmutableTree instance." % repr(tree)
                self.__head = (cargo,) + trees
            else:
                self.__head = (cargo,)
        else:
            self.__head = None

    #Properties.
    def __get_cargo(self):
        return self.__head[0]

    cargo = property(__get_cargo, None, None, "The element of the tree")

    def __get_parent(self):
        return self.__parent
    
    parent = property(__get_parent, None, None, "The parent node of the tree")

    def __get_children(self):
        def it(lst):
            for i in xrange(1, len(lst)):
                yield lst[i]

        if self:
            return it(self.__head)
        else:
            #Return empty iterator.
            return iter(())

    children = property(__get_children, None, None, "The iterator over the child subtrees.")

    def make_mutable(self):
        """Convert tree into a mutable tree."""
        if self:
            if self.IsAtom():
                return MutableTree(self.cargo)
            else:
                temp = tuple([subtree.ToMutableTree() for subtree in self.childs])
                return MutableTree(self.cargo, *temp)
        else:
            return MutableTree()


def path_to_root(treenode):
    '''
    Returns the path to the root of the given treenode inclusive itself.
    '''
    if not isinstance(treenode, ITree):
        raise TypeError('treenode must be a subtype of ITree')
    
    path = []
    node = treenode
    while node != None:
        path.append(node)
        node = node.parent
    return path

def path_from_root(treenode):
    '''
    Returns the path from the root of the given treenode inclusive itself.
    '''
    path = path_to_root(treenode)
    path.reverse()
    return path

def shortest_path(source, sink):
    '''
    Compute the shortest path between the source and the sink node inclusive
    themselves.
    '''
    if not isinstance(source, ITree):
        raise TypeError('source must be a subtype of ITree')
    if not isinstance(sink, ITree):
        raise TypeError('sink must be a subtype of ITree')
    
    ancestor = common_ancestor([source, sink])
    path = []
    node = source
    while id(node) != id(ancestor):
        path.append(node)
        node = node.parent
    offset = len(path)
    node = sink
    while id(node) != id(ancestor):
        path.insert(offset, node)
        node = node.parent
    path.insert(offset, ancestor)
    return path
    
def common_ancestor(treenodes):
    '''
    Returns the common ancestor of the given set of treenodes.
    '''
    if not isinstance(treenodes, Container):
        raise TypeError('treenodes must be a container of ITrees')
    
    n = len(treenodes)
    if n == 0:
        raise ValueError('container of treenodes must be nonempty')
    
    ancestor = treenodes[0]
    if n > 1:
        for i in xrange(1,n):
            ancestor = __common_ancestor_pair(ancestor, treenodes[i])
            if ancestor == None:
                raise ValueError('treenodes contains nodes from different trees')
            if ancestor.isroot():
                break
        
    return ancestor

def __common_ancestor_pair(node1, node2):
    path1 = path_from_root(node1)
    path2 = path_from_root(node2)
    
    ancestor = None
    while(len(path1)>0 and len(path2)>0):
        anc1 = path1.pop(0)
        anc2 = path2.pop(0)
        if id(anc1) != id(anc2):
            break
        ancestor = anc1
        
    return ancestor