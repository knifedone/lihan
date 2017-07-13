class Pynote(object):
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


class Tree(object):
    def __init__(self,data):
        length=len(data)
        self.root=Pynote(data[0])
        for i in range(1,length):
            print('init_tree',data[i])
            self.Insert(self.root,data[i])
        print('init done')
    def Insert(self,root,value):
        if root is None:
            root=Pynote(value)
        else:
            if value<root.data:
                root.left=Insert(root.left,value)
            elif value>root.data:
                root.right=Insert(root.right,value)
        return root
    def PreOrder(self):
        def PreOrder_(root):
            if root:
                print(root.data)
                PreOrder(root.left)
                PreOrder(root.right)
            return root
        PreOrder_(self.root)
    def delete(self,val):

        def _delete(root, val):
            if root is None:
                return
            if root.data != val:

                a = query(root, val)
                return _delete(a, val)
            else:
                if (root.left and root.right):
                    temp = find_min(root.right)
                    a = temp.data
                    root.data = a
                    root.right=_delete(root.right, a)

                else:
                    if root.left is None:
                        root = root.right
                    else:
                        root = root.left
                return root
        _delete(self.root, val)

#     def init_tree(self,data):
#         length=len(data)
#         self.root=Pynote(data[0])
#         for i in range(1,length):
#             print('init_tree',data[i])
#             self.Insert(data[i])
#         print('init done')

def Insert(root, value):
    if root is None:
        root = Pynote(value)
    else:
        if value < root.data:
            root.left = Insert(root.left, value)
        elif value > root.data:
            root.right = Insert(root.right, value)
    return root


def init_tree(data):
    length = len(data)
    root = Pynote(data[0])
    for i in range(1, length):
        print('init_tree', data[i])
        Insert(root, data[i])
    print('init done')
    return root


def PreOrder(root):
    if root:
        print(root.data)
        PreOrder(root.left)
        PreOrder(root.right)
    return root


def MidOrder(root):
    if root:
        MidOrder(root.left)
        print(root.data)
        MidOrder(root.right)
    return root


def BackOrder(root):
    if root:
        BackOrder(root.left)
        BackOrder(root.right)
        print(root.data)
    return root


def build(pre, Mid):
    root_data = pre[0]
    node_new = Pynote(root_data)
    index = Mid.index(root_data)
    left_length = index
    right_length = len(Mid) - index - 1
    if left_length > 0:
        node_new.left = build(pre[1:left_length + 1], Mid[0:left_length])
    if right_length > 0:
        node_new.right = build(pre[left_length + 1:right_length + left_length + 1],
                               Mid[index + 1:right_length + index + 1])
    return node_new


def query(root, val):
    if root is None:
        return;
    if root.data is val:
        return root;
    if root.data < val:
        return query(root.right, val);  # 递归地查询
    else:
        return query(root.left, val);


def find_min(root):
    if root.left:
        return find_min(root.left)
    else:
        return root


def find_max(root):
    if root.right:
        return find_max(root.right)
    else:
        return root



# def delete(root, val):
#     if root.data < val:
#         root.right=delete(root.right, val)
#         return root
#     elif root.data > val:
#
#         root.left=delete(root.left, val)
#         return root
#     else:
#         if (root.left and root.right):
#
#             temp = find_min(root.right)
#             a=temp.data
#             root.data = a
#             root.right=delete(root.right, a)
#         else:
#
#             if root.right is None:
#
#                 root = root.left
#
#
#             elif root.left is None:
#
#                 root = root.right
#
#             return root
def delete(root,val):
    if root.data!=val:

        a=query(root,val)
        delete(a,val)
    else:
        if (root.left and root.right):
            temp=find_min(root.right)
            a=temp.data
            root.data=a
            delete(root.right,a)

        elif root.left is None:
            root=root.right
        else:
            root=root.left




a = Tree([4,3,1,2,8,7,16,10,9,14])
a.PreOrder()
a.delete(8)
print('sssssssssss')
a.PreOrder()
