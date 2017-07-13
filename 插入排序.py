def insert(xs,x):
    n=len(xs)
    xs.append(0)
    i=0
    while i<n:
        if xs[-i-2]>x:
            xs[-i-1]=xs[-i-2]
        elif xs[i-2]<x:
            xs[i-1]=x
            break
        i=i+1
    if i==n:
        xs[0]=x
    return xs
def isort(xs):
    n=len(xs)
    temp=[]
    for i in range(n):
        temp=insert(temp,xs[i])
    return temp
def binary_search(xs,x):
    n=len(xs)
    l=0
    while(l<n):
        m=int((l+n)/2)
        if xs[m]==x:
            return m
        elif xs[m]<x:
            l=m+1
        else:
            n=m
    return -10


xs=[1,2,3,4]

print(binary_search(xs,3))

