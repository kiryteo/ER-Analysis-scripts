
def sl_win(ar, n, k):
    csm = 0
    sm = sum(ar[i] for i in range(k))
    csm = max(sm, csm)
    for i in range(k+1, n):
        sm += ar[i]
        sm -= ar[i-k-1]
        if sm > csm:
            csm = sm
    return csm

# ar = [1, 4, 2, 10, 2, 3, 1, 0, 20]
# print(sl_win(ar, 9, 4))
# exit()

# ar = [1, 2, 3, 1, 4, 5, 2, 3, 6]
ar = [8, 5, 10, 7, 9, 4, 15, 12, 90, 13]

def sl_win_max(ar, k):
    # for i in range(len(ar)-k+1):
    #     print(max(ar[i:i+k]))
    p1 = 0
    p2 = k - 1
    while p2 != len(ar):
        if p1 == p2:
            p2 = p2 + k - 1
        print(max(ar[p1], ar[p2]))
        p1 += 1



sl_win_max(ar, 3)
exit()


def perm_check(a, b):
    def create_dt(s):
        dt = {}
        for char in s:
            if char not in dt:
                dt[char] = 1
            else:
                dt[char] += 1
        return dt

    dt1 = create_dt(a)
    dt2 = create_dt(b)

    if dt1 == dt2:
        print('a')
    else:
        print('b')





class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def getSize(self):
        return len(self.items)

    def peek(self):
        if not self.isEmpty():
            return self.items[-1]

    def pop(self):
        return self.items.pop()

    def push(self, item):
        self.items.append(item)


def revstring(mstr):
    s = Stack()
    for char in mstr:
        s.push(char)
    new = ''
    while not s.isEmpty():
        char = s.peek()
        new += char
        s.pop()
    #
    return new


# nums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



# dp[i][j] =
# number of different ways I can get j with nums[0:i+1]

def get_count_expression():
    dp = [[0 for _ in range(2040)] for _ in range(21)]

    dp[0][0] = 1 # we can create 0 target with no numbers


    for i in range(1, n + 1):
        x = nums[i - 1] # next number
        for j in range(-1000, 1000+1):
            if j + x <= 1000:
                dp[i][j + x] += dp[i - 1][j]
            if j - x >= -1000:
                dp[i][j - x] += dp[i - 1][j]

    return dp[n][target]


def bin(n, l):
    s = ""
    for j in range(l):
        bit = (n >> j) & 1
        s += str(bit)
    return s


def count_expr():
    nums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    n = len(nums)
    target = 0

    count = 0

    for i in range(2**n):
        # print(bin(i,n))
        op = bin(i, n)
        # op == 00000
        sm = 0
        for j, bt in enumerate(op):
            if bt == '0':
                sm += (-1)*(nums[j])
            else:
                sm += nums[j]

        if sm == target:
            # print(op)
            count+=1

    return count


class deque:
    def __init__(self):
        self.data = []

    def addFront(self, item):
        self.data.append(item)

    def addRear(self, item):
        self.data.insert(0, item)

    def removeFr(self):
        return self.data.pop()

    def removeRear(self):
        return self.data.pop(0)

    def isEmpty(self):
        return self.data == []

    def size(self):
        return len(self.data)

# print(revstring('sdfsersfdf'))

def palchecker(s):
    d = deque()
    for char in s:
        d.addRear(char)

    count = 0
    orig_size = d.size()
    if d.size() % 2 == 0:
        while not d.isEmpty():
            if d.removeFr() == d.removeRear():
                count += 1
    else:
        while d.size() != 1:
            if d.removeFr() == d.removeRear():
                count += 1
    return count == orig_size // 2


class Node:
    def __init__(self, initdata):
        self.data = initdata
        self.next = None

    def getdata(self):
        return self.data

    def getnext(self):
        return self.next

    def setdata(self, ndata):
        self.data = ndata

    def setnext(self, val):
        self.next = val


class Unolist:
    def __init__(self):
        self.head = None

    def isEmpty(self):
        return self.head is None

    def add(self, item):
        temp = Node(item)
        temp.setnext(self.head)
        self.head = temp

    def size(self):
        temp = self.head
        count = 0
        while temp:
            count += 1
            temp = temp.getnext()
        return count

    def search(self, item):
        temp = self.head
        while temp:
            if temp.getnext() is None and temp.getdata() == item:
                return True
            if temp.getdata() == item:
                return True
            temp = temp.getnext()
        return False

    def remove(self, item):
        # if self.search(item):
        if self.head.getdata() == item:
            print('aa')
            temp = self.head.getnext()
            self.head.setnext(None)
            self.head = temp
            return self.head
        temp = self.head
        while temp.getnext():
            if temp.getnext().getdata() == item:
                temp.setnext(temp.getnext().getnext())
                temp.getnext().setnext(None)
                return self.head
            temp = temp.getnext()
        return False


class node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LL:
    def __init__(self):
        self.head = None

    def traverse(self):
        temp = self.head
        while temp:
            print(temp)
            temp = temp.next

    def insert_fr(self, item):
        temp = node(item)
        temp.next = self.head
        self.head = temp
        return self.head

    def insert_after(self, item, prev_node):
        temp = node(item)
        temp.next = prev_node.next
        prev_node.next = temp

    def insert_end(self, item):
        new_node = node(item)
        new_node.next = None
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next.next = new_node




class Unlist:
    def __init__(self):
        self.head = None

    def isemp(self):
        return self.head.getdata == []

    def add(self, item):
        temp = Node(item)
        temp.setnext(self.head)
        self.head = temp


class ordList:
    def __init__(self):
        self.head = None

    def add(self, item):
        if self.head is None:
            temp = node(item)
            temp.next = None
            self.head = temp
            return self.head
        curr = self.head
        new_node = node(item)
        while curr:
            if curr.next and curr.data < item and curr.next.data > item:
                new_node.next = curr.next
                curr.next = new_node
                return self.head
            curr = curr.next


def getsum(data):
    if len(data) == 1:
        return data[0]
    else:
        return data[0] + getsum(data[1:])


def fac(n):
    if n <= 1:
        return 1
    return n * fac(n-1)


def getstr(n):
    cstr = '0123456789'
    repr = ''
    # repr = []
    while n % 10 != 0:
        rem = n % 10
        # repr.append(rem)
        repr += cstr[rem]
        n = n // 10
    return repr[::-1]

def tostr(n):
    cstr = '0123456789'
    if n < 10:
        return cstr[n]
    else:
        return tostr(n//10) + cstr[n%10]


def binsearch(data, n):
    low = 0
    high = len(data) - 1
    found = False

    while low <= high and not found:
        mid = (low + high) // 2
        if data[mid] == n:
            found = True
        else:
            if n > data[mid]:
                low = mid+1
            else:
                high = mid-1
    return found

# data = [1,4,6,9,13]
# print(binsearch(data, 10))
# exit()

def bs_rec(data, n):
    if len(data) == 0:
        return False
    else:
        mid = len(data) // 2
        if data[mid] == n:
            return True
        else:
            if n < data[mid]:
                return bs_rec(data[:mid], n)
            else:
                return bs_rec(data[mid+1:], n)

# data = [1,4,6,9,13]
# print(bs_rec(data, 8))
# exit()

def bb_sort(data):
    for i in range(len(data)-1):
        for j in range(i+1, len(data)):
            if data[i] > data[j]:
                data[i], data[j] = data[j], data[i]
                # temp = data[i]
                # data[i] = data[j]
                # data[j] = temp
    return data

data = [1,4,9,6,13]
print(bb_sort(data))
exit()


def getbin(num, base):
    # dig = '0123456789ABCDEF'
    dig = '0123456789ABCDEFGHIJKLMNOP'
    # dig = '01234567'
    s = Stack()
    while num > 0:
        rem = num % base
        s.push(rem)
        num = num // base

    binary = ''
    while not s.isEmpty():
        binary += str(dig[s.peek()])
        s.pop()

    return binary


# print(getbin(26, 26))

class Queue:
    def __init__(self):
        self.nums = []

    def enq(self, item):
        self.nums.insert(0, item)

    def deq(self):
        return self.nums.pop()

    def isemp(self):
        return self.nums == []

    def size(self):
        return len(self.nums)


def HotPotato(namelist, num):
    q = Queue()
    for name in namelist:
        q.enq(name)
    while q.size() != 1:
        for _ in range(num):
            name = q.deq()
            q.enq(name)
        q.deq()
    return q.nums




# q = HotPotato(['bil', 'dav', 'jan', 'sus', 'ken', 'brad'], 7)
# print(q)


dt = {0: 0, 1: 1, 2: 2}


def stairs(n):
    if n in dt:
        return dt[n]
    for i in range(3, n + 1):
        dt[i] = dt[i - 1] + dt[i - 2]

    # return dt[n-1] + dt[n-2]
    return dt[n]


# print(stairs(7))
# exit()
import sys

# sys.setrecursionlimit(50000)


def uniqp(m, n):
    if m == 1 and n == 1:
        return 0
    if (m == 2 and n == 1) or (m == 1 and n == 2):
        return 1
    if m == 1:
        return uniqp(m, n-1)
    return uniqp(m-1, n) if n == 1 else uniqp(m-1, n) + uniqp(m, n-1)


print(uniqp(2, 3))
exit()


def dup(n):
    dt = {}
    for val in n:
        if val not in dt:
            dt[val] = 0
        dt[val] += 1

    print(dt)

    for k, v in dt.items():
        if v == 1:
            return k


print(dup([4, 1, 2, 1, 2]))

exit()

import random

alphabet = 'abcdefghijklmnopqrstuvwxyz '


def f1():
    result = ''
    while len(result) != 28:
        char = random.choice(alphabet)
        result += char

    return result


def f2(res, goal):
    num = sum(goal[i] == res[i] for i, char in enumerate(goal))
    return num / len(goal)


def f3(goal):
    mxscore = 0
    res = f1()
    score = f2(res, goal)
    cnt = 0
    best = ''
    while score < 1:
        if score > mxscore:
            best = res
        res = f1()
        score = f2(res, goal)
        if cnt % 1000 == 0:
            print(best, score)
        cnt += 1


f3('methinks it is like a weasel')
