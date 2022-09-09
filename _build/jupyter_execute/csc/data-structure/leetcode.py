#!/usr/bin/env python
# coding: utf-8

# # LeetCode

# ## Code

# In[1]:


from typing import List, Optional
from collections import defaultdict
from heapq import heappush, heappop


# In[ ]:


# 39. Combination Sum

class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        n = len(candidates)
        res = 0
        candidates = sorted(candidates, reverse=True)
        m = candidates[-1]
        # record = {}
        def f(target,index):
            if target==0:
                return [[]]
            if target<m:
                return []
            # if (target,index) in record:
                # return record[(target,index)]
            res = []
            for i in range(index,n):
                tmp = f(target-candidates[i], i)
                if tmp:
                    for l in tmp:
                        res.append([candidates[i]]+l)
            # record[(target,index)] = res
            return res
        return f(target, 0)

# Runtime: 40 ms, faster than 97.71% of Python online submissions for Combination Sum.
# Memory Usage: 13.7 MB, less than 35.79% of Python online submissions for Combination Sum.


# In[ ]:


# 221. Maximal Square

class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        res = 0
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            matrix[i][0] = int(matrix[i][0])
            res = max(matrix[i][0],res)
        for j in range(n):
            matrix[0][j] = int(matrix[0][j])
            res = max(matrix[0][j],res)
        for i in range(1,m):
            for j in range(1,n):
                matrix[i][j] = int(matrix[i][j])
                if int(matrix[i][j]):
                    matrix[i][j] = min([matrix[i][j-1],matrix[i-1][j],matrix[i-1][j-1]])+1
                    res = max(matrix[i][j],res)
        return res**2
                    
# Runtime: 486 ms, faster than 84.34% of Python online submissions for Maximal Square.
# Memory Usage: 27.4 MB, less than 94.13% of Python online submissions for Maximal Square.


# In[1]:


{i:0 for i in range(10)}


# In[ ]:


# 2337. Move Pieces to Obtain a String

class Solution(object):
    def canChange(self, start, target):
        """
        :type start: str
        :type target: str
        :rtype: bool
        """
        j = 0
        n = len(start)
        for i in range(n):
            if start[i] in ('L','R'):
                if j == n:
                    return False
                while target[j]=='_':
                    j += 1
                    if j == n:
                        return False
                if start[i] != target[j]:
                    return False
                if start[i] == 'L' and j>i:
                    return False
                if start[i] == 'R' and j<i:
                    return False
                j += 1
        while j < n:
            if target[j] != '_':
                return False
            j += 1
        return True

# Runtime: 254 ms, faster than 97.34% of Python online submissions for Move Pieces to Obtain a String.
# Memory Usage: 18.3 MB, less than 63.72% of Python online submissions for Move Pieces to Obtain a String.


# In[ ]:


# 299. Bulls and Cows

class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        n = len(secret)
        A = 0
        B = 0
        numDict = {str(i):0 for i in range(10)}
        for i in range(n):
            if secret[i] == guess[i]:
                A += 1
            else:
                if numDict[secret[i]]<0:
                    B += 1
                if numDict[guess[i]]>0:
                    B += 1
                numDict[secret[i]] += 1
                numDict[guess[i]] -= 1
        return str(A)+"A"+str(B)+"B"
        

# Runtime: 32 ms, faster than 85.59% of Python online submissions for Bulls and Cows.
# Memory Usage: 13.4 MB, less than 91.81% of Python online submissions for Bulls and Cows.


# In[ ]:


# 17. Letter Combinations of a Phone Number

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if digits == "":
            return []
        n = 3**len(digits)
        keyboard = {
            '2': ('a', 'b', 'c'),
            '3': ('d', 'e', 'f'),
            '4': ('g', 'h', 'i'),
            '5': ('j', 'k', 'l'),
            '6': ('m', 'n', 'o'),
            '7': ('p', 'q', 'r', 's'),
            '8': ('t', 'u', 'v'),
            '9': ('w', 'x', 'y', 'z')
        }
        res = []
        stack = []
        def combination(digits):
            if digits == '':
                res.append(''.join(stack))
                return
            digit = digits[0]
            for letter in keyboard[digit]:
                stack.append(letter)
                combination(digits[1:])
                stack.pop()
        combination(digits)
        return res
        
# Runtime: 46 ms, faster than 56.75% of Python3 online submissions for Letter Combinations of a Phone Number.
# Memory Usage: 13.9 MB, less than 31.70% of Python3 online submissions for Letter Combinations of a Phone Number.


# In[ ]:


# 13. Roman to Integer

class Solution:
    def romanToInt(self, s: str) -> int:
        mapping = {
            'I':1,
            'V':5,
            'X':10,
            'L':50,
            'C':100,
            'D':500,
            'M':1000
        }
        res = 0
        for i in range(len(s)-1):
            if mapping[s[i]]<mapping[s[i+1]]:
                res += -mapping[s[i]]
            else:
                res += mapping[s[i]]
        res += mapping[s[-1]]
        return res  

# Runtime: 71 ms, faster than 61.14% of Python3 online submissions for Roman to Integer.
# Memory Usage: 13.9 MB, less than 76.66% of Python3 online submissions for Roman to Integer.


# In[ ]:


# 19. Remove Nth Node From End of List

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        m = 1
        cursor = head
        while cursor.next:
            cursor = cursor.next
            m += 1
        
        k = m-n
        if k == 0:
            head = head.next
        else:
            cursor = head
            for i in range(k-1):
                cursor = cursor.next
            cursor.next = cursor.next.next    
        return head
            

# Runtime: 60 ms, faster than 30.47% of Python3 online submissions for Remove Nth Node From End of List.
# Memory Usage: 14 MB, less than 20.39% of Python3 online submissions for Remove Nth Node From End of List.


# In[ ]:


# 134. Gas Station

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        min_val = 0
        min_idx = 0
        cum = 0
        for i in range(n):
            cum += gas[i]-cost[i]
            if cum < min_val:
                min_val = cum
                min_idx = i
        if min_val == 0:
            return 0
        if cum<0:
            return -1
        else:
            return min_idx+1

# Runtime: 702 ms, faster than 91.91% of Python3 online submissions for Gas Station.
# Memory Usage: 19.1 MB, less than 66.67% of Python3 online submissions for Gas Station.


# In[ ]:


# 112. Path Sum

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def dfs(node, target):
            target -= node.val
            isLeaf = True
            result = False
            if node.left:
                isLeaf = False
                result |= dfs(node.left, target)
            if node.right:
                isLeaf = False
                result |= dfs(node.right, target)
            if isLeaf:
                result = target == 0
            return result
        if root:
            return dfs(root, targetSum)
        else:
            return False

# Runtime: 68 ms, faster than 49.17% of Python3 online submissions for Path Sum.
# Memory Usage: 15 MB, less than 93.19% of Python3 online submissions for Path Sum.


# In[ ]:


# 113. Path Sum II

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        result = []
        path = []
        def dfs(node, target):
            path.append(node.val)
            target -= node.val
            isLeaf = True
            if node.left:
                isLeaf = False
                dfs(node.left, target)
            if node.right:
                isLeaf = False
                dfs(node.right, target)
            if isLeaf and target==0:
                print(node.val, target, path)
                result.append(path.copy())
            path.pop()
        if root:
            dfs(root, targetSum)
        return result

# Runtime: 51 ms, faster than 85.61% of Python3 online submissions for Path Sum II.
# Memory Usage: 15.5 MB, less than 73.28% of Python3 online submissions for Path Sum II.


# In[ ]:


# 124. Binary Tree Maximum Path Sum

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        global MAX
        MAX = root.val
        def dfs(node):
            global MAX
            res = node.val
            lmax = 0
            rmax = 0
            if node.left:
                lmax = dfs(node.left)
                res = max(res, lmax+node.val)
            if node.right:
                rmax = dfs(node.right)
                res = max(res, rmax+node.val)
            MAX = max(res, MAX)
            MAX = max(node.val+lmax+rmax, MAX)
            return res
        dfs(root)
        return MAX


# Runtime: 112 ms, faster than 73.04% of Python3 online submissions for Binary Tree Maximum Path Sum.
# Memory Usage: 21.4 MB, less than 30.60% of Python3 online submissions for Binary Tree Maximum Path Sum.


# In[1]:


from heapq import heappush, heappop


# In[ ]:


# 502. IPO

class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        projects = sorted(list(zip(capital, profits)),reverse=True)
        h = []
        for i in range(k):
            while projects and projects[-1][0] <= w:
                heappush(h, -projects.pop()[1])
            if h:
                w -= heappop(h)
            else:
                break
        return w


# Runtime: 1991 ms, faster than 30.63% of Python3 online submissions for IPO.
# Memory Usage: 38.8 MB, less than 33.23% of Python3 online submissions for IPO.


# In[ ]:


# 338. Counting Bits

class Solution:
    def countBits(self, n: int) -> List[int]:
        res = [0 for _ in range(n+1)]
        for i in range(1,n+1):
            if i&(i-1) == 0:
                ptr = 0
            res[i] = 1+res[ptr]
            ptr += 1
        return res

# Runtime: 156 ms, faster than 43.56% of Python3 online submissions for Counting Bits.
# Memory Usage: 20.8 MB, less than 79.84% of Python3 online submissions for Counting Bits.


# In[ ]:


# 71. Simplify Path

class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        path += '/'
        n = len(path)
        start = 0
        end = 0
        i = 0
        while i<n:
            if path[i] == '/':
                if end>start:
                    if start+2 >= end:
                        tmp = path[start:end]
                        if tmp == '.':
                            pass
                        elif tmp == '..':
                            if stack:
                                stack.pop()
                        else:
                            stack.append((start,end))
                    else:
                        stack.append((start,end))
                start = i+1
                end = i+1
            else:
                end += 1
            i += 1
        result = ''
        for folder in stack:
            result += '/'+path[folder[0]:folder[1]]
        if result:
            return result
        else:
            return '/'

# Runtime: 54 ms, faster than 48.25% of Python3 online submissions for Simplify Path.
# Memory Usage: 14 MB, less than 41.53% of Python3 online submissions for Simplify Path.


# In[ ]:


# 2381. Shifting Letters II

class Node:
    def __init__(self, l, r):
        self.left = None
        self.right = None
        self.val = 0
        self.l = l
        self.r = r
        
    def construct(self):
        if self.l < self.r:        
            mid = (self.l+self.r) // 2
            self.left = self.__class__(self.l, mid)
            self.right = self.__class__(mid+1, self.r)
            self.left.construct()
            self.right.construct()
        
    def update(self, l, r, val):
        if self.r < l or self.l > r:
            return    
        elif self.l >= l and self.r<=r: #fully included
            self.val += val
        else:
            self.left.update(l, r, val)
            self.right.update(l, r, val)
            
    def traverse(self, shift=0):
        shift += self.val
        if self.left:
            return self.left.traverse(shift)+self.right.traverse(shift)            
        else:
            return [shift]

class Solution:
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        root = Node(0,len(s)-1)
        root.construct()
        for shift in shifts:
            root.update(shift[0],shift[1],2*shift[2]-1)
        alphabet = root.traverse()
        for i in range(len(alphabet)):
            alphabet[i] = chr(((alphabet[i]+ord(s[i])-97)%26)+97)
        return ''.join(alphabet)


# Runtime: 5630 ms, faster than 20.00% of Python3 online submissions for Shifting Letters II.
# Memory Usage: 43.7 MB, less than 40.00% of Python3 online submissions for Shifting Letters II.


# In[ ]:


# 2351. First Letter to Appear Twice

class Solution:
    def repeatedCharacter(self, s: str) -> str:
        d = dict()
        for c in s:
            d[c] = d.get(c,0)+1
            if d[c] == 2:
                return c

# Runtime: 36 ms, faster than 83.58% of Python3 online submissions for First Letter to Appear Twice.
# Memory Usage: 13.8 MB, less than 96.16% of Python3 online submissions for First Letter to Appear Twice.


# In[ ]:


# 222. Count Complete Tree Nodes

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        global res
        res = 0
        def count(node):
            global res
            res += 1
            if node.left:
                count(node.left)
            if node.right:
                count(node.right)
        if root:
            count(root)
        return res


# Runtime: 99 ms, faster than 78.58% of Python3 online submissions for Count Complete Tree Nodes.
# Memory Usage: 21.4 MB, less than 47.73% of Python3 online submissions for Count Complete Tree Nodes.


# In[ ]:


# 399. Evaluate Division

class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # def clean(A,B):
        #     global flag
        #     flag = True
        #     global graph
        #     d = dict()
        #     for char in A:
        #         if char in d:
        #             d[char] += 1
        #         else:
        #             d[char] = 1
        #         if char not in graph.keys():
        #             flag = False
        #     for char in B:
        #         if char in d:
        #             d[char] -= 1
        #         else:
        #             d[char] = -1
        #         if char not in graph.keys():
        #             flag = False
        #     a = ''
        #     b = ''
        #     for char in d:
        #         if d[char]>0:
        #             a += char*d[char]
        #         elif d[char]<0:
        #             b += char*(-d[char])
        #     return a, b
        
        
        def query(a, b):
            # global flag
            # a, b = clean(A, B)
            # if not flag:
            #     return -1
            global graph
            if a not in graph.keys() or b not in graph.keys():
                return -1.
            if a == b:
                return 1.
            res = -1.
            for k, v in graph[a].items():
                if k not in visited:
                    visited.add(k)
                    tmp = query(k, b)
                    if tmp > 0:
                        res = v*tmp
            return res
                    
        global visited
        global graph
        graph = dict()
        for i, (a, b) in enumerate(equations):
            # a, b = clean(A, B)
            if a in graph:
                graph[a][b] = values[i]
            else:
                graph[a] = {b : values[i]}
            if b in graph:
                graph[b][a] = 1. / values[i]
            else:
                graph[b] = {a: 1/values[i]}
        res = []
        for a, b in queries:
            visited = set()
            res.append(query(a,b))
        return res 

# Runtime: 40 ms, faster than 74.58% of Python3 online submissions for Evaluate Division.
# Memory Usage: 14 MB, less than 17.43% of Python3 online submissions for Evaluate Division.


# In[ ]:


# 2352. Equal Row and Column Pairs

class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        d = dict()
        for row in grid:
            tmp = tuple(row)
            d[tmp] = d.get(tmp, 0)+1
        res = 0
        for col in range(len(grid)):
            res += d.get(tuple(grid[i][col] for i in range(len(grid))),0)
        return res

# class Solution {
# public:
#     int equalPairs(vector<vector<int>>& grid) {
#         int n = grid.size();
#         cout << n << endl;
#         for (int k=0; k<n; k++) {
#             for (int i=0; i<n; i++) {
#                 for (int j=0; j<n; j++) {
#                     if (abs(grid[i][k]) != abs(grid[k][j])) {
#                         grid[i][j] = -abs(grid[i][j]);
#                     }
#                 }
#             }
#         }
#         int res = 0;
#         for (int i=0; i<n; i++) {
#             for (int j=0; j<n; j++) {
#                 if (grid[i][j]>0) {
#                     res++;
#                 }
#             }
#         }
#         return res;
#     }
# };

# Runtime: 739 ms, faster than 67.85% of Python3 online submissions for Equal Row and Column Pairs.
# Memory Usage: 18.9 MB, less than 76.68% of Python3 online submissions for Equal Row and Column Pairs.


# In[ ]:


# 2376. Count Special Integers

class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        d = {1:0, 2:9, 3:90, 4:738, 5:5274, 6:32490, 7:168570, 8:712890, 9:2345850, 10:5611770}
        nums = list(map(int,[*str(n)]))
        n = len(nums)
        res = 0
        subs = [0 for _ in range(n)]
        subs[0] -= 1
        flag = 0
        for i in range(n):
            if flag and i==flag:
                break
            for j in range(i+1, n):
                subs[j] -= (nums[i]<nums[j])
                if nums[i] == nums[j]:
                    flag = j+1
                    for k in range(j+1,n):
                        subs[k] = -nums[k]
                    break
        multiplier = 10
        for i in range(n):
            base = max(nums[i]+subs[i],0)
            res = res*multiplier+base
            multiplier -= 1
        res += d[n]
        res += len(set(nums))==len(nums)
        return res
        

# Runtime: 50 ms, faster than 63.08% of Python3 online submissions for Count Special Integers.
# Memory Usage: 13.9 MB, less than 54.61% of Python3 online submissions for Count Special Integers.


# In[ ]:


# 357. Count Numbers with Unique Digits

class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        d = {1:0, 2:9, 3:90, 4:738, 5:5274, 6:32490, 7:168570, 8:712890, 9:2345850, 10:5611770}
        return d[n+1]+1


# Runtime: 39 ms, faster than 73.97% of Python3 online submissions for Count Numbers with Unique Digits.
# Memory Usage: 13.8 MB, less than 57.45% of Python3 online submissions for Count Numbers with Unique Digits.


# In[ ]:


# 2332. The Latest Time to Catch a Bus

class Solution:
    def latestTimeCatchTheBus(self, buses: List[int], passengers: List[int], capacity: int) -> int:
        passengers = sorted(passengers)
        buses = sorted(buses)
        q = deque([])
        k = 0
        m = len(buses)
        i = 0
        n = len(passengers)
        while True:
            while (i<n) and (passengers[i] <= buses[k]):
                q.append(passengers[i])
                i+=1
            if k == m-1:
                break
            cnt = 0
            while q and cnt< capacity:
                q.popleft()
                cnt += 1
            k += 1
        if not q:
            return buses[-1]
        res = -1
        for j in range(1,capacity):
            if j == len(q):
                if q[-1]<buses[-1]:
                    return buses[-1]
                break                
            if q[j]>q[j-1]+1:
                res = q[j]-1
        if res > 0:
            return res
        i -= len(q)
        while passengers[i-1]+1==passengers[i]:
            i -= 1
        return passengers[i]-1


# Runtime: 764 ms, faster than 95.66% of Python3 online submissions for The Latest Time to Catch a Bus.
# Memory Usage: 32.9 MB, less than 83.68% of Python3 online submissions for The Latest Time to Catch a Bus.


# In[ ]:


# 2335. Minimum Amount of Time to Fill Cups

class Solution:
    def fillCups(self, amount: List[int]) -> int:
        amount = sorted(amount)
        if amount[0]+amount[1]<=amount[2]:
            return amount[2]
        else:
            return sum(amount)//2+(amount[0]+amount[1]-amount[2])%2

# Runtime: 28 ms, faster than 97.28% of Python3 online submissions for Minimum Amount of Time to Fill Cups.
# Memory Usage: 13.9 MB, less than 55.89% of Python3 online submissions for Minimum Amount of Time to Fill Cups.


# In[ ]:


# 2270. Number of Ways to Split Array

class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        S = sum(nums)
        res = 0+(nums[0]*2>=S)
        for i in range(1,len(nums)-1):
            nums[i] += nums[i-1]
            if nums[i]*2>=S:
                res += 1
        return res

# Runtime: 1055 ms, faster than 84.28% of Python3 online submissions for Number of Ways to Split Array.
# Memory Usage: 30 MB, less than 7.29% of Python3 online submissions for Number of Ways to Split Array.


# In[ ]:


# 2126. Destroying Asteroids

class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        asteroids = sorted(asteroids)
        res = asteroids[0] <= mass
        mass += asteroids[0]
        for i in range(1, len(asteroids)):
            res &= asteroids[i] <= mass
            mass += asteroids[i]
        return res

# Runtime: 1778 ms, faster than 47.92% of Python3 online submissions for Destroying Asteroids.
# Memory Usage: 27.9 MB, less than 43.41% of Python3 online submissions for Destroying Asteroids.


# In[ ]:


# 306. Additive Number

class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        def check(x, y, idx, num):
            if idx == len(num):
                return True
            s = x+y
            if (num[idx]=='0' and s==0) or (num[idx]!='0' and s == int(num[idx:idx+len(str(s))])):
                return check(y, s, idx+len(str(s)), num)
            else:
                return False

        for i in range(1,len(num)//2+1):
            if num[0]=='0' and i>1:
                continue
            x = int(num[:i])
            for j in range(1,len(num)//2+1):
                if j>1 and num[i]=='0':
                    break
                y = int(num[i:i+j]) 
                s = x+y
                k = len(str(s))
                if i+j+k<=len(num) and s == int(num[i+j:i+j+k]):
                    if check(x, y, i+j, num):
                        return True
        return False

# Runtime: 46 ms, faster than 60.31% of Python3 online submissions for Additive Number.
# Memory Usage: 14 MB, less than 30.16% of Python3 online submissions for Additive Number.


# In[ ]:


# 326. Power of Three

class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n==0:
            return False
        return True if n==1 else (n%3==0) and (self.isPowerOfThree(n//3))

# Runtime: 78 ms, faster than 95.72% of Python3 online submissions for Power of Three.
# Memory Usage: 13.8 MB, less than 96.78% of Python3 online submissions for Power of Three.


# In[ ]:


# 2116. Check if a Parentheses String Can Be Valid

class Solution:
    def canBeValid(self, s: str, locked: str) -> bool:
        n = len(s)
        if n%2:
            return False
        if n==0:
            return True
        stack = []
        tmp = 0
        for i in range(n):
            if locked[i] == '0':
                tmp += 1
                # print('*',stack, tmp)
            else:
                if s[i] == '(':
                    stack.append(tmp)
                    tmp = 0
                else:
                    if not stack:
                        tmp -= 1
                        if tmp < 0:
                            return False
                    else:
                        out = stack.pop()
                        tmp += out
                # print(s[i],stack, tmp)
        # print(stack, tmp)
        for i in range(len(stack)-1,-1,-1):
            tmp -= 1
            if tmp<0:
                return False
            tmp += stack[i]
        return True
                    

# Runtime: 232 ms, faster than 88.45% of Python3 online submissions for Check if a Parentheses String Can Be Valid.
# Memory Usage: 15.4 MB, less than 86.61% of Python3 online submissions for Check if a Parentheses String Can Be Valid.


# In[ ]:


# 278. First Bad Version

# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        def binary(l,r):
            if l+1 == r:
                if isBadVersion(l):
                    return l
                else:
                    return r 
            mid = (l+r)//2
            if isBadVersion(mid):
                if isBadVersion(mid-1):
                    return binary(l,mid-1)
                else:
                    return mid
            else:
                if isBadVersion(mid+1):
                    return mid+1
                else:
                    return binary(mid+1,r)
        return binary(1,n)

# Runtime: 33 ms, faster than 88.48% of Python3 online submissions for First Bad Version.
# Memory Usage: 13.9 MB, less than 62.49% of Python3 online submissions for First Bad Version.


# In[ ]:


# 419. Battleships in a Board

class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        m = len(board)
        n = len(board[0])
        cnt = 0
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'X':
                    cnt += 1
                    board[i][j] = '.'
                    tmp = j
                    while tmp+1<n and board[i][tmp+1] == 'X':
                        tmp += 1
                        board[i][tmp] = '.'
                    tmp = i
                    while tmp+1<m and board[tmp+1][j] == 'X':
                        tmp += 1
                        board[tmp][j] = '.'
        return cnt


# Runtime: 96 ms, faster than 72.53% of Python3 online submissions for Battleships in a Board.
# Memory Usage: 14.5 MB, less than 67.70% of Python3 online submissions for Battleships in a Board.


# In[9]:


# IMC OA

from heapq import heappop, heappush

def schedule(arrival, lane):
    q = []
    t = 0
    i = 0
    n = len(arrival)
    res = [0 for i in range(n)]
    while True:
        while i<n and arrival[i] == t:
            heappush(q, (1-lane[i], t, i))
            i += 1
        if i == n:
            break 
        while t<arrival[i]:
            if q:
                top = heappop(q)
                res[top[-1]] = t
            t += 1
    while q:
        top = heappop(q)
        res[top[-1]] = t
        t += 1
    return res

schedule([0,0,1,4],[0,1,1,0])


# In[18]:


# circular segement tree

class Node:
    def __init__(self, l, r, arr):
        self.val = 0
        self.left = None
        self.right = None
        self.lpos = l
        self.rpos = r
        self.leaf = False
        self.construct(arr)
    
    def construct(self, arr):
        if self.lpos == self.rpos:
            self.leaf = True
            self.val = arr[self.lpos]
        else:
            mid = (self.lpos + self.rpos) // 2
            self.left = Node(self.lpos, mid, arr)
            self.right = Node(mid+1, self.rpos, arr)
    
    def display(self):
        if self.leaf:
            return self.val
        else:
            return [self.val,[self.left.display(), self.right.display()]]

    def update(self,l,r,inc):
        # print(l,r,inc)
        assert r >= -1
        if r>=0 and l>=r:
            if l == r+1:
                self.val += inc
            else:
                self.update(0,r,inc)
                self.update(l,-1,inc)
        else:
            # fully included: l<=self.lpos<=self.rpos<=r
            if (l <= self.lpos) and ((self.rpos <= r) or (r == -1)):
                self.val += inc
            # not included
            elif ((r>=0) and (r<self.lpos)) or (l>self.rpos):
                pass
            else:
                self.left.update(l,r,inc)
                self.right.update(l,r,inc)


def circular_segment_tree(arr):
    root = Node(0,len(arr)-1, arr)
    print(root.display())
    root.update(2,1,3)
    print(root.display())
    
circular_segment_tree([1,2,3,4,5])


#           0
#     0        0
#  0     3    4 5
# 1 2      


# In[1]:


# 2136. Earliest Possible Day of Full Bloom

class Solution:
    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        pairs = sorted(list(zip(growTime, plantTime)), reverse=True)
        # print(pairs)
        presum = 0
        res = 0
        for i, (g, p) in enumerate(pairs):
            # print(presum, p, g)
            res = max(res, presum+g+p)
            presum += p

# Runtime: 1753 ms, faster than 96.50% of Python3 online submissions for Earliest Possible Day of Full Bloom.
# Memory Usage: 31.7 MB, less than 59.44% of Python3 online submissions for Earliest Possible Day of Full Bloom.


# In[ ]:


# 486. Predict the Winner

class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        @cache
        def dp(i, j, p):
            """
            p = 1 if (player 1) else -1
            """
            if i>j:
                return 0
            if p>0:
                return max(nums[i]*p+dp(i+1,j,-p), nums[j]*p+dp(i,j-1,-p))
            else:
                return min(nums[i]*p+dp(i+1,j,-p), nums[j]*p+dp(i,j-1,-p))
        return dp(0,len(nums)-1,1)>=0


# Runtime: 31 ms, faster than 98.73% of Python3 online submissions for Predict the Winner.
# Memory Usage: 14.2 MB, less than 31.22% of Python3 online submissions for Predict the Winner.


# In[ ]:


# 463. Island Perimeter

class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        row = len(grid)
        col = len(grid[0])
        cnt = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == 1:
                    neighbor = 0
                    if i>0:
                        neighbor += (grid[i-1][j]==1)
                    if j>0:
                        neighbor += (grid[i][j-1]==1)
                    if i<row-1:
                        neighbor += (grid[i+1][j]==1)
                    if j<col-1:
                        neighbor += (grid[i][j+1]==1)
                    cnt += 4-neighbor
        return cnt

# Runtime: 964 ms, faster than 37.18% of Python3 online submissions for Island Perimeter.
# Memory Usage: 14.3 MB, less than 72.53% of Python3 online submissions for Island Perimeter.


# In[ ]:


# 565. Array Nesting

class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        n = len(nums)
        res = 1
        i = 0
        while True:
            if nums[i]<0:
                if i < n-1:
                    i += 1
                else:
                    return res
            else:
                tmp = 1
                if nums[i] != i:
                    k = i
                    while nums[nums[k]] >= 0:
                        # print(nums[k])
                        tmp += 1
                        nums[k] = -nums[k]-1
                        k = -nums[k]-1
                    res = max(tmp, res)
                if i < n-1:
                    i += 1
                else:
                    return res
                    
# Runtime: 1369 ms, faster than 59.70% of Python3 online submissions for Array Nesting.
# Memory Usage: 28 MB, less than 69.59% of Python3 online submissions for Array Nesting.


# In[ ]:


# 554. Brick Wall

class Solution:
    def leastBricks(self, wall: List[List[int]]) -> int:
        d = {}
        row = len(wall)
        for i in range(row):
            tmp = 0
            for j in range(len(wall[i])-1):
                tmp += wall[i][j]
                # print(tmp, end=' ')
                d[tmp] = d.get(tmp,0) + 1
            # print()
        # print(d)
        if not d:
            return row
        else:
            return row-sorted(d.items(), key=lambda x: x[1])[-1][1]

# Runtime: 432 ms, faster than 5.92% of Python3 online submissions for Brick Wall.
# Memory Usage: 19.1 MB, less than 42.60% of Python3 online submissions for Brick Wall.


# In[ ]:


# 2216. Minimum Deletions to Make Array Beautiful

class Solution:
    def minDeletion(self, nums: List[int]) -> int:
        # from left to right scan until 
        n = len(nums)
        i = 0
        k = 1
        res = 0
        while True:
            if k >= n:
                # print(1)
                # print(i,k)
                return res + (n-res)%2
            if nums[i] != nums[k]:
                # print(i,k)
                i = k+1
                k = i+1
            if k >= n:
                # print(2)
                # print(i,k)
                return res+ (n-res)%2
            if nums[i] == nums[k]:
                res += 1
                k += 1

# Runtime: 1607 ms, faster than 83.17% of Python3 online submissions for Minimum Deletions to Make Array Beautiful.
# Memory Usage: 28 MB, less than 84.51% of Python3 online submissions for Minimum Deletions to Make Array Beautiful.


# In[ ]:


# 456. 132 Pattern

class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        n = len(nums)
        pairs = []
        tmp_m = nums[0]
        tmp_M = nums[0]
        for i in range(1,n):
            # print(i)
            # print(pairs)
            if tmp_m<nums[i]<tmp_M:
                return True
            for pair in pairs:
                    if pair[0]<nums[i]<pair[1]:
                        return True
            if nums[i]<tmp_m:
                if tmp_M>tmp_m:
                    while pairs and tmp_M>pairs[-1][1] and tmp_m<pairs[-1][0]:
                        pairs.pop()
                    pairs.append((tmp_m,tmp_M))
                tmp_m = nums[i]
                tmp_M = nums[i]
            tmp_M = max(tmp_M,nums[i])
            # print(tmp_m, tmp_M)
        return False

# Runtime: 1601 ms, faster than 5.11% of Python3 online submissions for 132 Pattern.
# Memory Usage: 32.2 MB, less than 16.14% of Python3 online submissions for 132 Pattern.


# In[ ]:


# 2390. Removing Stars From a String

class Solution:
    def removeStars(self, s: str) -> str:
        stack = []
        for letter in s:
            if letter == '*':
                stack.pop()
            else:
                stack.append(letter)
        return ''.join(stack)

# Runtime: 239 ms, faster than 94.27% of Python3 online submissions for Removing Stars From a String.
# Memory Usage: 15.8 MB, less than 13.78% of Python3 online submissions for Removing Stars From a String.


# In[ ]:


# 401. Binary Watch

class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        h = {0: ['0'], 1: ['1', '2', '4', '8'], 2: ['3', '5', '6', '9', '10'], 3: ['7', '11']}
        m = {0: ['00'], 
             1: ['01', '02', '04', '08', '16', '32'], 
             2: ['03', '05', '06', '09', '10', '12', '17', '18', '20', '24', '33', '34', '36', '40', '48'], 
             3: ['07', '11', '13', '14', '19', '21', '22', '25', '26', '28', '35', '37', '38', '41', '42', '44', '49', '50', '52', '56'], 
             4: ['15', '23', '27', '29', '30', '39', '43', '45', '46', '51', '53', '54', '57', '58'], 
             5: ['31', '47', '55', '59']}
        res = []
        for i in range(turnedOn+1):
            if i in h and turnedOn-i in m:
                for hr in h[i]:
                    for mi in m[turnedOn-i]:
                        res.append(f"{hr}:{mi}")
        return res

# Runtime: 53 ms, faster than 53.17% of Python3 online submissions for Binary Watch.
# Memory Usage: 13.8 MB, less than 73.05% of Python3 online submissions for Binary Watch.


# In[ ]:


# 447. Number of Boomerangs

class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        n = len(points)
        res = 0
        dist = {i: defaultdict(list) for i in range(n)}
        for i in range(n):
            for j in range(i+1, n):
                d = (points[i][0]-points[j][0])**2+(points[i][1]-points[j][1])**2
                res += len(dist[i][d])+len(dist[j][d])
                dist[i][d].append(j)
                dist[j][d].append(i)
        return res*2


# Runtime: 3212 ms, faster than 12.02% of Python3 online submissions for Number of Boomerangs.
# Memory Usage: 53.7 MB, less than 12.01% of Python3 online submissions for Number of Boomerangs.


class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        n = len(points)
        res = 0
        dist = {i: defaultdict(int) for i in range(n)}
        for i in range(n):
            for j in range(i+1, n):
                d = (points[i][0]-points[j][0])**2+(points[i][1]-points[j][1])**2
                res += dist[i][d]+dist[j][d]
                dist[i][d] += 1
                dist[j][d] += 1
        return res*2

# Runtime: 1879 ms, faster than 53.36% of Python3 online submissions for Number of Boomerangs.
# Memory Usage: 27.3 MB, less than 29.05% of Python3 online submissions for Number of Boomerangs.


# In[ ]:


# 309. Best Time to Buy and Sell Stock with Cooldown

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        def dp(a, b, flag, prices, buy, sell):
            # print(a, b,  flag)
            res = 0
            if a == b:
                res = 0
            else:
                if flag:
                    tmp = min(prices[sell[a]]-prices[buy[a+1]], prices[sell[a]]-prices[sell[a]-1])
                    return min(
                        tmp+dp(a+1, b, 1, prices, buy, sell),
                        prices[buy[a+1]+1]-prices[buy[a+1]]+dp(a+1, b, 0, prices, buy, sell)
                    )
                else:
                    res = dp(a+1, b, 1, prices, buy, sell)
            # print(res)
            return res
        
        n = prices
        buy = [0]
        sell = []
        bought = True
        for i, price in enumerate(prices):
            if i == 0:
                continue
            # print(i, price)
            if price<=prices[i-1]:
                if bought:
                    # print('update buy')
                    buy[-1] = i
                else:
                    bought = True
                    # print('buy')
                    buy.append(i)
            else:
                if bought:
                    bought = False
                    # print('sell')
                    sell.append(i)
                else:
                    # print('update sell')
                    sell[-1] = i
        
        if len(buy) > len(sell):
            buy.pop()
        n = len(buy)

        if not n:
            return 0
        print(buy)
        print(sell)


        res = prices[sell[0]]-prices[buy[0]] 
        recording = False
        for i in range(n-1):
            res += prices[sell[i+1]] - prices[buy[i+1]] 
            if buy[i+1] == sell[i]+1 == sell[i+1]-1:
                print(1 )
                if recording:
                    b = i+1
                else:
                    recording = True
                    a = i
                    b = i+1
            else:
                if recording:                    
                    recording = False
                    if buy[i+1] == sell[i]+1:
                        b += 1
                    
                    res -= dp(a,b,1,prices,buy,sell)
                        
                else:
                    res -= 0 if (buy[i+1] != sell[i]+1) else min(
                        prices[sell[i]]-prices[buy[i+1]],
                        min(
                            prices[sell[i]]-prices[sell[i]-1],
                            prices[buy[i+1]+1]-prices[buy[i+1]]
                        )
                    )
                    
        if recording:
            print(a,b)
            res -= dp(a,b,1,prices,buy,sell)
        return res

# Runtime: 38 ms, faster than 98.44% of Python3 online submissions for Best Time to Buy and Sell Stock with Cooldown.
# Memory Usage: 14.3 MB, less than 67.96% of Python3 online submissions for Best Time to Buy and Sell Stock with Cooldown.


# In[ ]:


# 1717. Maximum Score From Removing Substrings

class Solution:
    def maximumGain(self, s: str, x: int, y: int) -> int:
        def greedy(s):
            res = 0
            count = 0
            stage = 0
            if x>y:
                large, small = x, y
                upper, lower = 'a', 'b'
            else:
                large, small = y, x
                upper, lower = 'b', 'a'
                
            for l in s:
                if l == upper:
                    if count < 0:
                        stage += -count
                        count = 0
                    count += 1
                else:
                    count -= 1
                    if count >= 0:
                        res += large
            res += small*min(max(count,0),stage)
            return res
              
        res = 0
        recording = False
        for i in range(len(s)):
            if s[i] in ('a','b'):
                if recording:
                    b += 1
                else:
                    recording = True
                    a = i
                    b = i+1
            else:
                if recording:
                    res += greedy(s[a:b])
                    
                    recording = False
        if recording:
            res += greedy(s[a:b])
        return res


# Runtime: 1283 ms, faster than 32.32% of Python3 online submissions for Maximum Score From Removing Substrings.
# Memory Usage: 15 MB, less than 66.67% of Python3 online submissions for Maximum Score From Removing Substrings.


# In[ ]:


# 1658. Minimum Operations to Reduce X to Zero

class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        nums = [0] + nums + [0]
        n = len(nums)
        leftpre = nums.copy()
        rightpre = nums.copy()
        
        for i in range(1,n):
            leftpre[i] += leftpre[i-1]
            rightpre[-i-1] += rightpre[-i]
        # print(leftpre)
        # print(rightpre)
        if leftpre[-1]<x:
            return -1
        
        i = 1
        j = n-1
        res = n+1
        while rightpre[j] < x:
            j -= 1
        if rightpre[j] == x:
            res = n-j+1
        # print(i,j)
        # print(res)
        while i < res-1:
            if leftpre[i] == x:
                # print(i)
                res = min(res, i+2)
                break
            if leftpre[i] > x:
                break
            while leftpre[i]+rightpre[j]>x:
                
                j += 1
            # print(j)
            if leftpre[i] + rightpre[j] == x:
                res = min(res, i+1+n-j)
            else: j -= 1
            i += 1

        if res > n:
            return -1
        else:
            return res-2


# Runtime: 1967 ms, faster than 43.46% of Python3 online submissions for Minimum Operations to Reduce X to Zero.
# Memory Usage: 28.3 MB, less than 31.94% of Python3 online submissions for Minimum Operations to Reduce X to Zero.


# In[ ]:


# 2215. Find the Difference of Two Arrays

class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        s1 = set(nums1)
        s2 = set(nums2)
        res = []
        for s in s2:
            if s in s1:
                s1.remove(s)
            else:
                res.append(s)
        return [list(s1),res]

# Runtime: 206 ms, faster than 88.47% of Python3 online submissions for Find the Difference of Two Arrays.
# Memory Usage: 14.3 MB, less than 80.92% of Python3 online submissions for Find the Difference of Two Arrays.


# In[ ]:


# 1425. Constrained Subsequence Sum

class Solution:
    def constrainedSubsetSum(self, nums: List[int], k: int) -> int:

        n = len(nums)
        
        @cache
        def dp(start):
            if start>=len(nums):
                return 0
            res = nums[start]
            for i in range(k):
                if start+i < n:
                    res = max(res, nums[start]+dp(start+i+1))
            print(start, res)
            return res
        
        res = dp(0)
        i = 1
        while i < n:
            if nums[i]>0:
                res = max(res, dp(i))
            i+=1
        return res

class Solution:
    def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        tmp = nums.copy()
        for i in range(n):
            for j in range(min(i,k)):
                tmp[i] = max(tmp[i], nums[i]+tmp[i-j-1])
        return max(tmp)

class Solution:
    def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
        
        n = len(nums)
        
        res = nums[0]
        
        h = [(-nums[0],0)]
        
        for i in range(1,n):
            while h[0][1]+k<i:
                heappop(h)
            tmp = max(nums[i], nums[i]-h[0][0])
            res = max(res, tmp)
            heappush(h, (-tmp,i))

        return res


# Runtime: 4278 ms, faster than 8.84% of Python3 online submissions for Constrained Subsequence Sum.
# Memory Usage: 34.1 MB, less than 18.58% of Python3 online submissions for Constrained Subsequence Sum.


# In[ ]:


# 1732. Find the Highest Altitude

class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        tmp = gain[0]
        res = 0
        n = len(gain)
        for i in range(1,n):
            res = max(res, tmp)
            tmp += gain[i]
        res = max(res, tmp)
        return res

# Runtime: 42 ms, faster than 80.87% of Python3 online submissions for Find the Highest Altitude.
# Memory Usage: 13.9 MB, less than 51.70% of Python3 online submissions for Find the Highest Altitude.


# In[ ]:


# 1995. Count Special Quadruplets

class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            for j in range(i+1,n):
                for k in range(j+1,n):
                    for l in range(k+1,n):
                        if nums[i]+nums[j]+nums[k] == nums[l]:
                            res += 1
        
        return res

# Runtime: 1416 ms, faster than 55.61% of Python3 online submissions for Count Special Quadruplets.
# Memory Usage: 13.8 MB, less than 63.10% of Python3 online submissions for Count Special Quadruplets.

class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        count = defaultdict(lambda: 0)
        count[nums[-1]-nums[-2]]=1
        for b in range(n-3,0,-1):
            for a in range(b-1, -1, -1):
                res += count[nums[a]+nums[b]]
            for d in range(n-1,b,-1):
                count[nums[d]-nums[b]] += 1
        return res

# Runtime: 123 ms, faster than 83.16% of Python3 online submissions for Count Special Quadruplets.
# Memory Usage: 14 MB, less than 19.56% of Python3 online submissions for Count Special Quadruplets.


# In[ ]:


# 1979. Find Greatest Common Divisor of Array

class Solution:
    def findGCD(self, nums: List[int]) -> int:
        
        m = min(nums)
        M = max(nums)
 
        for i in range(m,0,-1):
            if m%i == 0 and M%i == 0:
                return i


# Runtime: 119 ms, faster than 13.06% of Python3 online submissions for Find Greatest Common Divisor of Array.
# Memory Usage: 14 MB, less than 81.44% of Python3 online submissions for Find Greatest Common Divisor of Array.


class Solution:
    def findGCD(self, nums: List[int]) -> int:

        m = max(nums)
        
        d = min(nums)
        r = m%d
        while r>0:
            m = d
            d = r
            r = m%d
        return d
            

# Runtime: 112 ms, faster than 19.41% of Python3 online submissions for Find Greatest Common Divisor of Array.
# Memory Usage: 14 MB, less than 81.44% of Python3 online submissions for Find Greatest Common Divisor of Array.


# In[ ]:


# 1551. Minimum Operations to Make Array Equal

class Solution:
    def minOperations(self, n: int) -> int:
        res = 0
        for i in range(1+n%2,n,2):
            res += i
        return res
    
class Solution:
    def minOperations(self, n: int) -> int:
        return (1+n%2+(n-1))*(n//2)//2


# Runtime: 46 ms, faster than 79.72% of Python3 online submissions for Minimum Operations to Make Array Equal.
# Memory Usage: 13.9 MB, less than 33.16% of Python3 online submissions for Minimum Operations to Make Array Equal.


# In[ ]:


# 1545. Find Kth Bit in Nth Binary String

class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        
        def f(k,l):
            if l == 1:
                return 0
            mid = (l+1)//2
            if k == mid:
                return 1
            elif k<mid:
                return f(k,(l-1)//2)
            else:
                return 1-f(2*mid-k, (l-1)//2)
                
        return str(f(k,2**n-1))

# Runtime: 32 ms, faster than 96.48% of Python3 online submissions for Find Kth Bit in Nth Binary String.
# Memory Usage: 13.9 MB, less than 92.96% of Python3 online submissions for Find Kth Bit in Nth Binary String.


# In[ ]:


# 695. Max Area of Island

class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        
        def dfs(i,j,n,m):
            global tmp
            if grid[i][j] == 1:
                tmp += 1
                grid[i][j] = -1
                if i > 0:
                    dfs(i-1,j,n,m)
                if j > 0:
                    dfs(i,j-1,n,m)
                if i < n-1:
                    dfs(i+1,j,n,m)
                if j < m-1:
                    dfs(i,j+1,n,m)
            
        res = 0
        
        n = len(grid)
        m = len(grid[0])
        
        global tmp
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    tmp = 0
                    dfs(i,j,n,m)
                    res = max(tmp,res)
        return res

# Runtime: 151 ms, faster than 87.91% of Python3 online submissions for Max Area of Island.
# Memory Usage: 16.6 MB, less than 59.41% of Python3 online submissions for Max Area of Island.


# In[ ]:


# 1768. Merge Strings Alternately

class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        n1 = len(word1)
        n2 = len(word2)
        res = ''
        if n1 >= n2:
            for i in range(n2):
                res += word1[i]+word2[i]
            res += word1[(i+1):]
        else:
            for i in range(n1):
                res += word1[i]+word2[i]
            res += word2[(i+1):]
        return res

# Runtime: 37 ms, faster than 83.99% of Python3 online submissions for Merge Strings Alternately.
# Memory Usage: 13.9 MB, less than 18.28% of Python3 online submissions for Merge Strings Alternately.


# In[ ]:


# 2119. A Number After a Double Reversal

class Solution:
    def isSameAfterReversals(self, num: int) -> bool:
        if num == 0:
            return True
        return num%10

# Runtime: 38 ms, faster than 77.70% of Python3 online submissions for A Number After a Double Reversal.
# Memory Usage: 13.9 MB, less than 52.97% of Python3 online submissions for A Number After a Double Reversal.


# In[ ]:


# 168. Excel Sheet Column Title

class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        res = ''
        n = columnNumber-1
        q = n//26
        r = n%26
        
        while n>=0:
            res = chr(r+65)+res
            n = q-1
            q = n//26
            r = n%26

        return res 

# Runtime: 33 ms, faster than 88.24% of Python3 online submissions for Excel Sheet Column Title.
# Memory Usage: 13.8 MB, less than 56.56% of Python3 online submissions for Excel Sheet Column Title.


# In[ ]:


# 1632. Rank Transform of a Matrix

class Solution:
    def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
        # return
        def rank(arr):
            tmp = [[num, i, -1] for i, num in enumerate(arr)]
            ordered = sorted(tmp)
            for i, pair in enumerate(ordered):
                if i == 0:
                    continue
                pair[2] = ordered[i-1][1]
            return([pair[2] for pair in tmp])
        
        n = len(matrix)
        m = len(matrix[0])
        
        mini = matrix[0][0]
        
        for i in range(n):
            for j in range(m):
                mini = min(mini, matrix[i][j])
                
        for i in range(n):
            for j in range(m):
                matrix[i][j] += (1-mini)
        
        row_prev = {row: rank(matrix[row]) for row in range(n)}
        col_prev = {col: rank([matrix[row][col] for row in range(n)]) for col in range(m)}

        
        res = deepcopy(matrix)
        update = 1
        cnt = n*m
        visited = set()
        while len(visited)<n*m:
            for i in range(n):
                for j in range(m):
                    flag = True
                    
                    pj = row_prev[i][j]   
                    pi = col_prev[j][i]
                    # print(pj,pi)
                    if pj >= 0:
                        if matrix[i][pj] == matrix[i][j]:
                            if res[i][pj] > res[i][j]:
                                res[i][j] += 1
                            if res[i][pj] > update:
                                flag = False
                        else:
                            if res[i][pj] >= res[i][j]:
                                res[i][j] += 1
                            if res[i][pj] >= update:
                                flag = False
                    if pi >= 0:
                        # print(i,j)
                        if matrix[pi][j] == matrix[i][j]:
                            if res[pi][j] > res[i][j]:
                                res[i][j] += 1
                            if res[pi][j] > update:
                                flag = False
                        else:
                            if res[pi][j] >= res[i][j]:
                                res[i][j] += 1
                            if res[pi][j] >= update:
                                flag = False
                    # save to update
                    if res[i][j] == update:
                        if i*m+j not in visited:
                            visited.add(i*m+j)
                            cnt -= 1
                            
                    elif res[i][j] > update:    
                        if flag:
                            res[i][j] = update
                            visited.add(i*m+j)
                            cnt -= 1
                    
                    if pj>=0 and matrix[i][pj] == matrix[i][j] and res[i][pj]<res[i][j]:
                        res[i][pj] = res[i][j]
                    
                    if pi>=0 and matrix[pi][j] == matrix[i][j] and res[pi][j]<res[i][j]:
                        res[pi][j] = res[i][j]
            # break
            # if update>60:
            #     for row in res:
            #         print(row)
            #     print()
            update += 1
        
        for i in range(n):
            for j in range(m):
                pj = row_prev[i][j]   
                pi = col_prev[j][i]
                if pj >= 0:
                    if matrix[i][pj] == matrix[i][j]:
                        if res[i][pj] > res[i][j]:
                            res[i][j] += 1
                    else:
                        if res[i][pj] >= res[i][j]:
                            res[i][j] += 1
                if pi >= 0:
                    if matrix[pi][j] == matrix[i][j]:
                        if res[pi][j] > res[i][j]:
                            res[i][j] += 1
                    else:
                        if res[pi][j] >= res[i][j]:
                            res[i][j] += 1
                        
        return res

# Time Limit Exceeded

