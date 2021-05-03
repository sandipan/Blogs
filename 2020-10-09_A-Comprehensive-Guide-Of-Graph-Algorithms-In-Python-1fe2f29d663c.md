A Comprehensive Guide Of Graph Algorithms In Python {.p-name}
===================================================

In this blog we shall discuss about a few popular graph algorithms and
their python implementations. The problems discussed appeared as…

* * * * *

### A Comprehensive Guide Of Graph Algorithms In Python {#b0e7 .graf .graf--h3 .graf--leading .graf--title name="b0e7"}

#### Python implementation of a few algorithms on graphs {#ff3a .graf .graf--h4 .graf-after--h3 .graf--subtitle name="ff3a"}

In this blog we shall discuss about a few popular graph algorithms and
their python implementations. The problems discussed here appeared as
programming assignments in the coursera course [Algorithms on
Graphs](https://www.coursera.org/learn/algorithms-on-graphs/) and on
[Rosalind](http://rosalind.info/problems/). The problem statements are
taken from the course itself.

The basic building blocks of graph algorithms such as computing the
number of connected components, checking whether there is a path between
the given two vertices, checking whether there is a cycle, etc are used
practically in many applications working with graphs: for example,
finding shortest paths on maps, analyzing social networks, analyzing
biological data.

For all the problems it can be assumed that the given input graph is
simple, i.e., it does not contain self-loops (edges going from a vertex
to itself) and parallel edges.

### Checking if two vertices are Reachable {#71ed .graf .graf--h3 .graf-after--p name="71ed"}

Given an undirected graph G=(V,E) and two distinct vertices 𝑢 and 𝑣,
check if there is a path between 𝑢 and 𝑣.

#### **Steps** {#262e .graf .graf--h4 .graf-after--p name="262e"}

1.  Starting from the node u, we can simply use breadth first search
    (bfs) or depth-first search (dfs) to explore the nodes reachable
    from u.
2.  As soon as we find v we can return the nodes are reachable from
    one-another.
3.  If v is not there in the nodes explored, we can conclude v is not
    reachable from u.
4.  The following implementation (demonstrated using the following
    animations) uses iterative dfs (with stack) to check if two nodes
    (initially colored pink) are reachable.
5.  We can optionally implement **coloring**of the nodes w.r.t. the
    following convention: initially the nodes are all **white**, when
    they are visited (pushed onto stack) they are marked as **gray**and
    finally when all the adjacent (children) nodes for a given node are
    visited, the node can be marked **black**.
6.  We can store the parent of each node in an array and we can extract
    the path between two given nodes using the parent array, if they are
    reachable.

![](https://cdn-images-1.medium.com/max/800/0*g11iLSYFcNSX64Vl.png)

Image taken from this
[lecture notes](https://d3c33hcgiwev3.cloudfront.net/_a9267009ba78cede2b66112aa0f9cdb5_09_graph_decomposition_4_connectivity.pdf?Expires=1602460800&Signature=lTaKEu2sJuc3VwwmZOXAzxfyXXizBlH-FCXMaw2oFr1UJBlHuPwrchJ-UT9pAsrQdGmFvsWrQ42-DtQur8tCvd~pM72Rsei2KP0nheQQ-cPkFkbjYBuiTryWumr2JBz2B1jLdpZLVAC5gEdgJW5dtO8OmrMIwMngO7uQbPCINoc_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

A very basic python implementation of the iterative dfs are shown below
(here adj represents the adjacency list representation of the input
graph):

``` {#e856 .graf .graf--pre .graf-after--p name="e856"}
def reach(adj, x, y):  visited = [False]*len(adj) stack = [x] visited[x] = True while len(stack) > 0:  u = stack.pop(-1)  for v in adj[u]:   if not visited[v]:    stack.append(v)    visited[v] = True    if v == y:     return 1 return 0
```

The following animations demonstrate how the algorithm works, the
**stack**is also shown at different points in time during the execution.
Finally the path between the nodes are shown if they are reachable.

![](https://cdn-images-1.medium.com/max/800/1*W7m7YtzEYbEWlH3TW4AXIQ.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/1*sWTcw5vXWWYRbhgQyTZ6MQ.gif)

Image by Author

The same algorithm can be used for finding an Exit from a Maze (check
whether there is a path from a given cell to a given exit).

### Find Connected Components in an Undirected Graph {#ef15 .graf .graf--h3 .graf-after--p name="ef15"}

Given an undirected graph with 𝑛 vertices and 𝑚 edges, compute the
number of connected components in it.

#### **Steps** {#5337 .graf .graf--h4 .graf-after--p name="5337"}

1.  The following simple modification in dfs can be used to find the
    number of connected components in an undirected graph, as shown in
    the following figure.
2.  From each node we need to find all the nodes yet to be explored.
3.  We can find the nodes in a given component by finding all the nodes
    reachable from a given node.
4.  The same iterative dfs implementation was used (demonstrated with
    the animations below).
5.  The nodes in a component found were colored using the same color.

![](https://cdn-images-1.medium.com/max/800/0*kPsQD6o9E4LlNO3d.png)

Image taken from this
[lecture note](https://d3c33hcgiwev3.cloudfront.net/_a9267009ba78cede2b66112aa0f9cdb5_09_graph_decomposition_4_connectivity.pdf?Expires=1602460800&Signature=lTaKEu2sJuc3VwwmZOXAzxfyXXizBlH-FCXMaw2oFr1UJBlHuPwrchJ-UT9pAsrQdGmFvsWrQ42-DtQur8tCvd~pM72Rsei2KP0nheQQ-cPkFkbjYBuiTryWumr2JBz2B1jLdpZLVAC5gEdgJW5dtO8OmrMIwMngO7uQbPCINoc_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

**Python Code**

``` {#911f .graf .graf--pre .graf-after--p name="911f"}
def number_of_components(adj): result = 0 n = len(adj) visited = [False]*len(adj) for x in range(n):  if not visited[x]:   result += 1   stack = [x]   visited[x] = True   while len(stack) > 0:    u = stack.pop(-1)    for v in adj[u]:     if not visited[v]:      stack.append(v)       visited[v] = True return result
```

![](https://cdn-images-1.medium.com/max/800/0*5gRFD8FpJU2xkPDL.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*jDDI7HlkF88vLUpm.gif)

Image by Author

The same algorithm can be used to decide that there are no dead zones in
a maze, that is, that at least one exit is reachable from each cell.

### Find Euler Tour / Circuit in a Directed Graph {#56b3 .graf .graf--h3 .graf-after--p name="56b3"}

Given A [directed graph](http://rosalind.info/glossary/directed-graph/)
that contains an Eulerian tour, where the graph is given in the form of
an [adjacency list](http://rosalind.info/glossary/adjacency-list/).

#### **Steps** {#e265 .graf .graf--h4 .graf-after--p name="e265"}

1.  While solving the famous Königsberg Bridge Problem, Euler proved
    that an Euler circuit in an undirected graph exists iff all its
    nodes have even degree.
2.  For an Euler tour to exist in an undirected graph, if there exists
    odd-degree nodes in the graph, there must be exactly 2 nodes with
    odd degrees — the tour will start from one such node and end in
    another node.
3.  The tour must visit all the edges in the graph exactly once.
4.  The degree test can be extended to directed graphs (with in-degrees
    and out-degrees) and can be used to determine the existence of Euler
    tour / circuit in a Digraph.
5.  Flury’s algorithm can be used to iteratively remove the edges
    (selecting the edges not burning the bridges in the graph as much as
    possible) from the graph and adding them to the tour.
6.  DFS can be modified to obtain the Euler tour (circuit) in a DiGraph.
7.  The following figure shows these couple of algorithms for finding
    the Euler tour / circuit in a graph if one exists (note in the
    figure **path** is meant to indicate **tour**). We shall use DFS to
    find Euler tour.

![](https://cdn-images-1.medium.com/max/800/1*CrYeq3zLOc64NSDB5hL5vg.png)

Image created from Youtube Videos cited inside the image

The following code snippets represent the functions used to find Euler
tour in a graph. Here, variables n and m represent the number of
vertices and edges of the input DiGraph, respectively, whereas adj
represents the corresponding adjacency list.

**Python Code**

``` {#2f0e .graf .graf--pre .graf-after--p name="2f0e"}
def count_in_out_degrees(adj): n = len(adj) in_deg, out_deg = [0]*n, [0]*n for u in range(n):  for v in adj[u]:   out_deg[u] += 1   in_deg[v] += 1 return in_deg, out_deg
```

``` {#4855 .graf .graf--pre .graf-after--pre name="4855"}
def get_start_if_Euler_tour_present(in_deg, out_deg): start, end, tour = None, None, True for i in range(len(in_deg)):   d = out_deg[i] - in_deg[i]  if abs(d) > 1:   tour = False   break  elif d == 1:   start = i  elif d == -1:   end = i tour = (start != None and end != None) or \        (start == None and end == None) if tour and start == None: # a circuit  start = 0 return (tour, start)
```

``` {#6a48 .graf .graf--pre .graf-after--pre name="6a48"}
def dfs(adj, v, out_deg, tour): while out_deg[v] > 0:  out_deg[v] -= 1  dfs(adj, adj[v][out_deg[v]], out_deg, tour) tour[:] = [v] + tour
```

``` {#d9d0 .graf .graf--pre .graf-after--pre name="d9d0"}
def compute_Euler_tour(adj): n, m = len(adj), sum([len(adj[i]) for i in range(len(adj))]) in_deg, out_deg = count_in_out_degrees(adj) tour_present, start = get_start_if_Euler_tour_present(in_deg, \                                                       out_deg) if not tour_present:  return None tour = [] dfs(adj, start, out_deg, tour) if len(tour) == m+1:  return tourreturn None
```

The following animations show how the algorithm works.

![](https://cdn-images-1.medium.com/max/800/1*u1TwK7emOkJixoyY62mGWA.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/1*mCUVRbjjwzjaLG1yCogM7A.gif)

Image by Author

### Cycle Detection in a Directed Graph {#d1cd .graf .graf--h3 .graf-after--figure name="d1cd"}

Check whether a given directed graph with 𝑛 vertices and 𝑚 edges
contains a cycle.

The following figure shows the classification of the edges encountered
in DFS:

![](https://cdn-images-1.medium.com/max/800/1*iDj0vN4x3T_7mn74ARmnOg.png)

Image taken from this
[lecture notes](https://courses.csail.mit.edu/6.006/fall11/rec/rec14.pdf)

It can be shown that whether a Directed Graph is acyclic (DAG) or not
(i.e. it contains a cycle), can be checked using the presence of a
back-edge while DFS traversal.

#### **Steps** {#2b29 .graf .graf--h4 .graf-after--p name="2b29"}

1.  Use the recursive DFS implementation (pseudo-code shown in the below
    figure)
2.  Track if a node to be visited is already on the stack, if it’s
    there, it forms a back edge.
3.  Use parents array to obtain the directed cycle, if found.

![](https://cdn-images-1.medium.com/max/800/1*wIYIvrtcV38TTeYw8oYd3Q.png)

Image taken from this
[lecture notes](http://www.cs.cmu.edu/afs/cs/academic/class/15750-s18/ScribeNotes/lecture9.pdf)

**Python code**

``` {#04f5 .graf .graf--pre .graf-after--p name="04f5"}
def acyclic(adj): n = len(adj) visited = [False]*n parents = [None]*n on_stack = [False]*n cycle = []  def dfs_visit(adj, u, cycle):  visited[u] = True  on_stack[u] = True  for v in adj[u]:   if not visited[v]:    parents[v] = u    dfs_visit(adj, v, cycle)   elif on_stack[v]:    x = u    while x != v:     cycle.append(x)     x = parents[x]    cycle = [v] + cycle    #print(cycle)      on_stack[u] = False   for v in range(n):  if not visited[v]:   dfs_visit(adj, v, cycle)  return int(len(cycle) > 0)
```

![](https://cdn-images-1.medium.com/max/800/0*UcmazpymDBv3wvzx.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*VjoOCp8LicC3uZvw.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*oOGHjI2fQhDPpeOE.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*Y_1ediDZ-sHThZ7T.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*DURgTCcZIXzGADnc.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*AHofhQdg-sZQuUSP.gif)

Image by Author

The above algorithm can be used to check consistency in a curriculum
(e.g., there is no cyclic dependency in the prerequisites for each
course listed).

### Topologically Order a Directed Graph {#0b8d .graf .graf--h3 .graf-after--p name="0b8d"}

Compute a topological ordering of a given directed acyclic graph (DAG)
with 𝑛 vertices and 𝑚 edges.

The following idea can be used to obtain such an ordering in a DiGraph
G:

-   Find sink.
-   Put at end of order.
-   Remove from graph.
-   Repeat.

It can be implemented efficiently with dfs by keeping track of the time
of pre and post visiting the nodes.

#### **Steps** {#b4b4 .graf .graf--h4 .graf-after--p name="b4b4"}

1.  Use the recursive implementation of dfs.
2.  When visiting a node with a recursive call, record the pre and post
    visit times, at the beginning and the end (once all the children of
    the given node already visited) of the recursive call, respectively.
3.  Reorder the nodes by their post-visit times in descending order, as
    shown in the following figure.

![](https://cdn-images-1.medium.com/max/800/0*U6f9dLzXnXjKCkgp.png)

Image taken from this
[lecture notes](https://d3c33hcgiwev3.cloudfront.net/_dc4aebf17f30b5504eb92843c2d9e0fc_09_graph_decomposition_7_topological-sort.pdf?Expires=1602460800&Signature=TVI3lkqb1QopEmCEco~YTOqdh-kmEZ7QbO7k~SguPAqCGSJtFTFPb4dS7VLNTrP7BVVdtZXPQtVBCEg529wqQpfIuoCmX8s~FPxFN6zz4voj8iF09bdXmnQoKS0pD7upXvXhw5zucgul8vwYOlKRv-YEbHTom1A5jxumF2oaQns_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

**Python code**

``` {#f4e3 .graf .graf--pre .graf-after--p name="f4e3"}
def toposort(adj): order = [] n = len(adj) visited = [False]*n previsit = [0]*n postvisit = [0]*n  def dfs_visit(adj, u):  global clock  visited[u] = True  previsit[u] = clock  clock += 1  for v in adj[u]:   if not visited[v]:    dfs_visit(adj, v)  postvisit[u] = clock  clock += 1  for v in range(n):  if not visited[v]:   dfs_visit(adj, v)  order = [x for _, x in sorted(zip(postvisit, range(n)), \                   key=lambda pair: pair[0], reverse=True)]  return order
```

![](https://cdn-images-1.medium.com/max/800/1*E9vGtdYsTFDr3zmzc3fFeA.gif)

![](https://cdn-images-1.medium.com/max/800/1*kcvA2tDnV6pwMR13coKOeA.gif)

The above algorithm can be used to determine the order of the courses in
a curriculum, taking care of the pre-requisites dependencies.

### KosaRaju’s Algorithm to find the Strongly Connected Components (SCCs) in a Digraph {#c67d .graf .graf--h3 .graf-after--p name="c67d"}

Compute the number of strongly connected components of a given directed
graph with 𝑛 vertices and 𝑚 edges.

Note the following:

-   dfs can be used to find SCCs in a Digraph.
-   We need to make it sure that dfs traversal does not leave a such
    component with an outgoing edge.
-   The sink component is one that does not have an outgoing edge. We
    need to find a sink component first.
-   The vertex with the largest post-visit time is a source component
    for dfs.
-   The reverse (or transpose) graph of G has same SCC as the original
    graph.
-   Source components of the transpose of G are sink components of G.

With all the above information, the following algorithm can be
implemented to obtain the SCCs in a digraph G.

![](https://cdn-images-1.medium.com/max/800/1*2YVgR68z02VtlPO-o4mGTQ.png)

Image taken from this
[lecture notes](http://www.cs.cmu.edu/afs/cs/academic/class/15750-s18/ScribeNotes/lecture10.pdf)

The runtime of the algorithm is again O(|V |+|E|). Alternatively, the
algorithm can be represented as follows:

![](https://cdn-images-1.medium.com/max/800/0*qhhOmPOP06gMxBra.png)

Taken from these
[lecture notes](https://d3c33hcgiwev3.cloudfront.net/_dc4aebf17f30b5504eb92843c2d9e0fc_09_graph_decomposition_9_computing-sccs.pdf?Expires=1602460800&Signature=AbrMOHjITscbzXEGTcFbu8j2tS3s9B0N-ExWCOYnMgPmItDp7MF68G4i2Y7whSUnh-gmgLk2R8EvwVREGAV2mZlOTbooVYEwMsOjtK7KKj4wAA30ZthK1EysblCa77Brqa~6ctSsWH4-eDPNY7-W8CUQezP~Tw2JHjVvw5gFIB4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

**Python Code**

``` {#0605 .graf .graf--pre .graf-after--p name="0605"}
def number_of_strongly_connected_components(adj): result = 0 visited = [False]*n previsit = [0]*n postvisit = [0]*n  def reverse_graph(adj):  n = len(adj)  new_adj = [ [] for _ in range(n)]  for i in range(n):   for j in adj[i]:    new_adj[j].append(i)  return new_adj  def dfs_visit(adj, u):  global clock  visited[u] = True  previsit[u] = clock  clock += 1  for v in adj[u]:   if not visited[v]:    dfs_visit(adj, v)  postvisit[u] = clock  clock += 1  for v in range(n):  if not visited[v]:   dfs_visit(adj, v) post_v = [x for _, x in sorted(zip(postvisit, range(n)), \                   key=lambda pair: pair[0], reverse=True)] rev_adj = reverse_graph(adj) visited = [False]*n for v in post_v:  if not visited[v]:   dfs_visit(rev_adj, v)   result += 1
```

``` {#f524 .graf .graf--pre .graf-after--pre name="f524"}
return result
```

The following animations show how the algorithm works:

![](https://cdn-images-1.medium.com/max/800/0*C9ZepPPjhGmo1bmx.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*mvjXunf9w3z1J0BG.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*ngp-LPeJQbZElsTE.gif)

Image by Author

### Shortest Path in an Undirected Graph with BFS {#6701 .graf .graf--h3 .graf-after--figure name="6701"}

Given an undirected graph with 𝑛 vertices and 𝑚 edges and two vertices 𝑢
and 𝑣, compute the length of a shortest path between 𝑢 and 𝑣 (that is,
the minimum number of edges in a path from 𝑢 to 𝑣).

-   The following figure shows the algorithm for bfs.
-   It uses a queue (FIFO) instead of a stack (LIFO) to store the nodes
    to be explored.
-   The traversal is also called level-order traversal, since the nodes
    reachable with smaller number of hops (shorter distances) from the
    source / start node are visited earlier than the higher-distance
    nodes.
-   The running time of bfs is again O(|E| + |V |).

![](https://cdn-images-1.medium.com/max/800/1*Cj-0Al3CRiNo_Tnb8Z4mXw.png)

Image taken from this
[lecture notes](https://d3c33hcgiwev3.cloudfront.net/_2db8444033550723d3a34a917a4ff044_10_shortest_paths_in_graphs_1_bfs.pdf?Expires=1602460800&Signature=HlgeOeup0zhbooyf8N1fsj3~Q7VpTsOVF4QjJLNXJKdU~g6w3B8P5F68Owy5NDjrJ~ILeGIIkNv1wzdEKehDVV1Uir5DluiLCDzsHB6~NPQ5Sm1BhDodzGHOruFmQeD8rCSOiWxNQXTIhygrabMudBvEmdye56ruzLU1OreP1KM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

**Python code**

``` {#5f7b .graf .graf--pre .graf-after--p name="5f7b"}
def distance(adj, s, t): inf = 10**9 #float('Inf') d = [inf]*len(adj) queue = [s] d[s] = 0 while len(queue) > 0:  u = queue.pop(0)  for v in adj[u]:   if d[v] ==  inf:    queue.append(v)    d[v] = d[u] + 1    if v == t:     return d[t] return -1
```

The following animations demonstrate how the algorithm works. The queue
used to store the vertices to be traversed is also shown.

![](https://cdn-images-1.medium.com/max/800/0*rQTYmdm1BnYkxwgu.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*36qA5obqflKZxmiC.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*yxouAYdcM9kw94R6.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*gn8jRZt-uEZxRN0U.gif)

Image by Author

The above algorithm can be used to compute the minimum number of flight
segments to get from one city to another one.

### Checking if a Graph is Bipartite (2-Colorable) {#911d .graf .graf--h3 .graf-after--p name="911d"}

Given an undirected graph with 𝑛 vertices and 𝑚 edges, check whether it
is bipartite.

#### **Steps** {#f145 .graf .graf--h4 .graf-after--p name="f145"}

1.  Note that a graph bipartite iff its vertices can be colored using 2
    colors (so that no adjacent vertices have the same color).
2.  Use bfs to traverse the graph from a starting node and color nodes
    in the alternate levels (measured by distances from source) with red
    (even level) and blue (odd level).
3.  If at any point in time two adjacent vertices are found that are
    colored using the same color, then the graph is not 2-colorable
    (hence not bipartite).
4.  If no such cases occur, then the graph is bipartite
5.  For a graph with multiple components, use bfs on each of them.
6.  Also, note that a graph is bipartite iff it contains an odd length
    cycle.

**Python code**

``` {#3fd2 .graf .graf--pre .graf-after--p name="3fd2"}
def bipartite(adj): color = [None]*len(adj) for vertex in range(len(adj)):  if not color[vertex]:   queue = [vertex]   color[vertex] = 'red'   while len(queue) > 0:    u = queue.pop(0)    for v in adj[u]:     if color[v] ==  color[u]:      return 0     if not color[v]:      queue.append(v)      color[v] = 'red' if color[u] == 'blue' else 'blue' return 1
```

The following animations demonstrate how the algorithm works.

![](https://cdn-images-1.medium.com/max/800/0*NQnCZnwnMbAHyYiT.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*DPSVRkSU295zmeGz.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*b094trb3yyHaaSnd.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*OY7Ptvt7oXGUyqRW.gif)

Image by Author

Notice that the last graph contained a length of cycle 4 (even length
cycle) and it was bipartite graph. The graph prior to this one contained
a triangle in it (cycle of odd length 3), it was not bipartite.

### Shortest Path in a Weighted Graph with Dijkstra {#a8d4 .graf .graf--h3 .graf-after--p name="a8d4"}

Given an directed graph with positive edge weights and with 𝑛 vertices
and 𝑚 edges as well as two vertices 𝑢 and 𝑣, compute the weight of a
shortest path between 𝑢 and 𝑣 (that is, the minimum total weight of a
path from 𝑢 to 𝑣).

-   Optimal substructure property : any subpath of an optimal path is
    also optimal.
-   Initially, we only know the distance to source node S and relax all
    the edges from S.
-   Maintain a set R of vertices for which dist is already set correctly
    (known region).
-   On each iteration take a vertex outside of R with the minimum
    dist-value, add it to R, and relax all its outgoing edges.
-   The next figure shows the pseudocode of the algorithm.
-   The algorithm works for any graph with non-negative edge weights.
-   The algorithm does not work if the input graph has negative edge
    weights (known region has the assumption that the dist can be
    reduced further, this assumption does not hold if the graph has
    negative edges).

![](https://cdn-images-1.medium.com/max/800/1*t-6VJF4RyqDwQ0K_9tdMiA.png)

Image created from this
[lecture notes](https://d3c33hcgiwev3.cloudfront.net/_dd335213aea9cbfb8509c60297a2019d_10_shortest_paths_in_graphs_2_dijkstra.pdf?Expires=1602460800&Signature=gNgLyoo5CtvDm6VfrRFO-UxrxGTdl0UJNhfUE2aI76DbMOCnTQIOIR9jNkB6Rq7z0U2r2S7~q9fQ3cf5Dhucnk6t-v9KGsdty4JdpX2~FGVMVLtFVr4r0TinbWd~DS9bfiRBpVQc0eDb8PWcHtr76k39Ag51RQfKEBshXGZjsoA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

**Python code**

``` {#13a9 .graf .graf--pre .graf-after--p name="13a9"}
import queuedef distance(adj, cost, s, t): inf = 10**19 n = len(adj) d = [inf]*n d[s] = 0 visited = [0]*n h = queue.PriorityQueue() for v in range(n):  h.put((d[v], v)) while not h.empty():  u = h.get()[1]  if visited[u]: continue  visited[u] = True  for i in range(len(adj[u])):   v = adj[u][i]   if d[v] > d[u] + cost[u][i]:    d[v] = d[u] + cost[u][i]    h.put((d[v], v)) return d[t] if d[t] != inf else -1
```

The following animations show how the algorithm works.

![](https://cdn-images-1.medium.com/max/800/1*CZZ1gjfs383Jgwmm6KRQAw.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/1*WF2sjI77bc2wop1QDJPU-g.gif)

Image by Author

The algorithm works for the undirected graphs as well. The following
animations show how the shortest path can be found on undirected graphs.

![](https://cdn-images-1.medium.com/max/800/0*NQPnzx-3H7FxoDr0.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*VGjdQ2rIvI5Q9wkL.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*YbQGAOpyy3FSqrfJ.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*ESjfoNvDiuK-4eot.gif)

Image by Author

### Detecting Negative Cycle in a Directed Graph with Bellman-Ford {#c208 .graf .graf--h3 .graf-after--figure name="c208"}

Given an directed graph with possibly negative edge weights and with 𝑛
vertices and 𝑚 edges, check whether it contains a cycle of negative
weight. Also, given a vertex 𝑠, compute the length of shortest paths
from 𝑠 to all other vertices of the graph.

-   For a Digraph with n nodes (without a negative cycle), the shortest
    path length in between two nodes (e.g., the source node and any
    other node) can be at most *n-1*.
-   Relax edges while dist changes (at most n-1 times, most of the times
    the distances will stop changing much before that).
-   This algorithm works even for negative edge weights.
-   The following figure shows the pseudocode of the algorithm.

![](https://cdn-images-1.medium.com/max/800/1*midVFvwnfA2UEBrpEmf7BQ.png)

Image created from this
[lecture notes](https://d3c33hcgiwev3.cloudfront.net/_9363153a6b5462c79d8ab30e2ad1708f_10_shortest_paths_in_graphs_3_bellman_ford.pdf?Expires=1602460800&Signature=ey5B6HBbQsl0FcZj0y1Hc5h1264Z7tpHTR22lsIQpniaxVW5BObeOFujmeOM-wUQJ6El4adPDJvJthNmyhZVJO-v3nv2hT3I9SAt5BK5sb585LwREV7~Tajt2JBkSWbk8Fs9JZjLM48Gy7OSuhb4igBow9VldtOAkIrwbY1-2ck_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

The above algorithm can also be used to detect a negative cycle in the
input graph.

-   Run n = |V | iterations of Bellman–Ford algorithm (i.e., just run
    the edge-relaxation for once more, for the n-th time).
-   If there exists a vertex, the distance of which still decreases, it
    implies that there exist a negative-weight cycle in the graph and
    the vertex is reachable from that cycle.
-   Save node v relaxed on the last iteration v is reachable from a
    negative cycle
-   Start from x ← v, follow the link x ← prev[x] for |V | times — will
    be definitely on the cycle
-   Save y ← x and go x ← prev[x] until x = y again

The above algorithm can be used to detect infinite arbitrage, with the
following steps:

-   Do |V | iterations of Bellman–Ford, save all nodes relaxed on V-th
    iteration-set A
-   Put all nodes from A in queue Q
-   Do breadth-first search with queue Q and find all nodes reachable
    from A
-   All those nodes and only those can have infinite arbitrage

**Python code**

``` {#a1db .graf .graf--pre .graf-after--p name="a1db"}
def negative_cycle(adj, cost): inf = 10**19 # float('Inf') n = len(adj) d = [inf]*n d[0] = 0 for k in range(n-1):  for u in range(n):   for i in range(len(adj[u])):    v = adj[u][i]    if d[v] > d[u] + cost[u][i]:     d[v] = d[u] + cost[u][i] for u in range(n):  for i in range(len(adj[u])):   v = adj[u][i]   if d[v] > d[u] + cost[u][i]:    d[v] = d[u] + cost[u][i]    return 1 return 0
```

The following animations demonstrate how the algorithm works.

![](https://cdn-images-1.medium.com/max/800/1*w0Nobom4cRKuefOwj-qjFw.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*pf74CRDGPx6vBM8t.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*tM1UdabPF85X2puM.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*xv2QNlEdoej7trrY.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*lWqc7mpo6bnF3Yc7.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/1*NH46CJYWCyIEawokUwxBFA.gif)

Image by Author

The last animation shows how the shortest paths can be computed with the
algorithm in the presence of negative edge weights.

### Find Minimum Spanning Tree in a Graph with Prim’s Greedy Algorithm {#a83e .graf .graf--h3 .graf-after--p name="a83e"}

Given 𝑛 points on a plane, connect them with segments of minimum total
length such that there is a path between any two points.

#### **Steps** {#e6a5 .graf .graf--h4 .graf-after--p name="e6a5"}

-   Let’s construct the following undirected graph G=(V, E): each node
    (in V) in the graph corresponds to a point in the plane and the
    weight of an edge (in E) in between any two nodes corresponds to the
    distance between the nodes.
-   The length of a segment with endpoints (𝑥1, 𝑦1) and (𝑥2, 𝑦2) is
    equal to the Euclidean distance √︀((𝑥1 − 𝑥2)² + (𝑦1 − 𝑦2)²).
-   Then the problem boils down to finding a Minimum Spanning Tree in
    the graph G.
-   The following figure defines the MST problem and two popular greedy
    algorithms Prim and Kruskal to solve the problem efficiently.

![](https://cdn-images-1.medium.com/max/800/1*DhmJK-GDXSbRTLNvjdRVcA.png)

Image created from this
[lecture notes](https://d3c33hcgiwev3.cloudfront.net/_33e1de78cbff8069444dd63dce86bb97_11_1_minimum_spanning_trees.pdf?Expires=1602460800&Signature=FP~xaOFtVd2b7d4t3nNGgUtMZfdQDS-z3bTQewTjTk4HL-H9jYqLSqoH1Bz~R4NGVExBL6PamIgcGKVSU7BEJXvJW2fXGxVCrsh9XiBnUtB3oAfFouEIjAJqskD5w~q2q3-OBCEP0i~j3k3~q3GksrShm9-~uWqK311rrlXibxA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

We shall use Prim’s algorithm to find the MST in the graph. Here are the
algorithm steps:

-   It starts with an initial vertex and is grown to a spanning tree X.
-   X is always a subtree, grows by one edge at each iteration
-   Add a lightest edge between a vertex of the tree and a vertex not in
    the tree
-   Very similar to Dijkstra’s algorithm
-   The pseudocode is shown in the following figure.

![](https://cdn-images-1.medium.com/max/800/1*Mwe1kV8yi6f7G96djtjBPA.png)

Image taken from this
[lecture notes](https://d3c33hcgiwev3.cloudfront.net/_33e1de78cbff8069444dd63dce86bb97_11_1_minimum_spanning_trees.pdf?Expires=1602460800&Signature=FP~xaOFtVd2b7d4t3nNGgUtMZfdQDS-z3bTQewTjTk4HL-H9jYqLSqoH1Bz~R4NGVExBL6PamIgcGKVSU7BEJXvJW2fXGxVCrsh9XiBnUtB3oAfFouEIjAJqskD5w~q2q3-OBCEP0i~j3k3~q3GksrShm9-~uWqK311rrlXibxA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

**Python code**

``` {#6bc0 .graf .graf--pre .graf-after--p name="6bc0"}
def minimum_distance(x, y): result = 0. inf = 10**19 n = len(x) adj = [[0 for _ in range(n)] for _ in range(n)] for i in range(n):  for j in range(i+1, n):   adj[i][j] = adj[j][i] = math.sqrt((x[i]-x[j])**2 + \                                     (y[i]-y[j])**2) c = [inf]*n s = 0 c[s] = 0 visited = [False]*n parent = [None]*n h = queue.PriorityQueue() for v in range(n):  h.put((c[v], v)) while not h.empty():  w, u = h.get()  if visited[u]: continue  visited[u] = True  for v in range(n):   if v == u: continue   if (not visited[v]) and (c[v] > adj[u][v]):    c[v] = adj[u][v]    parent[v] = u    h.put((c[v], v)) spanning_tree = [] for i in range(n):  spanning_tree.append((i, parent[i]))  if parent[i] != None:    result += adj[i][parent[i]] #print(spanning_tree) return result
```

The following animations demonstrate how the algorithm works for a few
different set of input points in 2-D plane.

![](https://cdn-images-1.medium.com/max/800/0*5DvhnkkvWz8bsj3s.gif)

Image by Author

The following two animations show the algorithm steps on the input
points and the corresponding graph formed, respectively.

![](https://cdn-images-1.medium.com/max/800/0*CyHR0iqACwyCvFrv.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*StBut9ui28SBlK0H.gif)

Image by Author

The following animation shows how Prim’s algorithm outputs an MST for 36
input points in a 6x6 grid.

![](https://cdn-images-1.medium.com/max/800/0*GRIrT11JzKIPF0If.gif)

Image by Author

As expected, the cost of the MCST formed is 35 for the 36 points, since
a spanning tree with n nodes has n-1 edges, each with unit length.

The above algorithm can be used to build roads between some pairs of the
given cities such that there is a path between any two cities and the
total length of the roads is minimized.

### Finding Minimum Spanning Tree and Hierarchical Clustering with Kruskal {#2fa6 .graf .graf--h3 .graf-after--p name="2fa6"}

Given 𝑛 points on a plane and an integer 𝑘, compute the largest possible
value of 𝑑 such that the given points can be partitioned into 𝑘
non-empty subsets in such a way that the distance between any two points
from different subsets is at least 𝑑.

#### **Steps** {#a8ca .graf .graf--h4 .graf-after--p name="a8ca"}

-   We shall use Kruskal’s algorithm to solve the above problem. Each
    point can be thought of a node in the graph, as earlier, with the
    edge weights between the nodes being equal to the Euclidean distance
    between them.
-   Start with n components, each node representing one.
-   Run iterations of the Kruskal’s algorithm and merge the components
    till there are exactly k (\< n) components left.
-   These k components will be the k desired clusters of the points and
    we can compute d to be the largest distance in between two points
    belonging to different clusters.

Here are the steps for the Kruskal’s algorithm to compute MST:

-   Start with each node in the graph as a single-node tree in a forest
    X.
-   Repeatedly add to X the next lightest edge e that doesn’t produce a
    cycle.
-   At any point of time, the set X is a forest (i.e., a collection of
    trees)
-   The next edge e connects two different trees — say, T1 and T2
-   The edge e is the lightest between T1 and V / T1, hence adding e is
    safe

![](https://cdn-images-1.medium.com/max/800/1*A0NY9udAhPvS1jGDI-INVw.png)

Image taken from this
[lecture notes](https://d3c33hcgiwev3.cloudfront.net/_33e1de78cbff8069444dd63dce86bb97_11_1_minimum_spanning_trees.pdf?Expires=1602460800&Signature=FP~xaOFtVd2b7d4t3nNGgUtMZfdQDS-z3bTQewTjTk4HL-H9jYqLSqoH1Bz~R4NGVExBL6PamIgcGKVSU7BEJXvJW2fXGxVCrsh9XiBnUtB3oAfFouEIjAJqskD5w~q2q3-OBCEP0i~j3k3~q3GksrShm9-~uWqK311rrlXibxA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

-   Use disjoint sets data structure for implementation
-   Initially, each vertex lies in a separate set
-   Each set is the set of vertices of a connected component
-   To check whether the current edge {u, v} produces a cycle, we check
    whether u and v belong to the same set (*find*).
-   Merge two sets using *union*operation.
-   The following figure shows a few algorithms to implement union-find
    abstractions for disjoint sets.

![](https://cdn-images-1.medium.com/max/800/1*jhdga-WCPk1xO2meBljLRw.png)

Image created from this
[lecture notes](https://www.cs.princeton.edu/~rs/AlgsDS07/01UnionFind.pdf)

**Python code**

``` {#8930 .graf .graf--pre .graf-after--p name="8930"}
def clustering(x, y, k): result = 0. inf = float('Inf') #10**19 n = len(x) adj = [] for i in range(n):  for j in range(i+1, n):   adj.append((math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2), i, j)) adj = sorted(adj) indices = {i:i for i in range(n)} while len(set(indices.values())) > k:  d, u, v = adj.pop(0)  iu, iv = indices[u], indices[v]  # implemented quick-find here  # To-do: implement weighted union with path-compression heuristic        if iu != iv:   indices[u] = indices[v] = min(iu, iv)    for j in range(n):    if indices[j] == max(iu, iv):      indices[j] = min(iu, iv)  clusters = {} for i in range(n):  ci = indices[i]  clusters[ci] = clusters.get(ci, []) + [i] #print(clusters) d = inf for i in list(clusters.keys()):  for j in list(clusters.keys()):   if i == j: continue   for vi in clusters[i]:    for vj in clusters[j]:     d = min(d, math.sqrt((x[vi]-x[vj])**2 + (y[vi]-y[vj])**2)) return d
```

The following animations demonstrate how the algorithm works. Next
couple of animations show how 8 points in a plane are (hierarchically)
clustered using Kruskal’s algorithm and finally the MST is computed. The
first animation shows how the points are clustered and the next one
shows how Kruskal works on the corresponding graph created from the
points.

![](https://cdn-images-1.medium.com/max/800/0*6DnGnSO8H0RFCIQM.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*Wuc181VqNvWEUYmE.gif)

Image by Author

The next animation again shows how a set of points in 2-D are clustered
using Kruskal and MST is computed.

![](https://cdn-images-1.medium.com/max/800/0*3RQHjhGquu0vhZ5a.gif)

Image by Author

**Clustering**is a fundamental problem in data mining. The goal is to
partition a given set of objects into subsets (or clusters) in such a
way that any two objects from the same subset are close (or similar) to
each other, while any two objects from different subsets are far apart.

Now we shall use Kruskal’s algorithm to cluster a small real-world
dataset named the **Iris**dataset (can be downloaded from the[UCI
Machine learning
repository](http://archive.ics.uci.edu/ml/datasets/Iris/)), often used
as a test dataset in Machine learning.

Next 3 animations show how Kruskal can be used to cluster Iris dataset .
The first few rows of the dataset is shown below:

![](https://cdn-images-1.medium.com/max/800/1*SMWscB8jifqkEZYNa4GMiQ.png)

Let’s use the first two features (SepalLength and SepalWidth), project
the dataset in 2-D and use Kruskal to cluster the dataset, as shown in
the following animation.

![](https://cdn-images-1.medium.com/max/800/0*0JNQyMizvqL73tUj.gif)

Image by Author

Now, let’s use the second and third feature variables (SepalWidth and
PetalLength), project the dataset in 2-D and use Kruskal to cluster the
dataset, as shown in the following animation.

![](https://cdn-images-1.medium.com/max/800/0*0X_Oz1GO9aB9h_W2.gif)

Image by Author

Finally, let’s use the third and fourth feature variables (PetalLength
and PetalWidth), project the dataset in 2-D and use Kruskal to cluster
the dataset, as shown in the following animation.

![](https://cdn-images-1.medium.com/max/800/0*xlINz4TIUtCQcSxm.gif)

Image by Author

### Friend Suggestion in Social Networks — Bidirectional Dijkstra {#d421 .graf .graf--h3 .graf-after--figure name="d421"}

Compute the distance between several pairs of nodes in the network.

#### **Steps** {#a0a8 .graf .graf--h4 .graf-after--p name="a0a8"}

-   Build reverse graph GR
-   Start Dijkstra from s in G and from t in GR
-   Alternate between Dijkstra steps in G and in GR
-   Stop when some vertex v is processed both in G and in GR
-   Compute the shortest path between s and t

**Meet-in-the-middle**

-   More general idea, not just for graphs
-   Instead of searching for all possible objects, search for first
    halves and for second halves separately
-   Then find *compatible* halves
-   Typically roughly O(√N) instead of O(N)

**Friends suggestions in social networks**

-   Find the shortest path from Michael to Bob via friends connections
-   For the two “farthest” people, Dijkstra has to look through 2
    billion people
-   If we only consider friends of friends of friends for both Michael
    and Bob, we will find a connection
-   Roughly 1M friends of friends of friends
-   1M + 1M = 2M people - 1000 times less
-   Dijkstra goes in **circles**
-   Bidirectional search idea can reduce the search space
-   Bidirectional Dijkstra can be 1000s times faster than Dijkstra for
    social networks
-   The following figure shows the algorithm

![](https://cdn-images-1.medium.com/max/800/1*5XotTQJDS4yaHDh9RHOf2A.png)

Created from this
[lecture note](https://d3c33hcgiwev3.cloudfront.net/_b1eab7f17568253a367e9651d64c34ef_19_advanced_shortest_paths_1_bidirectional_dijkstra.pdf?Expires=1602633600&Signature=dZ2ZOyuBh5keNLlwjLLexBMW4JmWMicfLcZI4EMPdqvufQ-SRfp3PEGmo7KTR~~4iQOxyllBiZUMdaycxk2ca8t7n6JyToEfBEPZinNjcvbDpZBDz9LR0nceHms7fVxybgbxYJwgeCjwtN8k5s53k2l-RW~QScxDadmB8M4EUkY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

The following python function implements the bidirectional Dijkstra, the
arguments *adj*and *adjR*represents the adjacency lists for the original
and the reverse graph, respectively.

**Python code**

``` {#7c95 .graf .graf--pre .graf-after--p name="7c95"}
def distance(adj, cost, adjR, costR, s, t):   def process(u, adj, cost, h, d, prev, visited):  for i in range(len(adj[u])):   v = adj[u][i]   if d[v] > d[u] + cost[u][i]:    d[v] = d[u] + cost[u][i]    h.put((d[v], v))    prev[v] = u  visited[u] = True  def shortest_path(s, dist, prev, t, distR, prevR, visited_any):  distance = inf  ubest = None  for u in visited_any:   if dist[u] + distR[u] < distance:    ubest = u    distance = dist[u] + distR[u]  return distance if ubest != None else -1  inf = 10**19 n = len(adj) d, dR = [inf]*n, [inf]*n d[s] = 0 dR[t] = 0 visited, visitedR = [False]*n, [False]*n visited_any = set([]) prev, prevR = [None]*n, [None]*n h = queue.PriorityQueue() h.put((d[s], s)) hR = queue.PriorityQueue() hr.put((dr[t], t))
```

``` {#60c6 .graf .graf--pre .graf-after--pre name="60c6"}
 while True:  u = h.get()[1]  if visited[u]: continue  process(u, adj, cost, h, d, prev, visited)  visited_any.add(u)  if visitedR[u]:   return shortest_path(s, d, prev, t, dr, prevR, visited_any)  uR = hR.get()[1]  if visitedR[uR]: continue  process(ur, adjR, costR, hR, dR, prevR, visitedR)  visited_any.add(uR)  if visited[uR]:   return shortest_path(s, d, prev, t, dR, prevR, visited_any)  if h.empty() or hr.empty():   return -1
```

The following animation shows how the algorithm works:

![](https://cdn-images-1.medium.com/max/800/1*8SpjQ_arDLcB2YiwhOtOCg.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/1*kp2daGqzG5D_gbznfSbheQ.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/1*JtR-cMJbUc3p7zfI0_M_qg.gif)

Image by Author

### Computing Distance Faster Using Coordinates with A\* search {#fa2a .graf .graf--h3 .graf-after--figure name="fa2a"}

Compute the distance between several pairs of nodes in the network. The
length l between any two nodes u an v is guaranteed to satisfy 𝑙 ≥
√︀((𝑥(𝑢) − 𝑥(𝑣))2 + (𝑦(𝑢) − 𝑦(𝑣))2).

-   Let’s say we are submitting a query to find a shortest path between
    nodes (s,t) to a graph.
-   A potential function 𝜋(v) for a node v in a is an estimation of d(v,
    t) — how far is it from here to t?
-   If we have such estimation, we can often avoid going wrong direction
    directed search
-   Take some potential function 𝜋 and run Dijkstra algorithm with edge
    weights ℓ𝜋
-   For any edge (u, v), the new length ℓ𝜋(u, v) must be non-negative,
    such 𝜋 is called feasible
-   Typically 𝜋(v) is a lower bound on d(v, t)
-   A\* is a directed search algorithm based on Dijkstra and potential
    functions
-   Run Dijkstra with the potential 𝜋 to find the shortest path — the
    resulting algorithm is A\* search, described in the following figure
-   On a real map a path from v to t cannot be shorter than the straight
    line segment from v to t
-   For each v, compute 𝜋(v) = dE (v, t)
-   If 𝜋(v) gives lower bound on d(v, t)
-   Worst case: 𝜋(v) = 0 for all v the same as Dijkstra
-   Best case: 𝜋(v) = d(v, t) for all v then ℓ𝜋(u, v) = 0 iff (u, v) is
    on a shortest path to t, so search visits only the edges of shortest
    s − t paths
-   It can be shown that the tighter are the lower bounds the fewer
    vertices will be scanned

![](https://cdn-images-1.medium.com/max/800/1*5KGMdpJGXvLJv6r71RxXXw.png)

The following animations show how the A\* search algorithm works on road
networks (and the corresponding graph representations). Each node in the
graph has the corresponding (distance, potential) pairs.

![](https://cdn-images-1.medium.com/max/800/0*SvyomK2id_2j82R1.gif)

Image by Author

![](https://cdn-images-1.medium.com/max/800/0*JzzMn5bUrw1AiYYO.gif)

Image by Author

The following animations show how the shortest path is computed between
two nodes for the USA road network for the San Francisco Bay area (the
corresponding graph containing 321270 nodes and 800172 edges), using
Dijkstra, bidirectional Dijkstra and A\* algorithms respectively, the
data can be downloaded from [9th DIMACS Implementation
Challenge — Shortest
Paths](http://users.diag.uniroma1.it/challenge9/download.shtml). Here,
with bidirectional Dijkstra, the nodes from forward and reverse priority
queues are popped in the same iteration.

![](https://cdn-images-1.medium.com/max/800/0*Vz-tnLSdbpJqDnyV.gif)

![](https://cdn-images-1.medium.com/max/800/0*JMTNijHYTER6OBjD.gif)

![](https://cdn-images-1.medium.com/max/800/0*F-XBArkqidbgc4A_.gif)

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [October 9,
2020](https://medium.com/p/1fe2f29d663c).

[Canonical
link](https://medium.com/@sandipan-dey/graph-algorithms-with-python-1fe2f29d663c)

Exported from [Medium](https://medium.com) on January 8, 2021.
