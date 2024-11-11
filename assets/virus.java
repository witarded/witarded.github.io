/// 1) Binary Search Implementation in Java

public class BinarySearch {
    public static int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target)
                return mid;
            if (arr[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return -1; // Target not found
    }

    public static void main(String[] args) {
        int[] arr = {2, 3, 4, 10, 40};
        int target = 10;
        int result = binarySearch(arr, target);
        System.out.println(result == -1 ? "Element not found" : "Element found at index " + result);
    }
}

/// 2) Merge Sort Implementation in Java

public class MergeSort {
    public static void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = (left + right) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
    }

    private static void merge(int[] arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        int[] L = new int[n1];
        int[] R = new int[n2];

        for (int i = 0; i < n1; i++)
            L[i] = arr[left + i];
        for (int j = 0; j < n2; j++)
            R[j] = arr[mid + 1 + j];

        int i = 0, j = 0, k = left;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k] = L[i];
                i++;
            } else {
                arr[k] = R[j];
                j++;
            }
            k++;
        }

        while (i < n1) {
            arr[k] = L[i];
            i++;
            k++;
        }

        while (j < n2) {
            arr[k] = R[j];
            j++;
            k++;
        }
    }

    public static void main(String[] args) {
        int[] arr = {12, 11, 13, 5, 6, 7};
        mergeSort(arr, 0, arr.length - 1);
        System.out.println("Sorted array:");
        for (int num : arr) {
            System.out.print(num + " ");
        }
    }
}

///3) Quick Sort Implementation in Java

public class QuickSort {
    public static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return i + 1;
    }

    public static void main(String[] args) {
        int[] arr = {10, 7, 8, 9, 1, 5};
        quickSort(arr, 0, arr.length - 1);
        System.out.println("Sorted array:");
        for (int num : arr) {
            System.out.print(num + " ");
        }
    }
}

///4) Recursive Max-Min Algorithm in Java

public class MaxMin {
    static class Pair {
        int max;
        int min;
    }

    public static Pair findMaxMin(int[] arr, int left, int right) {
        Pair result = new Pair();
        if (left == right) {
            result.max = arr[left];
            result.min = arr[left];
            return result;
        }
        if (right == left + 1) {
            if (arr[left] > arr[right]) {
                result.max = arr[left];
                result.min = arr[right];
            } else {
                result.max = arr[right];
                result.min = arr[left];
            }
            return result;
        }

        int mid = (left + right) / 2;
        Pair leftResult = findMaxMin(arr, left, mid);
        Pair rightResult = findMaxMin(arr, mid + 1, right);

        result.max = Math.max(leftResult.max, rightResult.max);
        result.min = Math.min(leftResult.min, rightResult.min);
        return result;
    }

    public static void main(String[] args) {
        int[] arr = {1000, 11, 445, 1, 330, 3000};
        Pair result = findMaxMin(arr, 0, arr.length - 1);
        System.out.println("Maximum element is " + result.max);
        System.out.println("Minimum element is " + result.min);
    }
}

///5) Fractional Knapsack Problem in Java

import java.util.Arrays;
import java.util.Comparator;

class Item {
    int value, weight;

    Item(int value, int weight) {
        this.value = value;
        this.weight = weight;
    }
}

public class FractionalKnapsack {
    private static double getMaxValue(Item[] items, int capacity) {
        Arrays.sort(items, new Comparator<Item>() {
            @Override
            public int compare(Item a, Item b) {
                double r1 = (double) a.value / a.weight;
                double r2 = (double) b.value / b.weight;
                return Double.compare(r2, r1);
            }
        });

        double totalValue = 0.0;
        for (Item item : items) {
            if (capacity >= item.weight) {
                capacity -= item.weight;
                totalValue += item.value;
            } else {
                totalValue += item.value * ((double) capacity / item.weight);
                break;
            }
        }
        return totalValue;
    }

    public static void main(String[] args) {
        Item[] items = {new Item(60, 10), new Item(100, 20), new Item(120, 30)};
        int capacity = 50;
        double maxValue = getMaxValue(items, capacity);
        System.out.println("Maximum value in Knapsack = " + maxValue);
    }
}

///6) Job Sequencing with Deadlines Problem in Java

import java.util.Arrays;
import java.util.Comparator;

class Job {
    int id, profit, deadline;

    Job(int id, int profit, int deadline) {
        this.id = id;
        this.profit = profit;
        this.deadline = deadline;
    }
}

public class JobSequencing {
    private static void scheduleJobs(Job[] jobs, int n) {
        Arrays.sort(jobs, (a, b) -> b.profit - a.profit);
        int maxDeadline = Arrays.stream(jobs).mapToInt(job -> job.deadline).max().orElse(0);
        boolean[] slots = new boolean[maxDeadline];
        int[] jobOrder = new int[maxDeadline];
        int totalProfit = 0;

        for (Job job : jobs) {
            for (int j = Math.min(job.deadline - 1, maxDeadline - 1); j >= 0; j--) {
                if (!slots[j]) {
                    slots[j] = true;
                    jobOrder[j] = job.id;
                    totalProfit += job.profit;
                    break;
                }
            }
        }

        System.out.println("Scheduled jobs for maximum profit:");
        for (int i = 0; i < maxDeadline; i++) {
            if (slots[i]) {
                System.out.print("Job " + jobOrder[i] + " ");
            }
        }
        System.out.println("\nTotal profit: " + totalProfit);
    }

    public static void main(String[] args) {
        Job[] jobs = {
            new Job(1, 100, 2),
            new Job(2, 19, 1),
            new Job(3, 27, 2),
            new Job(4, 25, 1),
            new Job(5, 15, 3)
        };
        scheduleJobs(jobs, jobs.length);
    }
}

/// 7) Prim's Algorithm for Minimum Cost Spanning Tree in Java

import java.util.Arrays;

public class PrimsAlgorithm {
    private static int findMinVertex(boolean[] visited, int[] weights) {
        int minVertex = -1;
        for (int i = 0; i < weights.length; i++) {
            if (!visited[i] && (minVertex == -1 || weights[i] < weights[minVertex])) {
                minVertex = i;
            }
        }
        return minVertex;
    }

    public static void prims(int[][] graph) {
        int n = graph.length;
        boolean[] visited = new boolean[n];
        int[] weights = new int[n];
        int[] parents = new int[n];
        Arrays.fill(weights, Integer.MAX_VALUE);
        weights[0] = 0;
        parents[0] = -1;

        for (int i = 0; i < n - 1; i++) {
            int minVertex = findMinVertex(visited, weights);
            visited[minVertex] = true;

            for (int j = 0; j < n; j++) {
                if (graph[minVertex][j] != 0 && !visited[j] && graph[minVertex][j] < weights[j]) {
                    weights[j] = graph[minVertex][j];
                    parents[j] = minVertex;
                }
            }
        }

        System.out.println("Minimum Spanning Tree:");
        for (int i = 1; i < n; i++) {
            System.out.println("Edge: " + parents[i] + " - " + i + " Weight: " + graph[i][parents[i]]);
        }
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 2, 0, 6, 0},
            {2, 0, 3, 8, 5},
            {0, 3, 0, 0, 7},
            {6, 8, 0, 0, 9},
            {0, 5, 7, 9, 0}
        };
        prims(graph);
    }
}

/// 8) Kruskal's Algorithm for Minimum Cost Spanning Tree in Java
java

import java.util.Arrays;

class Edge implements Comparable<Edge> {
    int src, dest, weight;

    public Edge(int src, int dest, int weight) {
        this.src = src;
        this.dest = dest;
        this.weight = weight;
    }

    public int compareTo(Edge compareEdge) {
        return this.weight - compareEdge.weight;
    }
}

class Subset {
    int parent, rank;
}

public class KruskalsAlgorithm {
    private int find(Subset[] subsets, int i) {
        if (subsets[i].parent != i)
            subsets[i].parent = find(subsets, subsets[i].parent);
        return subsets[i].parent;
    }

    private void union(Subset[] subsets, int x, int y) {
        int xroot = find(subsets, x);
        int yroot = find(subsets, y);

        if (subsets[xroot].rank < subsets[yroot].rank)
            subsets[xroot].parent = yroot;
        else if (subsets[xroot].rank > subsets[yroot].rank)
            subsets[yroot].parent = xroot;
        else {
            subsets[yroot].parent = xroot;
            subsets[xroot].rank++;
        }
    }

    public void kruskalMST(Edge[] edges, int V) {
        Arrays.sort(edges);
        Subset[] subsets = new Subset[V];
        for (int i = 0; i < V; i++) {
            subsets[i] = new Subset();
            subsets[i].parent = i;
            subsets[i].rank = 0;
        }

        int e = 0, i = 0;
        Edge[] result = new Edge[V - 1];

        while (e < V - 1 && i < edges.length) {
            Edge nextEdge = edges[i++];
            int x = find(subsets, nextEdge.src);
            int y = find(subsets, nextEdge.dest);

            if (x != y) {
                result[e++] = nextEdge;
                union(subsets, x, y);
            }
        }

        System.out.println("Edges in the MST:");
        for (i = 0; i < e; i++) {
            System.out.println(result[i].src + " - " + result[i].dest + " Weight: " + result[i].weight);
        }
    }

    public static void main(String[] args) {
        int V = 4;
        Edge[] edges = {
            new Edge(0, 1, 10),
            new Edge(0, 2, 6),
            new Edge(0, 3, 5),
            new Edge(1, 3, 15),
            new Edge(2, 3, 4)
        };

        KruskalsAlgorithm ka = new KruskalsAlgorithm();
        ka.kruskalMST(edges, V);
    }
}

/// 9) Optimal Storage on Tapes Problem in Java

import java.util.Arrays;

public class OptimalStorage {
    public static double getOptimalStorage(int[] programs) {
        Arrays.sort(programs);
        double sum = 0, totalTime = 0;
        for (int time : programs) {
            sum += time;
            totalTime += sum;
        }
        return totalTime / programs.length;
    }

    public static void main(String[] args) {
        int[] programs = {3, 5, 1, 2, 4};
        double avgAccessTime = getOptimalStorage(programs);
        System.out.println("Average retrieval time: " + avgAccessTime);
    }
}

/// 10) Optimal Merge Patterns Problem in Java

import java.util.PriorityQueue;

public class OptimalMergePattern {
    public static int optimalMerge(int[] files) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int file : files) {
            pq.add(file);
        }

        int totalCost = 0;
        while (pq.size() > 1) {
            int first = pq.poll();
            int second = pq.poll();
            int mergeCost = first + second;
            totalCost += mergeCost;
            pq.add(mergeCost);
        }
        return totalCost;
    }

    public static void main(String[] args) {
        int[] files = {5, 2, 4, 7};
        int minCost = optimalMerge(files);
        System.out.println("Minimum cost of merging: " + minCost);
    }
}

/// 11) Dijkstra’s Algorithm for Single Source Shortest Paths in Java

import java.util.*;

public class DijkstraAlgorithm {
    public static void dijkstra(int[][] graph, int src) {
        int V = graph.length;
        int[] dist = new int[V];
        boolean[] visited = new boolean[V];
        
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;
        
        for (int i = 0; i < V - 1; i++) {
            int u = minDistance(dist, visited);
            visited[u] = true;
            
            for (int v = 0; v < V; v++) {
                if (!visited[v] && graph[u][v] != 0 && dist[u] != Integer.MAX_VALUE && dist[u] + graph[u][v] < dist[v]) {
                    dist[v] = dist[u] + graph[u][v];
                }
            }
        }
        
        printSolution(dist);
    }
    
    private static int minDistance(int[] dist, boolean[] visited) {
        int min = Integer.MAX_VALUE, minIndex = -1;
        for (int v = 0; v < dist.length; v++) {
            if (!visited[v] && dist[v] <= min) {
                min = dist[v];
                minIndex = v;
            }
        }
        return minIndex;
    }
    
    private static void printSolution(int[] dist) {
        System.out.println("Vertex \t Distance from Source");
        for (int i = 0; i < dist.length; i++) {
            System.out.println(i + " \t " + dist[i]);
        }
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 10, 0, 0, 0, 0},
            {10, 0, 5, 0, 0, 15},
            {0, 5, 0, 10, 20, 0},
            {0, 0, 10, 0, 5, 0},
            {0, 0, 20, 5, 0, 15},
            {0, 15, 0, 0, 15, 0}
        };
        dijkstra(graph, 0);
    }
}

/// 12) Multistage Graph Problem (Forward Approach) in Java

import java.util.Arrays;

public class MultistageGraphForward {
    public static int minCost(int[][] graph, int stages) {
        int n = graph.length;
        int[] cost = new int[n];
        int[] path = new int[n];
        Arrays.fill(cost, Integer.MAX_VALUE);
        cost[n - 1] = 0;

        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (graph[i][j] != 0 && graph[i][j] + cost[j] < cost[i]) {
                    cost[i] = graph[i][j] + cost[j];
                    path[i] = j;
                }
            }
        }

        System.out.println("Minimum cost: " + cost[0]);
        System.out.print("Path: ");
        int current = 0;
        while (current < n - 1) {
            System.out.print(current + " -> ");
            current = path[current];
        }
        System.out.println(n - 1);
        return cost[0];
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 1, 2, 5, 0, 0, 0, 0},
            {0, 0, 0, 0, 4, 11, 0, 0},
            {0, 0, 0, 0, 9, 5, 16, 0},
            {0, 0, 0, 0, 0, 0, 2, 0},
            {0, 0, 0, 0, 0, 0, 0, 18},
            {0, 0, 0, 0, 0, 0, 0, 13},
            {0, 0, 0, 0, 0, 0, 0, 2},
            {0, 0, 0, 0, 0, 0, 0, 0}
        };
        int stages = 4;
        minCost(graph, stages);
    }
}

/// 13) Multistage Graph Problem (Backward Approach) in Java

import java.util.Arrays;

public class MultistageGraphBackward {
    public static int minCost(int[][] graph, int stages) {
        int n = graph.length;
        int[] cost = new int[n];
        int[] path = new int[n];
        Arrays.fill(cost, Integer.MAX_VALUE);
        cost[0] = 0;

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (graph[j][i] != 0 && cost[j] + graph[j][i] < cost[i]) {
                    cost[i] = cost[j] + graph[j][i];
                    path[i] = j;
                }
            }
        }

        System.out.println("Minimum cost: " + cost[n - 1]);
        System.out.print("Path: ");
        int current = n - 1;
        while (current > 0) {
            System.out.print(current + " <- ");
            current = path[current];
        }
        System.out.println(0);
        return cost[n - 1];
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 1, 2, 5, 0, 0, 0, 0},
            {0, 0, 0, 0, 4, 11, 0, 0},
            {0, 0, 0, 0, 9, 5, 16, 0},
            {0, 0, 0, 0, 0, 0, 2, 0},
            {0, 0, 0, 0, 0, 0, 0, 18},
            {0, 0, 0, 0, 0, 0, 0, 13},
            {0, 0, 0, 0, 0, 0, 0, 2},
            {0, 0, 0, 0, 0, 0, 0, 0}
        };
        int stages = 4;
        minCost(graph, stages);
    }
}

/// 14) All-Pairs Shortest Paths (Floyd-Warshall Algorithm) in Java

public class FloydWarshall {
    final static int INF = 99999;

    public static void floydWarshall(int[][] graph) {
        int V = graph.length;
        int[][] dist = new int[V][V];

        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                dist[i][j] = graph[i][j];
            }
        }

        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        printSolution(dist);
    }

    private static void printSolution(int[][] dist) {
        System.out.println("Shortest distances between every pair of vertices:");
        for (int i = 0; i < dist.length; i++) {
            for (int j = 0; j < dist.length; j++) {
                if (dist[i][j] == INF) {
                    System.out.print("INF ");
                } else {
                    System.out.print(dist[i][j] + " ");
                }
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 3, INF, 7},
            {8, 0, 2, INF},
            {5, INF, 0, 1},
            {2, INF, INF, 0}
        };
        floydWarshall(graph);
    }
}

/// 15) 0/1 Knapsack Problem using Dynamic Programming in Java

public class Knapsack {
    public static int knapsack(int W, int[] weights, int[] values, int n) {
        int[][] dp = new int[n + 1][W + 1];

        for (int i = 0; i <= n; i++) {
            for (int w = 0; w <= W; w++) {
                if (i == 0 || w == 0) {
                    dp[i][w] = 0;
                } else if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }

        return dp[n][W];
    }

    public static void main(String[] args) {
        int[] values = {60, 100, 120};
        int[] weights = {10, 20, 30};
        int W = 50;
        int n = values.length;
        System.out.println("Maximum value in knapsack = " + knapsack(W, weights, values, n));
    }
}


/// 16) Travelling Salesperson Problem (TSP) using Dynamic Programming in Java

public class TSP {
    private static final int INF = Integer.MAX_VALUE;

    public static int tsp(int[][] graph) {
        int n = graph.length;
        int VISITED_ALL = (1 << n) - 1;
        int[][] dp = new int[n][1 << n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < (1 << n); j++) {
                dp[i][j] = -1;
            }
        }

        return tspUtil(0, 1, graph, dp, VISITED_ALL);
    }

    private static int tspUtil(int pos, int mask, int[][] graph, int[][] dp, int VISITED_ALL) {
        if (mask == VISITED_ALL) {
            return graph[pos][0] == 0 ? INF : graph[pos][0];
        }

        if (dp[pos][mask] != -1) {
            return dp[pos][mask];
        }

        int ans = INF;
        for (int city = 0; city < graph.length; city++) {
            if ((mask & (1 << city)) == 0 && graph[pos][city] != 0) {
                int newAns = graph[pos][city] + tspUtil(city, mask | (1 << city), graph, dp, VISITED_ALL);
                ans = Math.min(ans, newAns);
            }
        }

        dp[pos][mask] = ans;
        return ans;
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 10, 15, 20},
            {10, 0, 35, 25},
            {15, 35, 0, 30},
            {20, 25, 30, 0}
        };
        System.out.println("Minimum cost of visiting all cities: " + tsp(graph));
    }
}

/// 17) 8-Queen’s Problem using Backtracking in Java

public class NQueens {
    private static final int N = 8;

    private static void printSolution(int[][] board) {
        for (int[] row : board) {
            for (int cell : row) {
                System.out.print(cell + " ");
            }
            System.out.println();
        }
    }

    private static boolean isSafe(int[][] board, int row, int col) {
        for (int i = 0; i < col; i++) {
            if (board[row][i] == 1) return false;
        }

        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 1) return false;
        }

        for (int i = row, j = col; i < N && j >= 0; i++, j--) {
            if (board[i][j] == 1) return false;
        }

        return true;
    }

    private static boolean solveNQueens(int[][] board, int col) {
        if (col >= N) {
            return true;
        }

        for (int i = 0; i < N; i++) {
            if (isSafe(board, i, col)) {
                board[i][col] = 1;
                if (solveNQueens(board, col + 1)) {
                    return true;
                }
                board[i][col] = 0;
            }
        }
        return false;
    }

    public static void main(String[] args) {
        int[][] board = new int[N][N];
        if (solveNQueens(board, 0)) {
            printSolution(board);
        } else {
            System.out.println("No solution exists");
        }
    }
}

/// 18) Sum of Subsets Problem using Backtracking in Java

public class SumOfSubsets {
    private static void findSubsets(int[] set, int target, int currentSum, int index, String currentSet) {
        if (currentSum == target) {
            System.out.println("Subset found: " + currentSet);
            return;
        }

        if (index >= set.length || currentSum > target) {
            return;
        }

        findSubsets(set, target, currentSum + set[index], index + 1, currentSet + " " + set[index]);
        findSubsets(set, target, currentSum, index + 1, currentSet);
    }

    public static void main(String[] args) {
        int[] set = {3, 34, 4, 12, 5, 2};
        int target = 9;
        findSubsets(set, target, 0, 0, "");
    }
}

/// 19) Graph Coloring Problem using Backtracking in Java

public class GraphColoring {
    private static boolean isSafe(int node, int[][] graph, int[] colors, int color) {
        for (int i = 0; i < graph.length; i++) {
            if (graph[node][i] == 1 && colors[i] == color) {
                return false;
            }
        }
        return true;
    }

    private static boolean graphColoringUtil(int[][] graph, int m, int[] colors, int node) {
        if (node == graph.length) {
            return true;
        }

        for (int color = 1; color <= m; color++) {
            if (isSafe(node, graph, colors, color)) {
                colors[node] = color;
                if (graphColoringUtil(graph, m, colors, node + 1)) {
                    return true;
                }
                colors[node] = 0;
            }
        }
        return false;
    }

    public static void graphColoring(int[][] graph, int m) {
        int[] colors = new int[graph.length];
        if (!graphColoringUtil(graph, m, colors, 0)) {
            System.out.println("No solution found");
        } else {
            for (int i = 0; i < graph.length; i++) {
                System.out.println("Vertex " + i + " is colored with color " + colors[i]);
            }
        }
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 1, 1, 1},
            {1, 0, 1, 0},
            {1, 1, 0, 1},
            {1, 0, 1, 0}
        };
        int m = 3; // Number of colors
        graphColoring(graph, m);
    }
}

/// 20) Hamiltonian Cycle Problem using Backtracking in Java

public class HamiltonianCycle {
    private static boolean isSafe(int v, int[][] graph, int[] path, int pos) {
        if (graph[path[pos - 1]][v] == 0) {
            return false;
        }

        for (int i = 0; i < pos; i++) {
            if (path[i] == v) {
                return false;
            }
        }
        return true;
    }

    private static boolean hamCycleUtil(int[][] graph, int[] path, int pos) {
        if (pos == graph.length) {
            return graph[path[pos - 1]][path[0]] == 1;
        }

        for (int v = 1; v < graph.length; v++) {
            if (isSafe(v, graph, path, pos)) {
                path[pos] = v;
                if (hamCycleUtil(graph, path, pos + 1)) {
                    return true;
                }
                path[pos] = -1;
            }
        }
        return false;
    }

    public static void hamCycle(int[][] graph) {
        int[] path = new int[graph.length];
        for (int i = 0; i < path.length; i++) {
            path[i] = -1;
        }

        path[0] = 0;
        if (!hamCycleUtil(graph, path, 1)) {
            System.out.println("No Hamiltonian Cycle found");
        } else {
            System.out.print("Hamiltonian Cycle: ");
            for (int v : path) {
                System.out.print(v + " ");
            }
            System.out.println(path[0]);
        }
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0, 1, 0, 1, 0},
            {1, 0, 1, 1, 1},
            {0, 1, 0, 0, 1},
            {1, 1, 0, 0, 1},
            {0, 1, 1, 1, 0}
        };
        hamCycle(graph);
    }
}