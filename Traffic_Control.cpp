// CSE 2202 - Lab Test 04 (A): Traffic load-balanced routing
// Fixed, self-contained C++17 solution.
//
// BUILD:
//   g++ -O2 -std=c++17 -Wall -Wextra -o traffic traffic_fixed.cpp
//
// RUN ONE INSTANCE:
//   ./traffic gen 20 3 5 30 42 > input.txt
//   ./traffic < input.txt
//   ./traffic --details my_routes.txt < input.txt
//   ./traffic --all-console < input.txt  // also print every route to the console
//
// AUTOMATED SCALABILITY TEST (V = 20, 50, 100, 200, 500, 1000, 2000):
//   ./traffic bench
//   ./traffic bench 3 8 20 42 benchmark_results.csv
//                    T PER NMAX SEED CSV_FILE
//
// INPUT:
//   V E T
//   u v w        (E undirected, positively weighted edges; vertex IDs 0..V-1)
//   k            (number of requests in time step 1)
//   n s d        (k requests: vehicle count, start, destination)
//   k            (number of requests in time step 2)
//   ...          (repeat for all T time steps)
//
// MODEL:
// - Every time step is INDEPENDENT. Loads reset to 0 at the next time step.
// - Load[e] is the number of assigned vehicle routes containing edge e.
// - C is a reference FAIR-SHARE load, not a real physical road capacity.
// - Proposed CABR edge cost: w[e] * (1 + lambda * (load[e]/C)^2).
// - CABR is a HEURISTIC, not a globally optimal algorithm. It retains the
//   best complete assignment encountered under its stated potential function.
//
// COMPLEXITY (one time step; adjacency-list graph + binary-heap Dijkstra):
//   Dijkstra:     O((V+E) log V) time, O(V+E) working memory.
//   Shortest:     O(K (V+E) log V) time.
//   Greedy:       O(N (V+E) log V) time, where N=sum(number of vehicles).
//   CABR:         O(K [2+B(R+1)] (V+E) log V) time, plus sorting.
//                 B <= maxBatches (default 4); R=rerouting rounds (default 3).
//   Graph:        O(V+E) memory. Stored routes additionally use O(total
//                 route-edge occurrences), at most O(K*B*V) for CABR.
//
// OUTPUT:
// - Console summary and metrics for EVERY step, all three methods.
// - Full per-request assignments and ALL per-road loads in traffic_details.txt
//   by default, even when V > 60. Override with --details filename.
// - Bench generates a CSV covering the requested 20..2000 vertex sizes.

#include <bits/stdc++.h>
using namespace std;

struct Net {
    int V = 0;
    vector<int> from, to, w;
    vector<vector<pair<int, int>>> adj; // neighbour, edge ID

    void init(int v) {
        V = v;
        from.clear(); to.clear(); w.clear();
        adj.assign(v, {});
    }
    int edges() const { return static_cast<int>(w.size()); }
    void addEdge(int u, int v, int weight) {
        int id = edges();
        from.push_back(u);
        to.push_back(v);
        w.push_back(weight);
        adj[u].push_back({v, id});
        adj[v].push_back({u, id});
    }
};

struct Req { int n, s, d; };
using Path = vector<int>;              // list of edge IDs
using Assign = vector<map<Path, int>>; // for each request: path -> vehicle count

struct Instance {
    Net graph;
    vector<vector<Req>> steps;
};

[[noreturn]] void fail(const string& message) {
    throw runtime_error(message);
}

// -------------------------- Shortest path with arbitrary positive edge costs
Path dijkstra(const Net& g, int s, int d, const vector<double>& cost) {
    const double INF = numeric_limits<double>::infinity();
    vector<double> dist(g.V, INF);
    vector<int> parent(g.V, -1), parentEdge(g.V, -1);
    using Item = pair<double, int>;
    priority_queue<Item, vector<Item>, greater<Item>> pq;
    dist[s] = 0.0;
    pq.push({0.0, s});

    while (!pq.empty()) {
        auto [best, u] = pq.top();
        pq.pop();
        if (best > dist[u]) continue;
        if (u == d) break;
        for (auto [v, eid] : g.adj[u]) {
            double candidate = best + cost[eid];
            if (candidate < dist[v]) {
                dist[v] = candidate;
                parent[v] = u;
                parentEdge[v] = eid;
                pq.push({candidate, v});
            }
        }
    }
    if (!isfinite(dist[d])) fail("No path exists; the input graph is disconnected.");

    Path result;
    for (int v = d; v != s; v = parent[v]) {
        if (v == -1 || parentEdge[v] == -1) fail("Path reconstruction failed.");
        result.push_back(parentEdge[v]);
    }
    reverse(result.begin(), result.end());
    return result;
}

long long pathLength(const Net& g, const Path& path) {
    long long distance = 0;
    for (int eid : path) distance += g.w[eid];
    return distance;
}

vector<double> originalCosts(const Net& g) {
    return vector<double>(g.w.begin(), g.w.end());
}

// -------------------------- Algorithm 1: all vehicles use the shortest path
Assign shortestPath(const Net& g, const vector<Req>& requests) {
    Assign result(requests.size());
    vector<double> cost = originalCosts(g);
    for (size_t i = 0; i < requests.size(); ++i) {
        const Req& q = requests[i];
        result[i][dijkstra(g, q.s, q.d, cost)] += q.n;
    }
    return result;
}

// -------------------------- Algorithm 2: vehicle-by-vehicle greedy balancing
// Current edge load drives routing; distance acts as a small tie breaker.
Assign greedyLB(const Net& g, const vector<Req>& requests) {
    Assign result(requests.size());
    vector<long long> load(g.edges(), 0);
    vector<double> cost(g.edges());
    for (size_t i = 0; i < requests.size(); ++i) {
        for (int v = 0; v < requests[i].n; ++v) {
            for (int e = 0; e < g.edges(); ++e)
                cost[e] = static_cast<double>(load[e]) + 0.001 * g.w[e];
            Path p = dijkstra(g, requests[i].s, requests[i].d, cost);
            for (int e : p) ++load[e];
            ++result[i][p];
        }
    }
    return result;
}

// This potential has derivative (with respect to load on each road)
// w[e] * (1 + lambda * (load[e]/C)^2), matching the proposed CABR cost.
// Keeping the lowest-potential complete solution avoids ending with a worse
// solution solely because a later rerouting round happened to be unfavorable.
long double routingPotential(const Net& g, const vector<long long>& load,
                             double lambda, double C) {
    long double objective = 0;
    for (int e = 0; e < g.edges(); ++e) {
        long double x = load[e];
        objective += static_cast<long double>(g.w[e]) *
                     (x + static_cast<long double>(lambda) * x*x*x / (3.0L*C*C));
    }
    return objective;
}

// -------------------------- Algorithm 3: congestion-aware batch routing
Assign cabr(const Net& g, const vector<Req>& requests,
            double lambda = 4.0, int rounds = 3, int maxBatches = 4,
            unsigned seed = 0) {
    if (lambda < 0 || rounds < 0 || maxBatches < 1) fail("Invalid CABR parameters.");
    const int K = static_cast<int>(requests.size());
    vector<double> hopCosts(g.edges(), 1.0), baseCosts = originalCosts(g);
    vector<long double> demand(K, 0);
    double totalHopDemand = 0;

    for (int i = 0; i < K; ++i) {
        const Req& q = requests[i];
        totalHopDemand += static_cast<double>(q.n) *
                          dijkstra(g, q.s, q.d, hopCosts).size();
        demand[i] = static_cast<long double>(q.n) *
                    pathLength(g, dijkstra(g, q.s, q.d, baseCosts));
    }
    // Reference load estimated from the shortest-in-hops routes.
    const double C = max(1.0, totalHopDemand / g.edges());
    vector<long long> load(g.edges(), 0);
    vector<double> cost(g.edges());
    Assign assignment(K);

    auto place = [&](int i) {
        // ceil(n/maxBatches) vehicles per batch: at most maxBatches paths.
        const int batchSize = max(1, static_cast<int>(
            (static_cast<long long>(requests[i].n) + maxBatches - 1) / maxBatches));
        int left = requests[i].n;
        while (left > 0) {
            int amount = min(batchSize, left);
            for (int e = 0; e < g.edges(); ++e) {
                double ratio = static_cast<double>(load[e]) / C;
                cost[e] = g.w[e] * (1.0 + lambda * ratio * ratio);
            }
            Path path = dijkstra(g, requests[i].s, requests[i].d, cost);
            for (int e : path) load[e] += amount;
            assignment[i][path] += amount;
            left -= amount;
        }
    };

    auto ripUp = [&](int i) {
        for (const auto& [path, amount] : assignment[i])
            for (int e : path) load[e] -= amount;
        assignment[i].clear();
    };

    vector<int> order(K);
    iota(order.begin(), order.end(), 0);
    stable_sort(order.begin(), order.end(),
                [&](int a, int b) { return demand[a] > demand[b]; });

    for (int i : order) place(i);
    Assign bestAssignment = assignment;
    long double bestPotential = routingPotential(g, load, lambda, C);

    mt19937 rng(seed);
    for (int round = 0; round < rounds; ++round) {
        vector<int> randomOrder = order;
        shuffle(randomOrder.begin(), randomOrder.end(), rng);
        for (int i : randomOrder) {
            ripUp(i);
            place(i);
        }
        long double potential = routingPotential(g, load, lambda, C);
        if (potential < bestPotential) {
            bestPotential = potential;
            bestAssignment = assignment;
        }
    }
    return bestAssignment;
}

// -------------------------- Evaluation and correctness checks
struct Metrics {
    long long maxLoad = 0;
    double averageLoad = 0;
    double variance = 0;
    long long totalDistance = 0;
};

vector<long long> roadLoads(const Net& g, const Assign& assignment) {
    vector<long long> result(g.edges(), 0);
    for (const auto& perRequest : assignment)
        for (const auto& [path, vehicles] : perRequest)
            for (int e : path) result[e] += vehicles;
    return result;
}

Metrics evaluate(const Net& g, const Assign& assignment) {
    vector<long long> load = roadLoads(g, assignment);
    Metrics m;
    m.maxLoad = *max_element(load.begin(), load.end());
    long double sum = 0, varianceSum = 0;
    for (long long x : load) sum += x;
    m.averageLoad = static_cast<double>(sum / g.edges());
    for (long long x : load) {
        long double delta = x - m.averageLoad;
        varianceSum += delta * delta;
    }
    m.variance = static_cast<double>(varianceSum / g.edges());
    for (int e = 0; e < g.edges(); ++e)
        m.totalDistance += load[e] * static_cast<long long>(g.w[e]);
    return m;
}

void validateAssignment(const Net& g, const vector<Req>& requests,
                        const Assign& assignment) {
    if (requests.size() != assignment.size()) fail("Assignment size mismatch.");
    for (size_t i = 0; i < requests.size(); ++i) {
        long long assigned = 0;
        for (const auto& [path, count] : assignment[i]) {
            if (count <= 0) fail("Non-positive assigned vehicle count.");
            assigned += count;
            int at = requests[i].s;
            for (int e : path) {
                if (e < 0 || e >= g.edges()) fail("Invalid edge ID in assigned route.");
                if (at != g.from[e] && at != g.to[e]) fail("Route is not continuous.");
                at = (at == g.from[e] ? g.to[e] : g.from[e]);
            }
            if (at != requests[i].d) fail("Assigned route has wrong destination.");
        }
        if (assigned != requests[i].n) fail("Vehicles lost or duplicated.");
    }
}

void printMetrics(ostream& out, const Metrics& m) {
    out << fixed << setprecision(2)
        << "  max_load=" << m.maxLoad
        << "  avg_load=" << m.averageLoad
        << "  variance=" << m.variance
        << "  total_distance=" << m.totalDistance << '\n';
}

// Detailed output always includes every assigned path and every edge's load.
void printDetails(ostream& out, const Net& g, const vector<Req>& requests,
                  const Assign& assignment) {
    for (size_t i = 0; i < requests.size(); ++i) {
        const Req& q = requests[i];
        out << "  Request " << i + 1 << ": " << q.n << " vehicles, "
            << q.s << " -> " << q.d << '\n';
        vector<pair<int, Path>> routes;
        for (const auto& [path, amount] : assignment[i])
            routes.push_back({amount, path});
        sort(routes.begin(), routes.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });
        for (const auto& [amount, path] : routes) {
            int at = q.s;
            out << "    " << amount << " vehicles: " << at;
            for (int e : path) {
                at = (at == g.from[e] ? g.to[e] : g.from[e]);
                out << " -> " << at;
            }
            out << "  [distance per vehicle=" << pathLength(g, path) << "]\n";
        }
    }
    vector<long long> loads = roadLoads(g, assignment);
    out << "  Per-road loads (including roads with 0 cars):\n";
    for (int e = 0; e < g.edges(); ++e)
        out << "    edge " << e << " (" << g.from[e] << "-" << g.to[e]
            << ", distance=" << g.w[e] << "): load=" << loads[e] << '\n';
}

// -------------------------- Connected random instance generator
Instance makeRandomInstance(int V, int T, int per, int nmax, unsigned seed) {
    if (V < 2 || T < 1 || per < 0 || nmax < 1)
        fail("Generator needs V>=2, T>=1, PER>=0, NMAX>=1.");
    mt19937 rng(seed);
    uniform_int_distribution<int> vertex(0, V-1);
    uniform_int_distribution<int> weight(1, 10);
    uniform_int_distribution<int> vehicles(1, nmax);

    Instance instance;
    Net& g = instance.graph;
    g.init(V);
    vector<int> order(V);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), rng);
    set<pair<int, int>> used;

    auto addUniqueEdge = [&](int a, int b) -> bool {
        if (a == b) return false;
        auto key = minmax(a, b);
        if (!used.insert({key.first, key.second}).second) return false;
        g.addEdge(a, b, weight(rng));
        return true;
    };

    // Random spanning tree: each vertex joins an already-connected vertex.
    for (int i = 1; i < V; ++i) {
        uniform_int_distribution<int> earlier(0, i-1);
        addUniqueEdge(order[i], order[earlier(rng)]);
    }
    // Try to add V additional distinct roads; small graphs may have fewer.
    int target = min(V, V*(V-1)/2 - (V-1));
    for (int added = 0, tries = 0; added < target && tries < 50*V; ++tries)
        if (addUniqueEdge(vertex(rng), vertex(rng))) ++added;

    instance.steps.resize(T);
    for (int t = 0; t < T; ++t) {
        auto& requests = instance.steps[t];
        requests.reserve(per);
        for (int i = 0; i < per; ++i) {
            int s = vertex(rng), d = vertex(rng);
            while (d == s) d = vertex(rng);
            requests.push_back({vehicles(rng), s, d});
        }
    }
    return instance;
}

void printInstance(ostream& out, const Instance& instance) {
    const Net& g = instance.graph;
    out << g.V << ' ' << g.edges() << ' ' << instance.steps.size() << '\n';
    for (int e = 0; e < g.edges(); ++e)
        out << g.from[e] << ' ' << g.to[e] << ' ' << g.w[e] << '\n';
    for (const auto& requests : instance.steps) {
        out << requests.size() << '\n';
        for (const Req& r : requests)
            out << r.n << ' ' << r.s << ' ' << r.d << '\n';
    }
}

Instance readInstance(istream& in) {
    int V, E, T;
    if (!(in >> V >> E >> T)) fail("Missing header: V E T.");
    if (V < 2 || E < V-1 || T < 1)
        fail("Input must have V>=2, E>=V-1 and T>=1.");
    Instance instance;
    instance.graph.init(V);
    for (int e = 0; e < E; ++e) {
        int u, v, w;
        if (!(in >> u >> v >> w)) fail("Could not read all edges.");
        if (u < 0 || u >= V || v < 0 || v >= V || u == v || w <= 0)
            fail("Invalid edge: need distinct vertices in range and weight>0.");
        instance.graph.addEdge(u, v, w);
    }
    vector<bool> visited(V, false);
    queue<int> q;
    visited[0] = true;
    q.push(0);
    int reached = 0;
    while (!q.empty()) {
        int u = q.front(); q.pop(); ++reached;
        for (auto [v, e] : instance.graph.adj[u]) {
            (void)e;
            if (!visited[v]) { visited[v] = true; q.push(v); }
        }
    }
    if (reached != V) fail("Input graph is disconnected.");

    instance.steps.resize(T);
    for (int t = 0; t < T; ++t) {
        int k;
        if (!(in >> k) || k < 0) fail("Invalid number of requests.");
        instance.steps[t].resize(k);
        for (Req& r : instance.steps[t]) {
            if (!(in >> r.n >> r.s >> r.d)) fail("Could not read all requests.");
            if (r.n < 1 || r.s < 0 || r.s >= V || r.d < 0 || r.d >= V || r.s == r.d)
                fail("Invalid request: need n>=1 and distinct s,d in range.");
        }
    }
    return instance;
}

// -------------------------- Execution, summaries and benchmarking
const array<string, 3> ALGORITHM_NAMES = {
    "Shortest-path routing", "Greedy load-balancing", "CABR (proposed)"
};

Assign runAlgorithm(int algorithm, const Net& g, const vector<Req>& requests) {
    if (algorithm == 0) return shortestPath(g, requests);
    if (algorithm == 1) return greedyLB(g, requests);
    return cabr(g, requests);
}

struct Summary {
    array<vector<Metrics>, 3> metrics;
    array<double, 3> seconds{};
};

Summary runAll(const Instance& instance, ostream* details, bool allConsole,
               bool progress = true) {
    const Net& g = instance.graph;
    Summary summary;
    const bool smallGraph = g.V <= 60;

    if (details) {
        *details << "Graph: V=" << g.V << ", E=" << g.edges() << ", T="
                 << instance.steps.size() << "\n"
                 << "Time steps are independent; loads reset each time step.\n\n";
    }
    for (size_t t = 0; t < instance.steps.size(); ++t) {
        if (progress) cout << "\n================ TIME STEP " << (t+1) << " ================\n";
        if (details) *details << "\n================ TIME STEP " << (t+1)
                              << " ================\n";
        for (int a = 0; a < 3; ++a) {
            auto start = chrono::steady_clock::now();
            Assign assignment = runAlgorithm(a, g, instance.steps[t]);
            double elapsed = chrono::duration<double>(chrono::steady_clock::now()-start).count();
            summary.seconds[a] += elapsed;
            validateAssignment(g, instance.steps[t], assignment);
            Metrics m = evaluate(g, assignment);
            summary.metrics[a].push_back(m);

            if (progress) {
                cout << '[' << ALGORITHM_NAMES[a] << "]\n";
                printMetrics(cout, m);
                if (smallGraph || allConsole)
                    printDetails(cout, g, instance.steps[t], assignment);
            }
            if (details) {
                *details << '[' << ALGORITHM_NAMES[a] << "]\n";
                printMetrics(*details, m);
                printDetails(*details, g, instance.steps[t], assignment);
            }
        }
    }
    return summary;
}

struct Average {
    double maxLoad = 0, avgLoad = 0, variance = 0, totalDistance = 0;
};
Average averageMetrics(const vector<Metrics>& results) {
    Average avg;
    for (const Metrics& m : results) {
        avg.maxLoad += m.maxLoad;
        avg.avgLoad += m.averageLoad;
        avg.variance += m.variance;
        avg.totalDistance += m.totalDistance;
    }
    if (!results.empty()) {
        double T = static_cast<double>(results.size());
        avg.maxLoad /= T;
        avg.avgLoad /= T;
        avg.variance /= T;
        avg.totalDistance /= T;
    }
    return avg;
}

void printSummary(ostream& out, const Summary& s) {
    out << "\n================ SUMMARY (means over time steps) ================\n";
    out << left << setw(25) << "Algorithm" << right << setw(11) << "MaxLoad"
        << setw(11) << "AvgLoad" << setw(13) << "Variance"
        << setw(15) << "TotalDist" << setw(13) << "Time(s)" << '\n';
    for (int a = 0; a < 3; ++a) {
        Average m = averageMetrics(s.metrics[a]);
        out << left << setw(25) << ALGORITHM_NAMES[a] << right
            << fixed << setprecision(2) << setw(11) << m.maxLoad
            << setw(11) << m.avgLoad << setw(13) << m.variance
            << setw(15) << m.totalDistance
            << setprecision(5) << setw(13) << s.seconds[a] << '\n';
    }
}

void benchmark(int T, int per, int nmax, unsigned seed, const string& filename) {
    ofstream csv(filename);
    if (!csv) fail("Cannot write benchmark CSV: " + filename);
    csv << "vertices,edges,time_steps,requests_per_step,max_vehicles,algorithm,"
        << "mean_max_load,mean_avg_load,mean_variance,mean_total_distance,total_runtime_seconds\n";

    cout << "Scalability benchmark (same input for each algorithm at each size):\n";
    cout << "V, E, Algorithm, AvgMaxLoad, AvgVariance, AvgTotalDist, Time(s)\n";
    vector<int> sizes = {20, 50, 100, 200, 500, 1000, 2000};
    for (int V : sizes) {
        Instance instance = makeRandomInstance(V, T, per, nmax, seed + V);
        Summary s = runAll(instance, nullptr, false, false);
        for (int a = 0; a < 3; ++a) {
            Average avg = averageMetrics(s.metrics[a]);
            csv << V << ',' << instance.graph.edges() << ',' << T << ',' << per
                << ',' << nmax << ',' << '"' << ALGORITHM_NAMES[a] << '"' << ','
                << fixed << setprecision(6) << avg.maxLoad << ',' << avg.avgLoad
                << ',' << avg.variance << ',' << avg.totalDistance << ','
                << setprecision(9) << s.seconds[a] << '\n';
            cout << V << ", " << instance.graph.edges() << ", " << ALGORITHM_NAMES[a]
                 << ", " << fixed << setprecision(2) << avg.maxLoad << ", "
                 << avg.variance << ", " << avg.totalDistance << ", "
                 << setprecision(5) << s.seconds[a] << '\n';
        }
    }
    cout << "\nBenchmark CSV saved to: " << filename << '\n';
    cout << "Big-O: Dijkstra O((V+E)logV); Shortest O(K*(V+E)logV); "
         << "Greedy O(N*(V+E)logV); "
         << "CABR O(K*(2+B*(R+1))*(V+E)logV).\n"
         << "K=requests/step, N=vehicles/step, B<=4 batches/request, R=3 rounds.\n";
}

void usage(const char* program) {
    cerr << "Usage:\n"
         << "  " << program << " [--details FILE] [--all-console] < input.txt\n"
         << "  " << program << " gen V T PER NMAX SEED > input.txt\n"
         << "  " << program << " bench [T=3] [PER=8] [NMAX=20] [SEED=42] [CSV=benchmark_results.csv]\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc >= 2 && string(argv[1]) == "gen") {
            if (argc != 7) { usage(argv[0]); return 1; }
            int V = stoi(argv[2]), T = stoi(argv[3]), per = stoi(argv[4]);
            int nmax = stoi(argv[5]);
            unsigned seed = static_cast<unsigned>(stoul(argv[6]));
            printInstance(cout, makeRandomInstance(V, T, per, nmax, seed));
            return 0;
        }
        if (argc >= 2 && string(argv[1]) == "bench") {
            if (argc > 7) { usage(argv[0]); return 1; }
            int T = argc > 2 ? stoi(argv[2]) : 3;
            int per = argc > 3 ? stoi(argv[3]) : 8;
            int nmax = argc > 4 ? stoi(argv[4]) : 20;
            unsigned seed = argc > 5 ? static_cast<unsigned>(stoul(argv[5])) : 42;
            string filename = argc > 6 ? argv[6] : "benchmark_results.csv";
            benchmark(T, per, nmax, seed, filename);
            return 0;
        }

        string detailsFile = "traffic_details.txt";
        bool allConsole = false;
        for (int i = 1; i < argc; ++i) {
            string arg = argv[i];
            if (arg == "--details" && i+1 < argc) detailsFile = argv[++i];
            else if (arg == "--all-console") allConsole = true;
            else { usage(argv[0]); return 1; }
        }
        Instance instance = readInstance(cin);
        ofstream details(detailsFile);
        if (!details) fail("Cannot write route details: " + detailsFile);
        Summary result = runAll(instance, &details, allConsole);
        printSummary(cout, result);
        printSummary(details, result);
        cout << "\nComplete routes and every road load saved to: " << detailsFile << '\n';
        return 0;
    } catch (const exception& error) {
        cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
