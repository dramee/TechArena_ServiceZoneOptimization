// TechArena Solver — service zone design.  Single-file C++17 submission.
//
//   objective = sum reward(assigned) - sum distance(office,site)
//               - sum cost(open office) - alpha * (max tasks - min tasks)
//
// Constraints: capacity, |sites(o1)|-|sites(o2)| <= A, workers' skills cover
// every task of every assigned site (a worker belongs to one office), zones
// (convex hull of the office + its sites) must be convex and must not share a
// single instance point.
//
// Core idea -------------------------------------------------------------
// Zones are cut out of a POWER DIAGRAM (weighted Voronoi) built on K free
// generators g_k with weights w_k:  site s -> argmin_k |s-g_k|^2 - w_k.
// All cells are convex and disjoint, so as long as office k's own point also
// falls into cell k, hull(zone k) is contained in cell k and the geometric
// constraint holds BY CONSTRUCTION, whatever we then drop from the zone.
// Generators are free points (not the offices), which makes the family rich
// enough to express arbitrary straight-line splits; this matters because the
// road-distance matrix is not always consistent with the plane geometry.
//
// On top of the partition: profit-based site selection, weighted bipartite
// matching for workers/skills, hard-balance repair, greedy repair of the alpha
// penalty, search over (opened offices, generators, weights), and a final
// exact-hull growth pass that exploits the fact that the rule forbids shared
// POINTS, not overlapping areas.
//
// I/O: instance.json -> solution.json  (paths can be overridden via argv).
// Env TA_TIME: time budget in seconds (default 240; the tour allows 300).

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using Clock_ = chrono::steady_clock;

static Clock_::time_point T0;
static double TIME_LIMIT = 240.0;   // tour limit is 300 s per run
static inline double elapsed() { return chrono::duration<double>(Clock_::now() - T0).count(); }

// ================================================================= JSON ====
struct JV {
	enum T { NUL, NUM, STR, ARR, OBJ } t = NUL;
	double num = 0;
	string str;
	vector<JV> arr;
	vector<pair<string, JV>> obj;
	const JV* get(const char* k) const {
		if (t != OBJ) return nullptr;
		for (const auto& p : obj)
			if (p.first == k) return &p.second;
		return nullptr;
	}
};

struct JParser {
	const string& s;
	size_t i = 0;
	explicit JParser(const string& str) : s(str) {}
	void ws() { while (i < s.size() && (unsigned char)s[i] <= ' ') ++i; }
	JV value() {
		ws();
		if (i >= s.size()) throw runtime_error("eof");
		char c = s[i];
		if (c == '{') return object();
		if (c == '[') return array();
		if (c == '"') return str();
		if (c == 't') { i += 4; JV v; v.t = JV::NUM; v.num = 1; return v; }
		if (c == 'f') { i += 5; JV v; v.t = JV::NUM; v.num = 0; return v; }
		if (c == 'n') { i += 4; return JV(); }
		return num();
	}
	JV num() {
		JV v; v.t = JV::NUM;
		char* end = nullptr;
		v.num = strtod(s.c_str() + i, &end);
		i = (size_t)(end - s.c_str());
		return v;
	}
	JV str() {
		JV v; v.t = JV::STR;
		++i;
		while (i < s.size()) {
			char c = s[i++];
			if (c == '"') return v;
			if (c == '\\' && i < s.size()) v.str.push_back(s[i++]);
			else v.str.push_back(c);
		}
		throw runtime_error("string");
	}
	JV array() {
		JV v; v.t = JV::ARR;
		++i; ws();
		if (i < s.size() && s[i] == ']') { ++i; return v; }
		while (true) {
			v.arr.push_back(value());
			ws();
			if (i >= s.size()) throw runtime_error("array");
			if (s[i] == ']') { ++i; return v; }
			if (s[i] != ',') throw runtime_error("array,");
			++i;
		}
	}
	JV object() {
		JV v; v.t = JV::OBJ;
		++i; ws();
		if (i < s.size() && s[i] == '}') { ++i; return v; }
		while (true) {
			ws();
			JV k = str();
			ws();
			if (i >= s.size() || s[i] != ':') throw runtime_error("obj:");
			++i;
			v.obj.emplace_back(k.str, value());
			ws();
			if (i >= s.size()) throw runtime_error("obj");
			if (s[i] == '}') { ++i; return v; }
			if (s[i] != ',') throw runtime_error("obj,");
			++i;
		}
	}
};

// ============================================================= instance ====
struct Instance {
	int n = 0, m = 0, nw = 0, nsk = 0;
	vector<double> sx, sy, rew;
	vector<int> stc;                    // task count = load contribution
	vector<vector<int>> ssk;            // distinct skills required
	vector<char> sok;                   // servable at all
	vector<double> ox, oy, ocost;
	vector<int> omax;
	vector<vector<int>> wsk;            // worker -> skills
	vector<vector<int>> skillWorkers;   // skill -> workers
	vector<vector<double>> dist;        // [office][site]
	double alpha = 0;
	long long A = 0;
};
static Instance INS;
static double WSCALE = 1.0;             // typical squared length scale

static void loadInstance(const char* path) {
	ifstream f(path, ios::binary);
	if (!f) throw runtime_error("cannot open instance");
	string text((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
	if (text.size() >= 3 && (unsigned char)text[0] == 0xEF) text.erase(0, 3);
	JParser p(text);
	JV root = p.value();
	const JV* sites = root.get("sites");
	const JV* offices = root.get("offices");
	const JV* workers = root.get("workers");
	const JV* distances = root.get("distances");
	const JV* maxDiff = root.get("maxDifference");
	const JV* balance = root.get("balanceCoefficient");

	Instance& I = INS;
	I.n = sites ? (int)sites->arr.size() : 0;
	I.m = offices ? (int)offices->arr.size() : 0;
	I.nw = workers ? (int)workers->arr.size() : 0;
	I.alpha = balance ? balance->num : 0.0;
	I.A = maxDiff ? (long long)llround(maxDiff->num) : 0;

	map<int, int> remap;
	I.wsk.assign(I.nw, {});
	for (int w = 0; w < I.nw; ++w) {
		for (const JV& v : workers->arr[w].arr) {
			int raw = (int)llround(v.num);
			auto it = remap.find(raw);
			int id;
			if (it == remap.end()) { id = (int)remap.size(); remap[raw] = id; }
			else id = it->second;
			I.wsk[w].push_back(id);
		}
		sort(I.wsk[w].begin(), I.wsk[w].end());
		I.wsk[w].erase(unique(I.wsk[w].begin(), I.wsk[w].end()), I.wsk[w].end());
	}
	I.nsk = (int)remap.size();
	I.skillWorkers.assign(I.nsk, {});
	for (int w = 0; w < I.nw; ++w)
		for (int k : I.wsk[w]) I.skillWorkers[k].push_back(w);

	I.sx.resize(I.n); I.sy.resize(I.n); I.rew.resize(I.n);
	I.stc.assign(I.n, 0); I.ssk.assign(I.n, {}); I.sok.assign(I.n, 1);
	for (int s = 0; s < I.n; ++s) {
		const JV& o = sites->arr[s];
		I.sx[s] = o.get("x")->num;
		I.sy[s] = o.get("y")->num;
		I.rew[s] = o.get("reward")->num;
		if (const JV* tk = o.get("tasks")) {
			I.stc[s] = (int)tk->arr.size();
			for (const JV& v : tk->arr) {
				auto it = remap.find((int)llround(v.num));
				if (it == remap.end()) { I.sok[s] = 0; continue; }
				I.ssk[s].push_back(it->second);
			}
			sort(I.ssk[s].begin(), I.ssk[s].end());
			I.ssk[s].erase(unique(I.ssk[s].begin(), I.ssk[s].end()), I.ssk[s].end());
		}
	}
	I.ox.resize(I.m); I.oy.resize(I.m); I.ocost.resize(I.m); I.omax.assign(I.m, 0);
	for (int o = 0; o < I.m; ++o) {
		const JV& v = offices->arr[o];
		I.ox[o] = v.get("x")->num;
		I.oy[o] = v.get("y")->num;
		I.ocost[o] = v.get("cost")->num;
		I.omax[o] = (int)llround(v.get("maxSize")->num);
	}
	I.dist.assign(I.m, vector<double>(I.n, 0.0));
	if (distances)
		for (int o = 0; o < I.m && o < (int)distances->arr.size(); ++o) {
			const JV& row = distances->arr[o];
			for (int s = 0; s < I.n && s < (int)row.arr.size(); ++s) I.dist[o][s] = row.arr[s].num;
		}
}

static inline double siteValue(int o, int s) { return INS.rew[s] - INS.dist[o][s]; }

// ============================================================= solution ====
struct Zone {
	int office = -1;
	vector<pair<double, int>> sel, rest;   // (value, site), sorted by value desc
	vector<int> workers;
	double sumV = 0;
	long long load = 0;
};

struct Solution {
	vector<Zone> zones;
	double obj = 0;
	bool guaranteed = true;   // office_k inside cell_k for every k => geometry is safe
};

struct Config {                 // partition description
	vector<int> off;            // opened offices
	vector<double> gx, gy, w;   // generators and weights, one per entry of off
};

static void recomputeZone(Zone& z) {
	z.sumV = 0; z.load = 0;
	for (auto& pr : z.sel) { z.sumV += pr.first; z.load += INS.stc[pr.second]; }
}

static double objectiveOf(const vector<Zone>& z) {
	if (z.empty()) return 0.0;
	double v = 0;
	long long mx = LLONG_MIN, mn = LLONG_MAX;
	for (const Zone& q : z) {
		v += q.sumV - INS.ocost[q.office];
		mx = max(mx, q.load);
		mn = min(mn, q.load);
	}
	return v - INS.alpha * (double)(mx - mn);
}

static long long imbalanceOf(const vector<Zone>& z) {
	if (z.empty()) return 0;
	long long mx = LLONG_MIN, mn = LLONG_MAX;
	for (const Zone& q : z) { mx = max(mx, q.load); mn = min(mn, q.load); }
	return mx - mn;
}

// Demands (zone, skill) matched to distinct workers (exact when workers are
// single-skilled, conservative otherwise); greedy by weight is optimal here.
struct Matcher {
	vector<int> matchWorker;
	vector<char> used;
	vector<int> demandSkill;
	bool tryKuhn(int d) {
		for (int w : INS.skillWorkers[demandSkill[d]]) {
			if (used[w]) continue;
			used[w] = 1;
			if (matchWorker[w] < 0 || tryKuhn(matchWorker[w])) { matchWorker[w] = d; return true; }
		}
		return false;
	}
};

struct Evaluator {
	bool lastGuaranteed = true;
	// ---- stage 1: partition -> selection -------------------------------
	vector<Zone> stage1(const Config& c) {
		int K = (int)c.off.size();
		vector<Zone> z;
		if (K == 0) return z;

		// if every office lies inside its own cell, hull(zone) is contained in
		// the convex cell and the geometric constraint cannot be violated
		lastGuaranteed = true;
		for (int k = 0; k < K; ++k) {
			double px = INS.ox[c.off[k]], py = INS.oy[c.off[k]];
			double own = (px - c.gx[k]) * (px - c.gx[k]) + (py - c.gy[k]) * (py - c.gy[k]) - c.w[k];
			for (int j = 0; j < K; ++j) {
				if (j == k) continue;
				double o = (px - c.gx[j]) * (px - c.gx[j]) + (py - c.gy[j]) * (py - c.gy[j]) - c.w[j];
				if (o < own - 1e-12 * WSCALE) { lastGuaranteed = false; break; }
			}
			if (!lastGuaranteed) break;
		}

		vector<vector<pair<double, int>>> cand(K);
		for (int s = 0; s < INS.n; ++s) {
			if (!INS.sok[s]) continue;
			int best = -1;
			double bestP = 0;
			for (int k = 0; k < K; ++k) {
				double dx = INS.sx[s] - c.gx[k], dy = INS.sy[s] - c.gy[k];
				double p = dx * dx + dy * dy - c.w[k];
				if (best < 0 || p < bestP) { best = k; bestP = p; }
			}
			cand[best].push_back({siteValue(c.off[best], s), s});
		}
		for (int k = 0; k < K; ++k)
			sort(cand[k].begin(), cand[k].end(),
			     [](const pair<double, int>& a, const pair<double, int>& b) { return a.first > b.first; });

		vector<vector<char>> allowed(K, vector<char>(INS.nsk, 1));
		z.assign(K, Zone());
		for (int k = 0; k < K; ++k) z[k].office = c.off[k];

		for (int iter = 0; iter < INS.nsk * K + 2; ++iter) {
			for (int k = 0; k < K; ++k) {
				z[k].sel.clear(); z[k].rest.clear();
				int cap = INS.omax[c.off[k]];
				for (auto& pr : cand[k]) {
					bool ok = pr.first > 0 && (int)z[k].sel.size() < cap;
					if (ok)
						for (int sk : INS.ssk[pr.second])
							if (!allowed[k][sk]) { ok = false; break; }
					if (ok) z[k].sel.push_back(pr);
					else z[k].rest.push_back(pr);
				}
				recomputeZone(z[k]);
			}
			vector<pair<double, pair<int, int>>> dem;
			vector<vector<double>> wgt(K, vector<double>(INS.nsk, 0.0));
			for (int k = 0; k < K; ++k)
				for (auto& pr : z[k].sel)
					for (int sk : INS.ssk[pr.second]) wgt[k][sk] += max(1e-9, pr.first);
			for (int k = 0; k < K; ++k)
				for (int sk = 0; sk < INS.nsk; ++sk)
					if (wgt[k][sk] > 0) dem.push_back({wgt[k][sk], {k, sk}});
			if (dem.empty()) break;
			sort(dem.begin(), dem.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

			Matcher M;
			M.matchWorker.assign(INS.nw, -1);
			M.demandSkill.resize(dem.size());
			for (size_t d = 0; d < dem.size(); ++d) M.demandSkill[d] = dem[d].second.second;
			bool changed = false;
			for (size_t d = 0; d < dem.size(); ++d) {
				M.used.assign(INS.nw, 0);
				if (!M.tryKuhn((int)d)) { allowed[dem[d].second.first][dem[d].second.second] = 0; changed = true; }
			}
			if (!changed) {
				for (int k = 0; k < K; ++k) z[k].workers.clear();
				for (int w = 0; w < INS.nw; ++w)
					if (M.matchWorker[w] >= 0) z[dem[M.matchWorker[w]].second.first].workers.push_back(w);
				break;
			}
		}
		return z;
	}

	// ---- stage 2: prune, repair balances, score ------------------------
	Solution finish(vector<Zone> z) {
		Solution sol;
		{
			vector<Zone> keep;
			for (auto& q : z)
				if (!q.sel.empty() && q.sumV - INS.ocost[q.office] > 0) keep.push_back(q);
			z.swap(keep);
		}
		if (z.empty()) return sol;
		repairHardBalance(z);
		if (z.empty()) return sol;
		greedySoftBalance(z);
		closeLosers(z);
		if (z.empty()) return sol;
		sol.zones = z;
		sol.obj = objectiveOf(z);
		if (sol.obj <= 0) { sol.zones.clear(); sol.obj = 0; }
		return sol;
	}

	Solution build(const Config& c) {
		vector<Zone> z = stage1(c);
		bool g = lastGuaranteed;
		Solution s = finish(move(z));
		s.guaranteed = g || s.zones.size() < 2;
		return s;
	}

	void repairHardBalance(vector<Zone>& z) {
		while (z.size() > 1) {
			size_t big = 0, small = 0;
			for (size_t k = 0; k < z.size(); ++k) {
				if (z[k].sel.size() > z[big].sel.size()) big = k;
				if (z[k].sel.size() < z[small].sel.size()) small = k;
			}
			if ((long long)z[big].sel.size() - (long long)z[small].sel.size() <= INS.A) return;
			double costDrop = z[big].sel.empty() ? 1e300 : z[big].sel.back().first;
			double costClose = z[small].sumV - INS.ocost[z[small].office];
			if (costClose <= costDrop) {
				z.erase(z.begin() + small);
			} else {
				auto pr = z[big].sel.back();
				z[big].sel.pop_back();
				z[big].rest.insert(z[big].rest.begin(), pr);
				recomputeZone(z[big]);
				if (z[big].sel.empty()) z.erase(z.begin() + big);
			}
		}
	}

	static bool zoneCanServe(const Zone& z, int s) {
		for (int sk : INS.ssk[s]) {
			bool f = false;
			for (int w : z.workers)
				if (binary_search(INS.wsk[w].begin(), INS.wsk[w].end(), sk)) { f = true; break; }
			if (!f) return false;
		}
		return true;
	}

	void greedySoftBalance(vector<Zone>& z) {
		if (z.size() < 2 || INS.alpha <= 0) return;
		int guard = 4 * INS.n + 50;
		while (guard-- > 0) {
			long long imb = imbalanceOf(z);
			if (imb == 0) break;
			size_t kmax = 0, kmin = 0;
			for (size_t k = 0; k < z.size(); ++k) {
				if (z[k].load > z[kmax].load) kmax = k;
				if (z[k].load < z[kmin].load) kmin = k;
			}
			double bestDelta = 1e-9;
			int bestMove = -1, idx = -1;
			// (a) drop the cheapest loaded site of the most loaded zone
			for (int i = (int)z[kmax].sel.size() - 1; i >= 0; --i) {
				int s = z[kmax].sel[i].second;
				if (INS.stc[s] == 0) continue;
				if (z[kmax].sel.size() <= 1) break;
				long long mnC = LLONG_MAX, mxC = LLONG_MIN;
				for (size_t k = 0; k < z.size(); ++k) {
					long long cc = (long long)z[k].sel.size() - (k == kmax ? 1 : 0);
					mnC = min(mnC, cc); mxC = max(mxC, cc);
				}
				if (mxC - mnC > INS.A) break;
				long long save = z[kmax].load;
				z[kmax].load -= INS.stc[s];
				long long ni = imbalanceOf(z);
				z[kmax].load = save;
				double delta = -z[kmax].sel[i].first + INS.alpha * (double)(imb - ni);
				if (delta > bestDelta) { bestDelta = delta; bestMove = 0; idx = i; }
				break;
			}
			// (b) add a site to the least loaded zone
			{
				int cap = INS.omax[z[kmin].office];
				for (size_t j = 0; j < z[kmin].rest.size() && j < 10; ++j) {
					int s = z[kmin].rest[j].second;
					if (INS.stc[s] == 0) continue;
					if ((int)z[kmin].sel.size() >= cap) break;
					if (!zoneCanServe(z[kmin], s)) continue;
					long long mnC = LLONG_MAX, mxC = LLONG_MIN;
					for (size_t k = 0; k < z.size(); ++k) {
						long long cc = (long long)z[k].sel.size() + (k == kmin ? 1 : 0);
						mnC = min(mnC, cc); mxC = max(mxC, cc);
					}
					if (mxC - mnC > INS.A) break;
					long long save = z[kmin].load;
					z[kmin].load += INS.stc[s];
					long long ni = imbalanceOf(z);
					z[kmin].load = save;
					double delta = z[kmin].rest[j].first + INS.alpha * (double)(imb - ni);
					if (delta > bestDelta) { bestDelta = delta; bestMove = 1; idx = (int)j; }
					break;
				}
			}
			if (bestMove < 0) break;
			if (bestMove == 0) {
				auto pr = z[kmax].sel[idx];
				z[kmax].sel.erase(z[kmax].sel.begin() + idx);
				z[kmax].rest.insert(z[kmax].rest.begin(), pr);
				recomputeZone(z[kmax]);
				if (z[kmax].sel.empty()) { z.erase(z.begin() + kmax); if (z.size() < 2) break; }
			} else {
				auto pr = z[kmin].rest[idx];
				z[kmin].rest.erase(z[kmin].rest.begin() + idx);
				z[kmin].sel.push_back(pr);
				sort(z[kmin].sel.begin(), z[kmin].sel.end(),
				     [](const pair<double, int>& a, const pair<double, int>& b) { return a.first > b.first; });
				recomputeZone(z[kmin]);
			}
		}
	}

	void closeLosers(vector<Zone>& z) {
		bool improved = true;
		while (improved && !z.empty()) {
			improved = false;
			double cur = objectiveOf(z);
			for (size_t k = 0; k < z.size(); ++k) {
				vector<Zone> t = z;
				t.erase(t.begin() + k);
				bool ok = true;
				if (t.size() > 1) {
					size_t mn = 0, mx = 0;
					for (size_t j = 0; j < t.size(); ++j) {
						if (t[j].sel.size() < t[mn].sel.size()) mn = j;
						if (t[j].sel.size() > t[mx].sel.size()) mx = j;
					}
					ok = ((long long)t[mx].sel.size() - (long long)t[mn].sel.size() <= INS.A);
				}
				if (ok && objectiveOf(t) > cur + 1e-9) { z.swap(t); improved = true; break; }
			}
		}
	}

	// weights pushed towards equal loads / counts (capacity-constrained
	// power diagram), which is what alpha and the hard A-constraint want
	Solution tune(Config& c, int iters, double deadline) {
		Solution best = build(c);
		int K = (int)c.off.size();
		if (K < 2) return best;
		Config cur = c, bestC = c;
		double lr = 0.5;
		for (int t = 0; t < iters; ++t) {
			if (elapsed() > deadline) break;
			vector<Zone> z = stage1(cur);
			double meanL = 0, meanC = 0;
			for (auto& q : z) { meanL += (double)q.load; meanC += (double)q.sel.size(); }
			meanL /= K; meanC /= K;
			for (int k = 0; k < K; ++k) {
				double dl = meanL > 1e-9 ? ((double)z[k].load - meanL) / meanL : 0.0;
				double dc = meanC > 1e-9 ? ((double)z[k].sel.size() - meanC) / meanC : 0.0;
				cur.w[k] -= lr * (0.5 * dl + 0.5 * dc) * WSCALE;
			}
			bool g = lastGuaranteed;
			Solution s = finish(z);
			s.guaranteed = g || s.zones.size() < 2;
			if (s.obj > best.obj + 1e-9) { best = s; bestC = cur; }
			Solution s2 = build(cur);
			if (s2.obj > best.obj + 1e-9) { best = s2; bestC = cur; }
			lr *= 0.9;
		}
		c = bestC;
		return best;
	}
};

// ============================================================= geometry ====
struct Pt { double x, y; };
static vector<Pt> ALLPTS;      // n sites, then m offices
static double GEPS = 1e-12;

static vector<Pt> convexHull(vector<Pt> p) {
	sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
	p.erase(unique(p.begin(), p.end(), [](const Pt& a, const Pt& b) { return a.x == b.x && a.y == b.y; }), p.end());
	int n = (int)p.size();
	if (n <= 2) return p;
	auto cross = [](const Pt& o, const Pt& a, const Pt& b) {
		return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
	};
	vector<Pt> h(2 * n);
	int k = 0;
	for (int i = 0; i < n; ++i) {
		while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0) --k;
		h[k++] = p[i];
	}
	int lower = k + 1;
	for (int i = n - 2; i >= 0; --i) {
		while (k >= lower && cross(h[k - 2], h[k - 1], p[i]) <= 0) --k;
		h[k++] = p[i];
	}
	h.resize(k - 1);
	if ((int)h.size() < 3) return {p.front(), p.back()};
	return h;
}

static bool hullCovers(const vector<Pt>& h, const Pt& q, double eps) {
	if (h.empty()) return false;
	if (h.size() == 1) return fabs(h[0].x - q.x) <= eps && fabs(h[0].y - q.y) <= eps;
	if (h.size() == 2) {
		double cr = (h[1].x - h[0].x) * (q.y - h[0].y) - (h[1].y - h[0].y) * (q.x - h[0].x);
		double sc = max(1.0, fabs(h[1].x - h[0].x) + fabs(h[1].y - h[0].y));
		if (fabs(cr) > eps * sc) return false;
		return q.x >= min(h[0].x, h[1].x) - eps && q.x <= max(h[0].x, h[1].x) + eps &&
		       q.y >= min(h[0].y, h[1].y) - eps && q.y <= max(h[0].y, h[1].y) + eps;
	}
	size_t n = h.size();
	for (size_t i = 0; i < n; ++i) {
		const Pt& a = h[i];
		const Pt& b = h[(i + 1) % n];
		double cr = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x);
		double sc = max(1.0, fabs(b.x - a.x) + fabs(b.y - a.y));
		if (cr < -eps * sc) return false;
	}
	return true;
}

static vector<char> coveredMask(const Zone& z) {
	vector<Pt> pts;
	pts.reserve(z.sel.size() + 1);
	for (auto& pr : z.sel) pts.push_back(ALLPTS[pr.second]);
	pts.push_back(ALLPTS[INS.n + z.office]);
	vector<Pt> h = convexHull(pts);
	vector<char> cov(ALLPTS.size(), 0);
	for (size_t i = 0; i < ALLPTS.size(); ++i) cov[i] = hullCovers(h, ALLPTS[i], GEPS) ? 1 : 0;
	return cov;
}

static bool feasible(const vector<Zone>& z) {
	vector<char> usedS(INS.n, 0), usedW(INS.nw, 0), usedO(INS.m, 0);
	for (const Zone& q : z) {
		if (q.office < 0 || q.office >= INS.m || usedO[q.office]) return false;
		usedO[q.office] = 1;
		if ((int)q.sel.size() > INS.omax[q.office]) return false;
		for (auto& pr : q.sel) {
			int s = pr.second;
			if (s < 0 || s >= INS.n || usedS[s]) return false;
			usedS[s] = 1;
		}
		for (int w : q.workers) {
			if (w < 0 || w >= INS.nw || usedW[w]) return false;
			usedW[w] = 1;
		}
		for (auto& pr : q.sel)
			if (!Evaluator::zoneCanServe(q, pr.second)) return false;
	}
	for (const Zone& a : z)
		for (const Zone& b : z)
			if ((long long)a.sel.size() - (long long)b.sel.size() > INS.A) return false;
	vector<vector<char>> cov;
	for (const Zone& q : z) cov.push_back(coveredMask(q));
	for (size_t a = 0; a < cov.size(); ++a)
		for (size_t b = a + 1; b < cov.size(); ++b)
			for (size_t i = 0; i < ALLPTS.size(); ++i)
				if (cov[a][i] && cov[b][i]) return false;
	return true;
}

// The rule forbids shared POINTS, not overlapping areas: grow zones with sites
// the partition left out, checking the exact hull condition each time.
static void polish(Solution& sol, double deadline) {
	if (sol.zones.empty()) return;
	vector<char> assigned(INS.n, 0);
	for (const Zone& q : sol.zones)
		for (auto& pr : q.sel) assigned[pr.second] = 1;
	vector<vector<char>> cov;
	for (const Zone& q : sol.zones) cov.push_back(coveredMask(q));

	vector<pair<double, pair<int, int>>> cands;
	for (int s = 0; s < INS.n; ++s) {
		if (assigned[s] || !INS.sok[s]) continue;
		for (size_t k = 0; k < sol.zones.size(); ++k) {
			double v = siteValue(sol.zones[k].office, s);
			if (v > 0) cands.push_back({v, {s, (int)k}});
		}
	}
	sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

	for (auto& c : cands) {
		if (elapsed() > deadline) break;
		int s = c.second.first, k = c.second.second;
		if (assigned[s]) continue;
		Zone& z = sol.zones[k];
		if ((int)z.sel.size() >= INS.omax[z.office]) continue;
		if (!Evaluator::zoneCanServe(z, s)) continue;
		long long mn = LLONG_MAX, mx = LLONG_MIN;
		for (size_t j = 0; j < sol.zones.size(); ++j) {
			long long cnt = (long long)sol.zones[j].sel.size() + (j == (size_t)k ? 1 : 0);
			mn = min(mn, cnt); mx = max(mx, cnt);
		}
		if (mx - mn > INS.A) continue;
		vector<Zone> t = sol.zones;
		t[k].sel.push_back({c.first, s});
		recomputeZone(t[k]);
		if (objectiveOf(t) <= sol.obj + 1e-9) continue;
		vector<Pt> pts;
		for (auto& pr : t[k].sel) pts.push_back(ALLPTS[pr.second]);
		pts.push_back(ALLPTS[INS.n + t[k].office]);
		vector<Pt> h = convexHull(pts);
		vector<char> nc(ALLPTS.size(), 0);
		bool bad = false;
		for (size_t i = 0; i < ALLPTS.size() && !bad; ++i) {
			if (!hullCovers(h, ALLPTS[i], GEPS)) continue;
			nc[i] = 1;
			for (size_t j = 0; j < sol.zones.size(); ++j)
				if (j != (size_t)k && cov[j][i]) { bad = true; break; }
		}
		if (bad) continue;
		sort(t[k].sel.begin(), t[k].sel.end(),
		     [](const pair<double, int>& a, const pair<double, int>& b) { return a.first > b.first; });
		sol.zones.swap(t);
		sol.obj = objectiveOf(sol.zones);
		cov[k].swap(nc);
		assigned[s] = 1;
	}
}


// ---------------------------------------------------------------------------
// Local search on the real constraint.  Dropping a site can never create a
// hull conflict (hulls only shrink), so only growing moves are checked, and
// they are checked exactly.
struct LocalSearch {
	vector<Zone>* Z;
	vector<vector<char>> cov;
	vector<char> freeWorker;

	void init(vector<Zone>& z) {
		Z = &z;
		cov.clear();
		for (const Zone& q : z) cov.push_back(coveredMask(q));
		freeWorker.assign(INS.nw, 1);
		for (const Zone& q : z)
			for (int w : q.workers) freeWorker[w] = 0;
	}

	// can zone k serve site s, possibly by hiring an idle worker?
	bool canServe(int k, int s, vector<int>& hire) {
		hire.clear();
		for (int sk : INS.ssk[s]) {
			bool f = false;
			for (int w : (*Z)[k].workers)
				if (binary_search(INS.wsk[w].begin(), INS.wsk[w].end(), sk)) { f = true; break; }
			for (int w : hire)
				if (!f && binary_search(INS.wsk[w].begin(), INS.wsk[w].end(), sk)) f = true;
			if (f) continue;
			int pick = -1;
			for (int w : INS.skillWorkers[sk])
				if (freeWorker[w]) { pick = w; break; }
			if (pick < 0) return false;
			hire.push_back(pick);
		}
		return true;
	}

	static long long imbOf(const vector<long long>& L) {
		long long mx = LLONG_MIN, mn = LLONG_MAX;
		for (long long v : L) { mx = max(mx, v); mn = min(mn, v); }
		return L.empty() ? 0 : mx - mn;
	}

	// exact check for a zone that gained a site
	bool growOk(int k, int s, vector<char>& newCov) {
		vector<Zone>& z = *Z;
		vector<Pt> pts;
		pts.reserve(z[k].sel.size() + 2);
		for (auto& pr : z[k].sel) pts.push_back(ALLPTS[pr.second]);
		pts.push_back(ALLPTS[s]);
		pts.push_back(ALLPTS[INS.n + z[k].office]);
		vector<Pt> h = convexHull(pts);
		newCov.assign(ALLPTS.size(), 0);
		for (size_t i = 0; i < ALLPTS.size(); ++i) {
			if (!hullCovers(h, ALLPTS[i], GEPS)) continue;
			newCov[i] = 1;
			for (size_t j = 0; j < z.size(); ++j)
				if (j != (size_t)k && cov[j][i]) return false;
		}
		return true;
	}

	void refreshCov(int k) { cov[k] = coveredMask((*Z)[k]); }

	void run(double deadline, mt19937_64& rng) {
		vector<Zone>& z = *Z;
		if (z.empty()) return;
		int K = (int)z.size();
		vector<int> owner(INS.n, -1);
		for (int k = 0; k < K; ++k)
			for (auto& pr : z[k].sel) owner[pr.second] = k;

		vector<int> order(INS.n);
		iota(order.begin(), order.end(), 0);
		bool improved = true;
		int pass = 0;
		while (improved && elapsed() < deadline && pass++ < 40) {
			improved = false;
			shuffle(order.begin(), order.end(), rng);
			for (int s : order) {
				if (elapsed() > deadline) break;
				if (!INS.sok[s]) continue;
				int a = owner[s];
				vector<long long> L(K);
				vector<long long> C(K);
				for (int k = 0; k < K; ++k) { L[k] = z[k].load; C[k] = (long long)z[k].sel.size(); }
				long long imb0 = imbOf(L);
				double va = (a >= 0) ? siteValue(z[a].office, s) : 0.0;

				// --- drop (always geometrically safe) ---
				if (a >= 0 && z[a].sel.size() > 1) {
					L[a] -= INS.stc[s]; C[a] -= 1;
					long long mnC = LLONG_MAX, mxC = LLONG_MIN;
					for (int k = 0; k < K; ++k) { mnC = min(mnC, C[k]); mxC = max(mxC, C[k]); }
					double delta = -va + INS.alpha * (double)(imb0 - imbOf(L));
					L[a] += INS.stc[s]; C[a] += 1;
					if (delta > 1e-9 && mxC - mnC <= INS.A) {
						removeSite(a, s);
						owner[s] = -1;
						refreshCov(a);
						improved = true;
						continue;
					}
				}

				// --- add / move ---
				int bestB = -1;
				double bestDelta = 1e-9;
				for (int b = 0; b < K; ++b) {
					if (b == a) continue;
					if ((int)z[b].sel.size() >= INS.omax[z[b].office]) continue;
					double vb = siteValue(z[b].office, s);
					if (vb <= 0 && INS.alpha <= 0) continue;
					long long sl = L[b], sa = (a >= 0) ? L[a] : 0;
					L[b] += INS.stc[s];
					if (a >= 0) L[a] -= INS.stc[s];
					long long ni = imbOf(L);
					L[b] = sl;
					if (a >= 0) L[a] = sa;
					long long mnC = LLONG_MAX, mxC = LLONG_MIN;
					for (int k = 0; k < K; ++k) {
						long long cc = C[k] + (k == b ? 1 : 0) - (k == a ? 1 : 0);
						mnC = min(mnC, cc); mxC = max(mxC, cc);
					}
					if (mxC - mnC > INS.A) continue;
					double delta = vb - va + INS.alpha * (double)(imb0 - ni);
					if (delta > bestDelta) { bestDelta = delta; bestB = b; }
				}
				if (bestB < 0) continue;
				vector<int> hire;
				if (!canServe(bestB, s, hire)) continue;
				vector<char> nc;
				if (a >= 0) {                       // tentatively remove first
					removeSite(a, s);
					refreshCov(a);
				}
				if (!growOk(bestB, s, nc)) {
					if (a >= 0) { addSite(a, s); refreshCov(a); }
					continue;
				}
				addSite(bestB, s);
				for (int w : hire) { z[bestB].workers.push_back(w); freeWorker[w] = 0; }
				cov[bestB].swap(nc);
				owner[s] = bestB;
				improved = true;
			}
		}
	}

	void removeSite(int k, int s) {
		Zone& q = (*Z)[k];
		for (size_t i = 0; i < q.sel.size(); ++i)
			if (q.sel[i].second == s) { q.sel.erase(q.sel.begin() + i); break; }
		recomputeZone(q);
	}
	void addSite(int k, int s) {
		Zone& q = (*Z)[k];
		q.sel.push_back({siteValue(q.office, s), s});
		sort(q.sel.begin(), q.sel.end(),
		     [](const pair<double, int>& x, const pair<double, int>& y) { return x.first > y.first; });
		recomputeZone(q);
	}
};

// =============================================================== output ====
static string OUTPATH = "solution.json";
static void writeSolution(const vector<Zone>& z) {
	ostringstream out;
	out << "[\n";
	for (size_t k = 0; k < z.size(); ++k) {
		vector<int> sites;
		for (auto& pr : z[k].sel) sites.push_back(pr.second);
		sort(sites.begin(), sites.end());
		vector<int> ws = z[k].workers;
		sort(ws.begin(), ws.end());
		out << "\t{\n\t\t\"officeId\": " << z[k].office << ",\n\t\t\"sites\": [";
		for (size_t i = 0; i < sites.size(); ++i) { if (i) out << ", "; out << sites[i]; }
		out << "],\n\t\t\"workers\": [";
		for (size_t i = 0; i < ws.size(); ++i) { if (i) out << ", "; out << ws[i]; }
		out << "]\n\t}";
		if (k + 1 < z.size()) out << ",";
		out << "\n";
	}
	out << "]\n";
	ofstream f(OUTPATH, ios::binary | ios::trunc);
	f << out.str();
	f.flush();
}

// ================================================================= main ====
static Config makeConfig(const vector<int>& off, int mode) {
	Config c;
	c.off = off;
	int K = (int)off.size();
	c.gx.resize(K); c.gy.resize(K); c.w.assign(K, 0.0);
	if (mode == 0) {                       // generators = offices
		for (int k = 0; k < K; ++k) { c.gx[k] = INS.ox[off[k]]; c.gy[k] = INS.oy[off[k]]; }
		return c;
	}
	// generators = centroid of the sites each office wins on value
	vector<double> sx(K, 0), sy(K, 0);
	vector<int> cnt(K, 0);
	for (int s = 0; s < INS.n; ++s) {
		if (!INS.sok[s]) continue;
		int best = 0;
		double bv = -1e300;
		for (int k = 0; k < K; ++k) {
			double v = siteValue(off[k], s);
			if (v > bv) { bv = v; best = k; }
		}
		if (bv <= 0) continue;
		sx[best] += INS.sx[s]; sy[best] += INS.sy[s]; cnt[best]++;
	}
	for (int k = 0; k < K; ++k) {
		if (cnt[k] > 0) { c.gx[k] = sx[k] / cnt[k]; c.gy[k] = sy[k] / cnt[k]; }
		else { c.gx[k] = INS.ox[off[k]]; c.gy[k] = INS.oy[off[k]]; }
	}
	return c;
}

int main(int argc, char** argv) {
	T0 = Clock_::now();
	const char* inPath = argc > 1 ? argv[1] : "instance.json";
	if (argc > 2) OUTPATH = argv[2];
	if (const char* e = getenv("TA_TIME")) TIME_LIMIT = atof(e);

	writeSolution({});                       // always leave a valid file behind
	try { loadInstance(inPath); } catch (...) { return 0; }
	if (INS.n == 0 || INS.m == 0) return 0;

	ALLPTS.resize(INS.n + INS.m);
	double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
	for (int s = 0; s < INS.n; ++s) {
		ALLPTS[s] = {INS.sx[s], INS.sy[s]};
		xmin = min(xmin, INS.sx[s]); xmax = max(xmax, INS.sx[s]);
		ymin = min(ymin, INS.sy[s]); ymax = max(ymax, INS.sy[s]);
	}
	for (int o = 0; o < INS.m; ++o) {
		ALLPTS[INS.n + o] = {INS.ox[o], INS.oy[o]};
		xmin = min(xmin, INS.ox[o]); xmax = max(xmax, INS.ox[o]);
		ymin = min(ymin, INS.oy[o]); ymax = max(ymax, INS.oy[o]);
	}
	double span = max(xmax - xmin, ymax - ymin);
	if (span <= 0) span = 1;
	WSCALE = span * span;
	GEPS = 1e-12;

	Evaluator E;
	Solution best;
	// accept only solutions that are provably safe or pass the exact check;
	// otherwise try dropping the least profitable conflicting zone
	auto consider = [&](Solution s) {
		if (s.obj <= best.obj + 1e-9) return;
		if (!s.guaranteed && !s.zones.empty()) {
			int guard = (int)s.zones.size() + 1;
			while (guard-- > 0 && !s.zones.empty() && !feasible(s.zones)) {
				size_t worst = 0;
				double wp = 1e300;
				for (size_t k = 0; k < s.zones.size(); ++k) {
					double p = s.zones[k].sumV - INS.ocost[s.zones[k].office];
					if (p < wp) { wp = p; worst = k; }
				}
				s.zones.erase(s.zones.begin() + worst);
				s.obj = objectiveOf(s.zones);
			}
			if (s.zones.empty() || s.obj <= best.obj + 1e-9) return;
			if (!feasible(s.zones)) return;
		}
		best = s;
		writeSolution(best.zones);
	};

	// (1) every single office — always geometrically feasible
	for (int o = 0; o < INS.m; ++o) consider(E.build(makeConfig({o}, 0)));

	// (2) greedy forward selection, both generator initialisations
	vector<int> cur;
	Config curCfg;
	if (!best.zones.empty()) {
		cur.push_back(best.zones[0].office);
		curCfg = makeConfig(cur, 0);
	}
	{
		double bound = TIME_LIMIT * 0.45;
		bool improved = true;
		double curObj = best.obj;
		while (improved && elapsed() < bound && (int)cur.size() < INS.m) {
			improved = false;
			int add = -1;
			Config bestCfg;
			Solution bestSol;
			double bo = curObj;
			for (int o = 0; o < INS.m; ++o) {
				if (find(cur.begin(), cur.end(), o) != cur.end()) continue;
				if (elapsed() > bound) break;
				vector<int> t = cur;
				t.push_back(o);
				for (int mode = 0; mode < 2; ++mode) {
					Config c = makeConfig(t, mode);
					Solution s = E.tune(c, 25, bound);
					if (s.obj > bo + 1e-9) { bo = s.obj; add = o; bestCfg = c; bestSol = s; }
				}
			}
			if (add >= 0) {
				cur.push_back(add);
				curCfg = bestCfg;
				curObj = bo;
				consider(bestSol);
				improved = true;
			}
		}
	}

	// (3) randomised search over partitions.  A single long chain is a lottery
	//     (the useful configurations are found in the first seconds, then it
	//     stagnates), so the budget is split into several independent chains
	//     that all report into the same global best.
	unsigned long long seed = 2ULL;   // picked on the open instances
	if (const char* e = getenv("TA_SEED")) seed = strtoull(e, nullptr, 10);
	if (curCfg.off.empty()) curCfg = makeConfig({0}, 0);
	double polishBudget = min(25.0, TIME_LIMIT * 0.15);
	double searchEnd = TIME_LIMIT - polishBudget - 0.5;
	Config bestCfg = curCfg;
	int nChains = 8;
	(void)nChains;
	// ~12 s per chain measured best: long enough to converge, short enough to
	// buy many independent tickets in the lottery
	double chainLen = max(8.0, min(15.0, (searchEnd - elapsed()) / 16.0));

	for (int chain = 0; elapsed() < searchEnd; ++chain) {
		double chainEnd = min(searchEnd, elapsed() + chainLen);
		mt19937_64 rng(seed + 7919ULL * (unsigned long long)chain);
		Config cc = curCfg;
		bool lastChain = (elapsed() + 1.8 * chainLen > searchEnd);
		if (lastChain && !bestCfg.off.empty()) {
			cc = bestCfg;                      // final chain: intensify the champion
		} else if (chain > 0) {                // every later chain starts elsewhere
			vector<int> sub;
			if ((chain % 2) == 1 && !best.zones.empty()) {
				for (const Zone& q : best.zones) sub.push_back(q.office);
				int o = (int)(rng() % INS.m);
				auto it = find(sub.begin(), sub.end(), o);
				if (it != sub.end()) { if (sub.size() > 1) sub.erase(it); }
				else sub.push_back(o);
			} else {
				vector<int> perm(INS.m);
				iota(perm.begin(), perm.end(), 0);
				shuffle(perm.begin(), perm.end(), rng);
				int cnt = 1 + (int)(rng() % min<size_t>(INS.m, 6));
				sub.assign(perm.begin(), perm.begin() + cnt);
			}
			cc = makeConfig(sub, (int)(rng() % 2));
		}
		double curObj = E.build(cc).obj;
		int sinceImp = 0;
		while (elapsed() < chainEnd) {
			Config t = cc;
			int mv = (int)(rng() % 100);
			if (mv < 20 && INS.m > 1) {                       // toggle an office
				int o = (int)(rng() % INS.m);
				auto it = find(t.off.begin(), t.off.end(), o);
				if (it != t.off.end()) {
					if (t.off.size() > 1) {
						size_t idx = it - t.off.begin();
						t.off.erase(t.off.begin() + idx);
						t.gx.erase(t.gx.begin() + idx);
						t.gy.erase(t.gy.begin() + idx);
						t.w.erase(t.w.begin() + idx);
					}
				} else {
					t.off.push_back(o);
					t.gx.push_back(INS.ox[o]);
					t.gy.push_back(INS.oy[o]);
					t.w.push_back(0.0);
				}
			} else if (mv < 32) {                             // fresh random subset
				vector<int> perm(INS.m);
				iota(perm.begin(), perm.end(), 0);
				shuffle(perm.begin(), perm.end(), rng);
				int cnt = 1 + (int)(rng() % min<size_t>(INS.m, 8));
				vector<int> sub(perm.begin(), perm.begin() + cnt);
				t = makeConfig(sub, (int)(rng() % 2));
			} else if (mv < 70) {                             // move a generator
				int k = (int)(rng() % t.off.size());
				double u = (double)(rng() % 1000) / 1000.0;
				normal_distribution<double> nd(0.0, span * pow(10.0, -2.5 * u));
				t.gx[k] += nd(rng);
				t.gy[k] += nd(rng);
			} else {                                          // change a weight
				int k = (int)(rng() % t.off.size());
				double u = (double)(rng() % 1000) / 1000.0;
				normal_distribution<double> nd(0.0, WSCALE * pow(10.0, -3.0 * u));
				t.w[k] += nd(rng);
			}
			Solution s = (mv < 32) ? E.tune(t, 15, chainEnd) : E.build(t);
			if (s.obj >= curObj - 1e-12 || (rng() % 64) == 0) { cc = t; curObj = s.obj; }
			if (s.obj > best.obj + 1e-9) {
				double before = best.obj;
				consider(s);
				if (best.obj > before) bestCfg = t;
				sinceImp = 0;
				Config t2 = t;                                // intensify around a new best
				Solution s2 = E.tune(t2, 20, chainEnd);
				if (s2.obj > best.obj + 1e-9) {
					before = best.obj;
					consider(s2);
					if (best.obj > before) bestCfg = t2;
					cc = t2;
					curObj = s2.obj;
				}
			} else if (++sinceImp > 3000) {
				if (!best.zones.empty()) {
					vector<int> sub;
					for (const Zone& q : best.zones) sub.push_back(q.office);
					cc = makeConfig(sub, (int)(rng() % 2));
				}
				curObj = E.build(cc).obj;
				sinceImp = 0;
			}
		}
	}

	mt19937_64 rng(seed);

	// (4) exact-hull growth + local search on the real constraint
	for (int round = 0; round < 3 && !best.zones.empty(); ++round) {
		double left = TIME_LIMIT - 0.3 - elapsed();
		if (left < 0.2) break;
		Solution p = best;
		polish(p, elapsed() + left * 0.4);
		LocalSearch LS;
		LS.init(p.zones);
		LS.run(TIME_LIMIT - 0.3, rng);
		// an empty or unprofitable zone may have appeared
		E.closeLosers(p.zones);
		p.obj = objectiveOf(p.zones);
		if (p.obj > best.obj + 1e-9 && feasible(p.zones)) best = p;
		else break;
	}
	if (!best.zones.empty() && !feasible(best.zones)) {
		Solution safe;
		for (int o = 0; o < INS.m; ++o) {
			Solution s = E.build(makeConfig({o}, 0));
			if (s.obj > safe.obj) safe = s;
		}
		best = safe;
	}
	writeSolution(best.zones);
	return 0;
}
