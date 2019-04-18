#ifndef __BENCHMARK_SPECIAL_GRAPH_H
#define __BENCHMARK_SPECIAL_GRAPH_H

#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <utility>
#include <cstdlib>

namespace graph{

template <typename T>
inline bool overlap(T i, T j, bool left=true) {
  if (j == 0)
    return false;
  if (left)
    return ((i >> (j - 1)) & 1) == 0;
  else
    return ((i >> (j - 1)) & 1) == 1;
}

template <typename T>
class SegTree{
private:
  T length, n_nodes, n_lvl, n_edges;
  bool triu;
  std::vector<T> topdown, bottomup;
  std::vector<T> edges[2];
  std::vector<T> n_nodes_arr, shift;

  void build_graph() {
    T i = length;
    shift.push_back(0);
    while (i >= 2) {
      n_nodes_arr.push_back(i);
      n_nodes += i;
      shift.push_back(n_nodes);
      n_lvl++;
      i = (i + 1) >> 1;
    }

    n_nodes_arr.push_back(1);
    n_nodes++;

    n_edges = 0;
    for (T i = 0; i < length; i++) {
      T v = i;
      for (T j = 0; j < n_lvl; j++) {
        // Self loop
        if (i == (v << j)) {
          edges[0].push_back(shift[j] + v);
          edges[1].push_back(shift[j] + v);
          if (j == 0)
            topdown.push_back(n_edges);
          else
            bottomup.push_back(n_edges);
          n_edges++;
        }

        // top-down edges (right)
        if (!triu && (i + (1 << j) < length)) {
          if (overlap(i, j, false))
            edges[0].push_back(shift[j - 1] + (((v + 1) << 1) + 1));
          else
            edges[0].push_back(shift[j] + (v + 1));
          edges[1].push_back(i);
          topdown.push_back(n_edges);
          n_edges++;
        }

        // top-down edges (left)
        if (v >= 1) {
          if (overlap(i, j, true))
            edges[0].push_back(shift[j - 1] + ((v - 1) << 1));
          else
            edges[0].push_back(shift[j] + (v - 1));
          edges[1].push_back(i);
          topdown.push_back(n_edges);
          n_edges++;
        }

        // bottom up connection
        if (j > 0) {
          edges[0].push_back(i);
          edges[1].push_back(shift[j] + v);
          bottomup.push_back(n_edges);
          n_edges++;
        }

        v >>= 1;
      }
    }
  }

public:
  SegTree(T length, bool triu=false) {
    n_nodes = 0;
    this->length = length;
    this->triu = triu;
    n_lvl = 1;
    build_graph();
  }

  ~SegTree() {}

  std::vector<std::pair<T, T> > get_edges(T v_shift) const{
    std::vector<std::pair<T, T> > res;
    for (size_t i = 0; i < edges[0].size(); i++)
      res.push_back(std::make_pair(edges[0][i] + v_shift, edges[1][i] + v_shift));
    return res;
  }

  T number_of_nodes() const{
    return n_nodes;
  }

  T number_of_edges() const{
    return n_edges;
  }
};


template<typename T>
void segtree_transformer_csr(int64_t N, int64_t min_length, int64_t max_length, bool triu,
                             std::vector<T>& row_offsets,
                             std::vector<T>& column_indices) {
  T v_shift = 0;
  std::unordered_map<T, SegTree<T> > tree_map;
  row_offsets.push_back(0);
  for (int64_t i = 0; i < N; i++) {
    T length = min_length + rand() % std::max<T>((max_length - min_length), 1);
    SegTree<T> *tree;
    auto search = tree_map.find(length);
    if (search != tree_map.end()) { // found
      tree = &(search->second);
    } else { // not found
      tree = new SegTree<T>(length, triu);
    }
    auto edges = tree->get_edges(v_shift);
    std::sort(edges.begin(), edges.end(), [](const std::pair<T, T> &e1, const std::pair<T, T> &e2) {
        if (e1.first == e2.first)
          return e1.second < e2.second;
        else
          return e1.first < e2.first;
      });
    row_offsets.resize(1 + v_shift + tree->number_of_nodes(), 0);
    for (auto &e: edges) {
      T src = e.first, dst = e.second;
      row_offsets[src + 1]++;
      column_indices.push_back(dst);
    }
    for (T j = v_shift + 1; j < v_shift + tree->number_of_nodes() + 1; j++) {
      row_offsets[j] = row_offsets[j - 1] + row_offsets[j];
    }
    v_shift += tree->number_of_nodes();
  }
}

template<typename T>
void full_transformer_csr(int64_t N, int64_t min_length, int64_t max_length, bool triu,
                          std::vector<T>& row_offsets,
                          std::vector<T>& column_indices) {
  T v_shift = 0, e_shift = 0;
  for (int64_t i = 0; i < N; i++) {
    T length = min_length + rand() % std::max<T>((max_length - min_length), 1);
    for (T u = v_shift; u < v_shift + length; u++) {
      row_offsets.push_back(e_shift);
      T v_start = triu ? u: v_shift;
      for (T v = v_start; v < v_shift + length; v++) {
        column_indices.push_back(v);
        e_shift++;
      }
    }
    row_offsets.push_back(e_shift);
    v_shift += length;
  }
}

}
#endif
