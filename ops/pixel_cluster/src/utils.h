#ifndef OPS_UTILS_H
#define OPS_UTILS_H
namespace utils {
template<class T>
int disjoint_set_find_impl(T* p, int x)
{
    if (p[x].parent != x) {
        p[x].parent = disjoint_set_find_impl(p, p[x].parent);
    }
    return p[x].parent;
}

template <class T>
int disjoint_set_find(T* p, int x)
{
    int y = p[x].parent;
    if (p[y].parent == y) {
        return y;
    }
    int root = disjoint_set_find_impl(p, p[y].parent);
    p[x].parent = root;
    p[y].parent = root;
    return root;
}

template <class T>
void disjoint_set_union(T* p, int x, int y)
{
    x = disjoint_set_find(p, x);
    y = disjoint_set_find(p, y);
    if (x == y) {
        return;
    }
    if (p[x].rank < p[y].rank) {
        p[x].parent = y;
    } else if (p[y].rank < p[x].rank) {
        p[y].parent =x;
    } else {
        p[x].parent = y;
        p[y].rank++;
    }
}
}
#endif