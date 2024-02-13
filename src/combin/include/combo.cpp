#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace std;

#include "combo.hpp"

void backtrack(int l, int iarray[], int& indx, int& k, int& nstack, int stack[], int maxstack) {
  if (indx == 0) {
    k = 1;
    nstack = 0;
    indx = 2;
    return;
  }
  for (;;) {
    nstack = nstack - 1;
    if (stack[nstack] != 0) {
      iarray[k - 1] = stack[nstack - 1];
      stack[nstack - 1] = stack[nstack] - 1;

      if (k != l) {
        k = k + 1;
        indx = 2;
      } else {
        indx = 1;
      }
      break;
    } else {
      k = k - 1;

      if (k <= 0) {
        indx = 3;
        break;
      }
    }
  }
  return;
}

bool bal_seq_check(int n, int t[]) {
  bool check;
  int i;
  int one_count;
  int zero_count;

  check = true;

  if (n < 1) {
    check = false;
    return check;
  }

  one_count = 0;
  zero_count = 0;

  for (i = 0; i < 2 * n; i++) {
    if (t[i] == 0) {
      zero_count = zero_count + 1;
    } else if (t[i] == 1) {
      one_count = one_count + 1;
    } else {
      check = false;
      return check;
    }

    if (zero_count < one_count) {
      check = false;
      return check;
    }
  }

  if (one_count != zero_count) {
    check = false;
  }

  return check;
}

int bal_seq_enum(int n) {
  int value;

  value = i4_choose(2 * n, n) / (n + 1);

  return value;
}

int bal_seq_rank(int n, int t[]) {
  bool check;
  int mxy;
  int rank;
  int x;
  int y;
  check = bal_seq_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "BAL_SEQ_RANK(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  y = 0;
  rank = 0;

  for (x = 1; x <= 2 * n - 1; x++) {
    if (t[x - 1] == 0) {
      y = y + 1;
    } else {
      mxy = mountain(n, x, y + 1);
      rank = rank + mxy;
      y = y - 1;
    }
  }

  return rank;
}

void bal_seq_successor(int n, int t[], int& rank) {
  bool check;
  int i;
  int j;
  int open;
  int open_index;
  int slot;
  int slot_index;
  int slot_ones;
  if (rank == -1) {
    for (i = 0; i < n; i++) {
      t[i] = 0;
    }
    for (i = n; i < 2 * n; i++) {
      t[i] = 1;
    }
    rank = 0;
    return;
  }
  check = bal_seq_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "BAL_SEQ_SUCCESSOR(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }
  slot = 0;
  slot_index = 0;
  slot_ones = 0;

  open = 0;
  open_index = 0;

  for (i = 1; i <= 2 * n; i++) {
    if (t[i - 1] == 0) {
      if (0 < slot) {
        if (slot_ones < slot) {
          open = slot;
          open_index = slot_index;
        }
      }
      slot = slot + 1;
      slot_index = i;
    } else {
      slot_ones = slot_ones + 1;
    }
  }
  if (open != 0) {
    j = open_index + 1;

    while (t[j - 1] == 1) {
      j = j + 1;
    }

    t[j - 1] = 1;

    for (i = open + 1; i <= n; i++) {
      j = j + 1;
      t[j - 1] = 0;
    }

    for (i = j + 1; i <= 2 * n; i++) {
      t[i - 1] = 1;
    }
  } else {
    for (i = 0; i < n; i++) {
      t[i] = 0;
    }
    for (i = n; i < 2 * n; i++) {
      t[i] = 1;
    }
    rank = 0;
    return;
  }
  rank = rank + 1;

  return;
}

int* bal_seq_to_tableau(int n, int t[]) {
  int c[2];
  bool check;
  int i;
  int r;
  int* tab;
  check = bal_seq_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "BAL_SEQ_TO_TABLEAU(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  tab = new int[2 * n];

  c[0] = 0;
  c[1] = 0;

  for (i = 1; i <= 2 * n; i++) {
    r = t[i - 1] + 1;
    c[r - 1] = c[r - 1] + 1;
    tab[r - 1 + (c[r - 1] - 1) * 2] = i;
  }

  return tab;
}

int* bal_seq_unrank(int rank, int n) {
  int low;
  int m;
  int nseq;
  int* t;
  int x;
  int y;
  if (n < 1) {
    cerr << "\n";
    cerr << "BAL_SEQ_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  nseq = bal_seq_enum(n);

  if (rank < 0 || nseq < rank) {
    cerr << "\n";
    cerr << "BAL_SEQ_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }

  t = new int[2 * n];

  y = 0;
  low = 0;

  for (x = 0; x < 2 * n; x++) {
    m = mountain(n, x + 1, y + 1);

    if (rank <= low + m - 1) {
      y = y + 1;
      t[x] = 0;
    } else {
      low = low + m;
      y = y - 1;
      t[x] = 1;
    }
  }
  return t;
}

int* bell_numbers(int m) {
  int* b;
  int i;
  int j;

  b = new int[m + 1];

  b[0] = 1;
  for (j = 1; j <= m; j++) {
    b[j] = 0;
    for (i = 0; i < j; i++) {
      b[j] = b[j] + i4_choose(j - 1, i) * b[i];
    }
  }
  return b;
}

void bell_values(int& n_data, int& n, int& c) {
#define N_MAX 11

  static int c_vec[N_MAX] = {1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975};

  static int n_vec[N_MAX] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  if (n_data < 0) {
    n_data = 0;
  }

  n_data = n_data + 1;

  if (N_MAX < n_data) {
    n_data = 0;
    n = 0;
    c = 0;
  } else {
    n = n_vec[n_data - 1];
    c = c_vec[n_data - 1];
  }

  return;
#undef N_MAX
}

bool cycle_check(int n, int ncycle, int t[], int index[]) {
  bool check;
  int i;
  int ifind;
  int iseek;

  check = true;
  if (n < 1) {
    check = false;
    return check;
  }
  if (ncycle < 1 || n < ncycle) {
    check = false;
    return check;
  }
  for (i = 0; i < ncycle; i++) {
    if (index[i] < 1 || n < index[i]) {
      check = false;
      return check;
    }
  }
  if (i4vec_sum(ncycle, index) != n) {
    check = false;
    return check;
  }
  for (i = 0; i < n; i++) {
    if (t[i] < 1 || n < t[i]) {
      check = false;
      return check;
    }
  }
  for (iseek = 1; iseek <= n; iseek++) {
    ifind = -1;

    for (i = 0; i < n; i++) {
      if (t[i] == iseek) {
        ifind = i + 1;
        break;
      }
    }

    if (ifind == -1) {
      check = false;
      return check;
    }
  }
  return check;
}

int* cycle_to_perm(int n, int ncycle, int t[], int index[]) {
  bool check;
  int i;
  int j;
  int jhi;
  int jlo;
  int* p;
  check = cycle_check(n, ncycle, t, index);

  if (!check) {
    cerr << "\n";
    cerr << "CYCLE_TO_PERM(): Fatal error!\n";
    cerr << "  The cycle is not legal.\n";
    exit(1);
  }

  p = new int[n];

  jhi = 0;

  for (i = 1; i <= ncycle; i++) {
    jlo = jhi + 1;
    jhi = jhi + index[i - 1];

    for (j = jlo; j <= jhi; j++) {
      if (j < jhi) {
        p[t[j - 1] - 1] = t[j];
      } else {
        p[t[j - 1] - 1] = t[jlo - 1];
      }
    }
  }

  return p;
}

int dist_enum(int k, int m) {
  int value;

  value = i4_choose(m + k - 1, m);

  return value;
}

void dist_next(int k, int m, int q[], int& leftmost, bool& more) {
  int i;
  if (!more) {
    more = true;
    for (i = 0; i < k - 1; i++) {
      q[i] = 0;
    }
    q[k - 1] = m;

    leftmost = k + 1;
  } else if (q[0] == m) {
    more = false;

    for (i = 0; i < k - 1; i++) {
      q[i] = 0;
    }
    q[k - 1] = m;

    leftmost = k + 1;
  } else if (leftmost < k + 1) {
    leftmost = leftmost - 1;
    q[k - 1] = q[leftmost - 1] - 1;
    q[leftmost - 1] = 0;
    q[leftmost - 2] = q[leftmost - 2] + 1;
    if (q[k - 1] != 0) {
      leftmost = k + 1;
    }
  } else {
    if (q[k - 1] == 1) {
      leftmost = k;
    }
    q[k - 1] = q[k - 1] - 1;
    q[k - 2] = q[k - 2] + 1;
  }
  return;
}

bool edge_check(int n_node, int n_edge, int t[]) {
  bool check;
  int i;
  int j;
  int j2;

  check = true;

  if (n_node < 1) {
    check = false;
    return check;
  }

  if (n_edge < 1) {
    check = false;
    return check;
  }
  for (i = 0; i < 2; i++) {
    for (j = 0; j < n_edge; j++) {
      if (t[i + j * 2] < 1 || n_node < t[i + j * 2]) {
        check = false;
        return check;
      }
    }
  }
  for (j = 0; j < n_edge; j++) {
    if (t[0 + j * 2] == t[1 + j * 2]) {
      check = false;
      return check;
    }
  }
  for (j = 0; j < n_edge - 1; j++) {
    for (j2 = j + 1; j2 < n_edge; j2++) {
      if (t[0 + j * 2] == t[0 + j2 * 2] && t[1 + j * 2] == t[1 + j2 * 2]) {
        check = false;
        return check;
      } else if (t[0 + j * 2] == t[1 + j2 * 2] && t[1 + j * 2] == t[0 + j2 * 2]) {
        check = false;
        return check;
      }
    }
  }
  return check;
}

int* edge_degree(int n_node, int n_edge, int t[]) {
  bool check;
  int* d;
  int i;
  int j;
  check = edge_check(n_node, n_edge, t);

  if (!check) {
    cerr << "\n";
    cerr << "EDGE_DEGREE(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }
  d = new int[n_node];

  for (i = 0; i < n_node; i++) {
    d[i] = 0;
  }
  for (j = 0; j < n_edge; j++) {
    d[t[0 + j * 2] - 1] = d[t[0 + j * 2] - 1] + 1;
    d[t[1 + j * 2] - 1] = d[t[1 + j * 2] - 1] + 1;
  }

  return d;
}

int edge_enum(int n_node) {
  int value;

  value = (n_node * (n_node - 1)) / 2;

  return value;
}

void gamma_log_values(int& n_data, double& x, double& fx) {
#define N_MAX 20

  static double fx_vec[N_MAX] = {0.1524063822430784E+01,  0.7966778177017837E+00,  0.3982338580692348E+00,  0.1520596783998375E+00,  0.0000000000000000E+00,
                                 -0.4987244125983972E-01, -0.8537409000331584E-01, -0.1081748095078604E+00, -0.1196129141723712E+00, -0.1207822376352452E+00,
                                 -0.1125917656967557E+00, -0.9580769740706586E-01, -0.7108387291437216E-01, -0.3898427592308333E-01, 0.00000000000000000E+00,
                                 0.69314718055994530E+00, 0.17917594692280550E+01, 0.12801827480081469E+02, 0.39339884187199494E+02, 0.71257038967168009E+02};

  static double x_vec[N_MAX] = {0.20E+00, 0.40E+00, 0.60E+00, 0.80E+00, 1.00E+00, 1.10E+00, 1.20E+00, 1.30E+00,  1.40E+00,  1.50E+00,
                                1.60E+00, 1.70E+00, 1.80E+00, 1.90E+00, 2.00E+00, 3.00E+00, 4.00E+00, 10.00E+00, 20.00E+00, 30.00E+00};

  if (n_data < 0) {
    n_data = 0;
  }

  n_data = n_data + 1;

  if (N_MAX < n_data) {
    n_data = 0;
    x = 0.0;
    fx = 0.0;
  } else {
    x = x_vec[n_data - 1];
    fx = fx_vec[n_data - 1];
  }

  return;
#undef N_MAX
}

bool gray_code_check(int n, int t[]) {
  bool check;
  int i;

  check = true;

  if (n < 1) {
    check = false;
    return check;
  }

  for (i = 0; i < n; i++) {
    if (t[i] != 0 && t[i] != 1) {
      check = false;
      return check;
    }
  }

  return check;
}

int gray_code_enum(int n) {
  int value;

  value = i4_power(2, n);

  return value;
}

int* gray_code_random(int n) {
  int gray_num;
  int rank;
  int* t;
  if (n < 1) {
    cerr << "\n";
    cerr << "gray_code_random(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }
  gray_num = gray_code_enum(n);
  rank = 1 + (rand() % gray_num);
  t = gray_code_unrank(rank, n);

  return t;
}

int gray_code_rank(int n, int t[]) {
  int b;
  bool check;
  int i;
  int rank;
  check = gray_code_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "GRAY_CODE_RANK(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  rank = 0;
  b = 0;

  for (i = n - 1; 0 <= i; i--) {
    if (t[n - i - 1] != 0) {
      b = 1 - b;
    }
    if (b == 1) {
      rank = rank + i4_power(2, i);
    }
  }
  return rank;
}

void gray_code_successor(int n, int t[], int& rank) {
  bool check;
  int i;
  int weight;
  if (rank == -1) {
    for (i = 0; i < n; i++) {
      t[i] = 0;
    }
    rank = 0;
    return;
  }
  check = gray_code_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "GRAY_CODE_SUCCESSOR(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  weight = i4vec_sum(n, t);

  if ((weight % 2) == 0) {
    if (t[n - 1] == 0) {
      t[n - 1] = 1;
    } else {
      t[n - 1] = 0;
    }
    rank = rank + 1;
    return;
  } else {
    for (i = n - 1; 1 <= i; i--) {
      if (t[i] == 1) {
        if (t[i - 1] == 0) {
          t[i - 1] = 1;
        } else {
          t[i - 1] = 0;
        }
        rank = rank + 1;
        return;
      }
    }
    for (i = 0; i < n; i++) {
      t[i] = 0;
    }
    rank = 0;
  }
  return;
}

int* gray_code_unrank(int rank, int n) {
  int b;
  int bprime;
  int i;
  int ngray;
  int rank_copy;
  int* t;
  if (n < 1) {
    cerr << "\n";
    cerr << "GRAY_CODE_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  ngray = gray_code_enum(n);

  if (rank < 0 || ngray < rank) {
    cerr << "\n";
    cerr << "GRAY_CODE_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }

  t = new int[n];

  rank_copy = rank;
  for (i = 0; i < n; i++) {
    t[i] = 0;
  }
  bprime = 0;

  for (i = n - 1; 0 <= i; i--) {
    b = rank_copy / i4_power(2, i);

    if (b != bprime) {
      t[n - i - 1] = 1;
    }
    bprime = b;
    rank_copy = rank_copy - b * i4_power(2, i);
  }
  return t;
}

int i4_choose(int n, int k) {
  int i;
  int mn;
  int mx;
  int value;

  mn = k;
  if (n - k < mn) {
    mn = n - k;
  }

  if (mn < 0) {
    value = 0;
  } else if (mn == 0) {
    value = 1;
  } else {
    mx = k;
    if (mx < n - k) {
      mx = n - k;
    }
    value = mx + 1;

    for (i = 2; i <= mn; i++) {
      value = (value * (mx + i)) / i;
    }
  }

  return value;
}

int i4_factorial(int n) {
  int i;
  int value;

  value = 1;

  if (13 < n) {
    cerr << "\n";
    cerr << "I4_FACTORIAL(): Fatal error!\n";
    cerr << "  I4_FACTORIAL(N) cannot be computed as an integer\n";
    cerr << "  for 13 < N.\n";
    cerr << "  Input value N = " << n << "\n";
    exit(1);
  }

  for (i = 1; i <= n; i++) {
    value = value * i;
  }

  return value;
}

void i4_factorial_values(int& n_data, int& n, int& fn) {
#define N_MAX 13

  static int fn_vec[N_MAX] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600};

  static int n_vec[N_MAX] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  if (n_data < 0) {
    n_data = 0;
  }

  n_data = n_data + 1;

  if (N_MAX < n_data) {
    n_data = 0;
    n = 0;
    fn = 0;
  } else {
    n = n_vec[n_data - 1];
    fn = fn_vec[n_data - 1];
  }

  return;
#undef N_MAX
}

int i4_fall(int x, int n) {
  int i;
  int value;

  value = 1;

  if (0 < n) {
    for (i = 1; i <= n; i++) {
      value = value * x;
      x = x - 1;
    }
  } else if (n < 0) {
    for (i = -1; n <= i; i--) {
      value = value * x;
      x = x + 1;
    }
  }

  return value;
}

void i4_fall_values(int& n_data, int& m, int& n, int& fmn) {
#define N_MAX 15

  static int fmn_vec[N_MAX] = {1, 5, 20, 60, 120, 120, 0, 1, 10, 4000, 90, 4896, 24, 912576, 0};

  static int m_vec[N_MAX] = {5, 5, 5, 5, 5, 5, 5, 50, 10, 4000, 10, 18, 4, 98, 1};

  static int n_vec[N_MAX] = {0, 1, 2, 3, 4, 5, 6, 0, 1, 1, 2, 3, 4, 3, 7};

  if (n_data < 0) {
    n_data = 0;
  }

  n_data = n_data + 1;

  if (N_MAX < n_data) {
    n_data = 0;
    m = 0;
    n = 0;
    fmn = 0;
  } else {
    m = m_vec[n_data - 1];
    n = n_vec[n_data - 1];
    fmn = fmn_vec[n_data - 1];
  }

  return;
#undef N_MAX
}

int i4_huge() { return 2147483647; }

int i4_max(int i1, int i2) {
  int value;

  if (i2 < i1) {
    value = i1;
  } else {
    value = i2;
  }
  return value;
}

int i4_min(int i1, int i2) {
  int value;

  if (i1 < i2) {
    value = i1;
  } else {
    value = i2;
  }
  return value;
}

int i4_power(int i, int j) {
  int k;
  int value;

  if (j < 0) {
    if (i == 1) {
      value = 1;
    } else if (i == 0) {
      cerr << "\n";
      cerr << "I4_POWER(): Fatal error!\n";
      cerr << "  I^J requested, with I = 0 and J negative.\n";
      exit(1);
    } else {
      value = 0;
    }
  } else if (j == 0) {
    if (i == 0) {
      cerr << "\n";
      cerr << "I4_POWER(): Fatal error!\n";
      cerr << "  I^J requested, with I = 0 and J = 0.\n";
      exit(1);
    } else {
      value = 1;
    }
  } else if (j == 1) {
    value = i;
  } else {
    value = 1;
    for (k = 1; k <= j; k++) {
      value = value * i;
    }
  }
  return value;
}

int i4_uniform_ab(int a, int b, int& seed) {
  int c;
  const int i4_huge = 2147483647;
  int k;
  float r;
  int value;

  if (seed == 0) {
    cerr << "\n";
    cerr << "I4_UNIFORM_AB(): Fatal error!\n";
    cerr << "  Input value of SEED = 0.\n";
    exit(1);
  }
  if (b < a) {
    c = a;
    a = b;
    b = c;
  }

  k = seed / 127773;

  seed = 16807 * (seed - k * 127773) - k * 2836;

  if (seed < 0) {
    seed = seed + i4_huge;
  }

  r = (float)(seed)*4.656612875E-10;
  r = (1.0 - r) * ((float)a - 0.5) + r * ((float)b + 0.5);
  value = round(r);
  if (value < a) {
    value = a;
  }
  if (b < value) {
    value = b;
  }

  return value;
}

int* i4mat_copy_new(int m, int n, int a1[]) {
  int* a2;
  int i;
  int j;

  a2 = new int[m * n];

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      a2[i + j * m] = a1[i + j * m];
    }
  }
  return a2;
}

void i4mat_print(int m, int n, int a[], string title) {
  i4mat_print_some(m, n, a, 1, 1, m, n, title);

  return;
}

void i4mat_print_some(int m, int n, int a[], int ilo, int jlo, int ihi, int jhi, string title) {
#define INCX 10

  int i;
  int i2hi;
  int i2lo;
  int j;
  int j2hi;
  int j2lo;

  cout << "\n";
  cout << title << "\n";

  if (m <= 0 || n <= 0) {
    cout << "\n";
    cout << "  (None)\n";
    return;
  }
  for (j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX) {
    j2hi = j2lo + INCX - 1;
    j2hi = i4_min(j2hi, n);
    j2hi = i4_min(j2hi, jhi);

    cout << "\n";
    cout << "  Col:";
    for (j = j2lo; j <= j2hi; j++) {
      cout << "  " << setw(6) << j - 1;
    }
    cout << "\n";
    cout << "  Row\n";
    cout << "\n";
    i2lo = i4_max(ilo, 1);
    i2hi = i4_min(ihi, m);

    for (i = i2lo; i <= i2hi; i++) {
      cout << setw(5) << i - 1 << ":";
      for (j = j2lo; j <= j2hi; j++) {
        cout << "  " << setw(6) << a[i - 1 + (j - 1) * m];
      }
      cout << "\n";
    }
  }

  return;
#undef INCX
}

void i4vec_backtrack(int n, int maxstack, int stack[], int x[], int& indx, int& k, int& nstack, int ncan[]) {
  if (indx == 0) {
    k = 1;
    nstack = 0;
    indx = 2;
    return;
  }
  for (;;) {
    if (0 < ncan[k - 1]) {
      x[k - 1] = stack[nstack - 1];
      nstack = nstack - 1;

      ncan[k - 1] = ncan[k - 1] - 1;

      if (k != n) {
        k = k + 1;
        indx = 2;
      } else {
        indx = 1;
      }
      break;
    } else {
      k = k - 1;

      if (k <= 0) {
        indx = 3;
        break;
      }
    }
  }
  return;
}

int* i4vec_copy_new(int n, int a1[]) {
  int* a2;
  int i;

  a2 = new int[n];

  for (i = 0; i < n; i++) {
    a2[i] = a1[i];
  }
  return a2;
}

int i4vec_dot_product(int n, int x[], int y[]) {
  int i;
  int value;

  value = 0;
  for (i = 0; i < n; i++) {
    value = value + x[i] * y[i];
  }

  return value;
}

void i4vec_indicator1(int n, int a[]) {
  int i;

  for (i = 0; i < n; i++) {
    a[i] = i + 1;
  }
  return;
}

int i4vec_max(int n, int a[]) {
  int i;
  int value;

  if (n <= 0) {
    return 0;
  }

  value = a[0];

  for (i = 1; i < n; i++) {
    if (value < a[i]) {
      value = a[i];
    }
  }

  return value;
}

int* i4vec_part1_new(int n, int npart) {
  int i;
  int* x;

  if (npart < 1 || n < npart) {
    cerr << "\n";
    cerr << "I4VEC_PART1_NEW(): Fatal error!\n";
    cerr << "  The input value of NPART is illegal.\n";
    exit(1);
  }

  x = new int[npart];

  x[0] = n + 1 - npart;
  for (i = 1; i < npart; i++) {
    x[i] = 1;
  }

  return x;
}

void i4vec_part2(int n, int npart, int x[]) {
  int i;
  int j;

  if (npart < 1) {
    cerr << "\n";
    cerr << "I4VEC_PART2(): Fatal error!\n";
    cerr << "  The input value of NPART is illegal.\n";
    exit(1);
  }

  for (i = 0; i < npart; i++) {
    x[i] = 0;
  }

  if (0 < n) {
    j = 1;
    for (i = 1; i <= n; i++) {
      x[j - 1] = x[j - 1] + 1;
      j = j + 1;
      if (npart < j) {
        j = 1;
      }
    }
  } else if (n < 0) {
    j = 1;
    for (i = n; i <= -1; i++) {
      x[j - 1] = x[j - 1] - 1;
      j = j + 1;
      if (npart < j) {
        j = 1;
      }
    }
  }

  return;
}

int* i4vec_part2_new(int n, int npart) {
  int* x;

  x = new int[npart];

  i4vec_part2(n, npart, x);

  return x;
}

void i4vec_print(int n, int a[], string title) {
  int i;

  cout << "\n";
  cout << title << "\n";
  cout << "\n";
  for (i = 0; i < n; i++) {
    cout << "  " << setw(8) << i << ": " << setw(8) << a[i] << "\n";
  }
  return;
}

void i4vec_reverse(int n, int a[]) {
  int i;
  int j;

  for (i = 0; i < n / 2; i++) {
    j = a[i];
    a[i] = a[n - 1 - i];
    a[n - 1 - i] = j;
  }

  return;
}

int i4vec_search_binary_a(int n, int a[], int b) {
  int high;
  int index;
  int low;
  int mid;
  if (n <= 0) {
    cerr << "\n";
    cerr << "I4VEC_SEARCH_BINARY_A(): Fatal error!\n";
    cerr << "  The array dimension N is less than 1.\n";
    exit(1);
  }

  index = -1;

  low = 1;
  high = n;

  while (low <= high) {
    mid = (low + high) / 2;

    if (a[mid - 1] == b) {
      index = mid;
      break;
    } else if (a[mid - 1] < b) {
      low = mid + 1;
    } else if (b < a[mid - 1]) {
      high = mid - 1;
    }
  }
  return index;
}

int i4vec_search_binary_d(int n, int a[], int b) {
  int high;
  int index;
  int low;
  int mid;
  if (n <= 0) {
    cerr << "\n";
    cerr << "I4VEC_SEARCH_BINARY_D(): Fatal error!\n";
    cerr << "  The array dimension N is less than 1.\n";
    exit(1);
  }

  index = -1;

  low = 1;
  high = n;

  while (low <= high) {
    mid = (low + high) / 2;

    if (a[mid - 1] == b) {
      index = mid;
      break;
    } else if (b < a[mid - 1]) {
      low = mid + 1;
    } else if (a[mid - 1] < b) {
      high = mid - 1;
    }
  }
  return index;
}

void i4vec_sort_insert_a(int n, int a[]) {
  int i;
  int j;
  int x;

  for (i = 1; i < n; i++) {
    x = a[i];

    j = i;

    while (1 <= j && x < a[j - 1]) {
      a[j] = a[j - 1];
      j = j - 1;
    }

    a[j] = x;
  }

  return;
}

void i4vec_sort_insert_d(int n, int a[]) {
  int i;
  int j;
  int x;

  for (i = 1; i < n; i++) {
    x = a[i];
    j = i;

    while (1 <= j && a[j - 1] < x) {
      a[j] = a[j - 1];
      j = j - 1;
    }
    a[j] = x;
  }

  return;
}

int i4vec_sum(int n, int a[]) {
  int i;
  int sum;

  sum = 0;
  for (i = 0; i < n; i++) {
    sum = sum + a[i];
  }
  return sum;
}

void i4vec_transpose_print(int n, int a[], string title) {
  int i;
  int ihi;
  int ilo;
  int title_len;

  title_len = title.length();

  for (ilo = 1; ilo <= n; ilo = ilo + 10) {
    ihi = i4_min(ilo + 10 - 1, n);
    if (ilo == 1) {
      cout << title;
    } else {
      for (i = 1; i <= title_len; i++) {
        cout << " ";
      }
    }
    for (i = ilo; i <= ihi; i++) {
      cout << "  " << setw(5) << a[i - 1];
    }
    cout << "\n";
  }

  return;
}

int* i4vec_uniform_ab_new(int n, int a, int b, int& seed) {
  int c;
  int i;
  const int i4_huge = 2147483647;
  int k;
  float r;
  int value;
  int* x;

  if (seed == 0) {
    cerr << "\n";
    cerr << "I4VEC_UNIFORM_AB_NEW(): Fatal error!\n";
    cerr << "  Input value of SEED = 0.\n";
    exit(1);
  }
  if (b < a) {
    c = a;
    a = b;
    b = c;
  }

  x = new int[n];

  for (i = 0; i < n; i++) {
    k = seed / 127773;

    seed = 16807 * (seed - k * 127773) - k * 2836;

    if (seed < 0) {
      seed = seed + i4_huge;
    }

    r = (float)(seed)*4.656612875E-10;
    r = (1.0 - r) * ((float)a - 0.5) + r * ((float)b + 0.5);
    value = round(r);
    if (value < a) {
      value = a;
    }
    if (b < value) {
      value = b;
    }

    x[i] = value;
  }

  return x;
}

void knapsack_01(int n, double mass_limit, double p[], double w[], double x[], double& mass, double& profit) {
  int i;
  int indx;
  int k;
  double mass_1;
  double mass_2;
  double mass_best;
  double mass_remaining;
  int maxstack = 100;
  int* ncan;
  int nstack;
  double profit_1;
  double profit_2;
  double profit_best;
  double* stack;
  double* x_best;

  ncan = new int[n];
  stack = new double[maxstack];
  x_best = new double[n];

  nstack = 0;
  for (i = 0; i < n; i++) {
    x_best[i] = 0.0;
  }
  profit_best = 0.0;
  mass_best = 0;
  indx = 0;

  for (;;) {
    r8vec_backtrack(n, maxstack, stack, x, indx, k, nstack, ncan);
    if (indx == 1) {
      profit = r8vec_dot_product(n, p, x);
      mass = r8vec_dot_product(n, w, x);

      if (profit_best < profit || (profit == profit_best && mass < mass_best)) {
        profit_best = profit;
        mass_best = mass;
        for (i = 0; i < n; i++) {
          x_best[i] = x[i];
        }
      }
    } else if (indx == 2) {
      ncan[k - 1] = 0;

      mass_1 = w[k - 1];
      for (i = 0; i < k - 1; i++) {
        mass_1 = mass_1 + w[i] * x[i];
      }

      if (mass_1 <= mass_limit) {
        mass_remaining = mass_limit - mass_1;

        profit_1 = p[k - 1];
        for (i = 0; i < k - 1; i++) {
          profit_1 = profit_1 + p[i] * x[i];
        }

        if (k < n) {
          knapsack_rational(n - k, mass_remaining, p + k, w + k, x + k, mass_2, profit_2);
        } else {
          profit_2 = 0.0;
        }

        if (profit_best < profit_1 + profit_2) {
          if (maxstack <= nstack) {
            cerr << "\n";
            cerr << "KNAPSACK_01(): Fatal error!\n";
            cerr << "  Exceeded stack space.\n";
            return;
          }
          ncan[k - 1] = ncan[k - 1] + 1;
          nstack = nstack + 1;
          stack[nstack - 1] = 1.0;
        }
      }

      if (maxstack <= nstack) {
        cerr << "\n";
        cerr << "KNAPSACK_01(): Fatal error!\n";
        cerr << "  Exceeded stack space.\n";
        return;
      }

      ncan[k - 1] = ncan[k - 1] + 1;
      nstack = nstack + 1;
      stack[nstack - 1] = 0.0;
    } else {
      profit = profit_best;
      mass = mass_best;
      for (i = 0; i < n; i++) {
        x[i] = x_best[i];
      }
      break;
    }
  }

  delete[] ncan;
  delete[] stack;
  delete[] x_best;

  return;
}

void knapsack_rational(int n, double mass_limit, double p[], double w[], double x[], double& mass, double& profit) {
  int i;

  mass = 0.0;
  profit = 0.0;

  for (i = 0; i < n; i++) {
    if (mass_limit <= mass) {
      x[i] = 0.0;
    } else if (mass + w[i] <= mass_limit) {
      x[i] = 1.0;
      mass = mass + w[i];
      profit = profit + p[i];
    } else {
      x[i] = (mass_limit - mass) / w[i];
      mass = mass_limit;
      profit = profit + p[i] * x[i];
    }
  }
  return;
}

void knapsack_reorder(int n, double p[], double w[]) {
  int i;
  int j;
  double t;
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      if (p[i] * w[j] < p[j] * w[i]) {
        t = p[i];
        p[i] = p[j];
        p[j] = t;

        t = w[i];
        w[i] = w[j];
        w[j] = t;
      }
    }
  }
  return;
}

bool ksubset_colex_check(int k, int n, int t[]) {
  bool check;
  int i;
  int tmax;

  check = true;

  if (n < 0) {
    check = false;
    return check;
  }

  if (k < 0 || n < k) {
    check = false;
    return check;
  }

  tmax = n + 1;

  for (i = 0; i < k; i++) {
    if (t[i] <= 0 || tmax <= t[i]) {
      check = false;
      return check;
    }
    tmax = t[i];
  }
  return check;
  ;
}

int ksubset_colex_rank(int k, int n, int t[]) {
  bool check;
  int i;
  int rank;
  check = ksubset_colex_check(k, n, t);

  if (!check) {
    cerr << "\n";
    cerr << "KSUBSET_COLEX_RANK(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  rank = 0;

  for (i = 0; i < k; i++) {
    rank = rank + i4_choose(t[i] - 1, k - i);
  }

  return rank;
}

void ksubset_colex_successor(int k, int n, int t[], int& rank) {
  bool check;
  int i;
  if (rank == -1) {
    for (i = 1; i <= k; i++) {
      t[i - 1] = k + 1 - i;
    }
    rank = 0;
    return;
  }
  check = ksubset_colex_check(k, n, t);

  if (!check) {
    cerr << "\n";
    cerr << "KSUBSET_COLEX_SUCCESSOR(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  for (i = k - 1; 1 <= i; i--) {
    if (t[k - i] + 1 < t[k - i - 1]) {
      t[k - i] = t[k - i] + 1;
      rank = rank + 1;
      return;
    }
  }

  if (t[0] < n) {
    t[0] = t[0] + 1;
    for (i = 1; i <= k - 1; i++) {
      t[k - i] = i;
    }
    rank = rank + 1;
    return;
  }
  for (i = 1; i <= k; i++) {
    t[i - 1] = k + 1 - i;
  }

  rank = 0;

  return;
}

int* ksubset_colex_unrank(int rank, int k, int n) {
  int i;
  int nksub;
  int rank_copy;
  int* t;
  int x;
  if (n < 1) {
    cerr << "\n";
    cerr << "KSUBSET_COLEX_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  if (k == 0) {
    t = new int[k];
    return t;
  }

  if (k < 0 || n < k) {
    cerr << "\n";
    cerr << "KSUBSET_COLEX_UNRANK(): Fatal error!\n";
    cerr << "  Input K is illegal.\n";
    exit(1);
  }

  nksub = ksubset_enum(k, n);

  if (rank < 0 || nksub < rank) {
    cerr << "\n";
    cerr << "KSUBSET_COLEX_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }
  rank_copy = rank;

  x = n;

  t = new int[k];

  for (i = 1; i <= k; i++) {
    while (rank_copy < i4_choose(x, k + 1 - i)) {
      x = x - 1;
    }

    t[i - 1] = x + 1;
    rank_copy = rank_copy - i4_choose(x, k + 1 - i);
  }

  return t;
}

int ksubset_enum(int k, int n) {
  int value;

  value = i4_choose(n, k);

  return value;
}

bool ksubset_lex_check(int k, int n, int t[]) {
  bool check;
  int i;
  int tmin;

  check = true;

  if (n < 0) {
    check = false;
    return check;
  }

  if (k < 0 || n < k) {
    check = false;
    return check;
  }

  tmin = 0;

  for (i = 0; i < k; i++) {
    if (t[i] <= tmin || n < t[i]) {
      check = false;
      return check;
    }
    tmin = t[i];
  }
  return check;
}

int ksubset_lex_rank(int k, int n, int t[]) {
  int check;
  int i;
  int j;
  int rank;
  int tim1;
  check = ksubset_lex_check(k, n, t);

  if (!check) {
    cerr << "\n";
    cerr << "KSUBSET_LEX_RANK(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  rank = 0;

  for (i = 1; i <= k; i++) {
    if (i == 1) {
      tim1 = 0;
    } else {
      tim1 = t[i - 2];
    }

    if (tim1 + 1 <= t[i - 1] - 1) {
      for (j = tim1 + 1; j <= t[i - 1] - 1; j++) {
        rank = rank + i4_choose(n - j, k - i);
      }
    }
  }

  return rank;
}

void ksubset_lex_successor(int k, int n, int t[], int& rank) {
  bool check;
  int i;
  int isave;
  int j;
  if (rank == -1) {
    i4vec_indicator1(k, t);
    rank = 0;
    return;
  }
  check = ksubset_lex_check(k, n, t);

  if (!check) {
    cerr << "\n";
    cerr << "KSUBSET_LEX_SUCCESSOR(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  isave = 0;

  for (i = k; 1 <= i; i--) {
    if (t[i - 1] != n - k + i) {
      isave = i;
      break;
    }
  }
  if (isave == 0) {
    i4vec_indicator1(k, t);
    rank = 0;
  } else {
    for (j = k; isave <= j; j--) {
      t[j - 1] = t[isave - 1] + 1 + j - isave;
    }
    rank = rank + 1;
  }

  return;
}

int* ksubset_lex_unrank(int rank, int k, int n) {
  int i;
  int nksub;
  int rank_copy;
  int* t;
  int x;
  if (n < 1) {
    cerr << "\n";
    cerr << "KSUBSET_LEX_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  if (k == 0) {
    t = new int[k];
    return t;
  }

  if (k < 0 || n < k) {
    cerr << "\n";
    cerr << "KSUBSET_LEX_UNRANK(): Fatal error!\n";
    cerr << "  Input K is illegal.\n";
    exit(1);
  }

  nksub = ksubset_enum(k, n);

  if (rank < 0 || nksub < rank) {
    cerr << "\n";
    cerr << "KSUBSET_LEX_UNRANK(): Fatal error!\n";
    cerr << "  Input rank is illegal.\n";
    exit(1);
  }

  t = new int[k];

  rank_copy = rank;

  x = 1;

  for (i = 1; i <= k; i++) {
    while (i4_choose(n - x, k - i) <= rank_copy) {
      rank_copy = rank_copy - i4_choose(n - x, k - i);
      x = x + 1;
    }

    t[i - 1] = x;
    x = x + 1;
  }

  return t;
}

int ksubset_revdoor_rank(int k, int n, int t[]) {
  bool check;
  int i;
  int rank;
  int s;
  check = ksubset_lex_check(k, n, t);

  if (!check) {
    cerr << "\n";
    cerr << "KSUBSET_REVDOOR_RANK(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  if ((k % 2) == 0) {
    rank = 0;
  } else {
    rank = -1;
  }

  s = 1;

  for (i = k; 1 <= i; i--) {
    rank = rank + s * i4_choose(t[i - 1], i);
    s = -s;
  }

  return rank;
}

void ksubset_revdoor_successor(int k, int n, int t[], int& rank) {
  bool check;
  int j;
  if (rank == -1) {
    i4vec_indicator1(k, t);
    rank = 0;
    return;
  }
  check = ksubset_lex_check(k, n, t);

  if (!check) {
    cerr << "\n";
    cerr << "KSUBSET_REVDOOR_SUCCESSOR(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  j = 0;

  for (;;) {
    if (0 < j || (k % 2) == 0) {
      j = j + 1;

      if (k < j) {
        t[k - 1] = k;
        rank = 0;
        return;
      }

      if (t[j - 1] != j) {
        t[j - 1] = t[j - 1] - 1;

        if (j != 1) {
          t[j - 2] = j - 1;
        }
        rank = rank + 1;
        return;
      }
    }
    j = j + 1;

    if (j < k) {
      if (t[j - 1] != t[j] - 1) {
        break;
      }
    } else {
      if (t[j - 1] != n) {
        break;
      }
    }
  }

  t[j - 1] = t[j - 1] + 1;

  if (j != 1) {
    t[j - 2] = t[j - 1] - 1;
  }

  rank = rank + 1;

  return;
}

int* ksubset_revdoor_unrank(int rank, int k, int n) {
  int i;
  int nksub;
  int rank_copy;
  int* t;
  int x;
  if (n < 1) {
    cerr << "\n";
    cerr << "KSUBSET_REVDOOR_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  if (k < 1 || n < k) {
    cerr << "\n";
    cerr << "KSUBSET_REVDOOR_UNRANK(): Fatal error!\n";
    cerr << "  Input K is illegal.\n";
    exit(1);
  }

  nksub = ksubset_enum(k, n);

  if (rank < 0 || nksub < rank) {
    cerr << "\n";
    cerr << "KSUBSET_REVDOOR_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }

  rank_copy = rank;

  t = new int[k];

  x = n;

  for (i = k; 1 <= i; i--) {
    while (rank_copy < i4_choose(x, i)) {
      x = x - 1;
    }

    t[i - 1] = x + 1;
    rank_copy = i4_choose(x + 1, i) - rank_copy - 1;
  }
  return t;
}

void marriage(int n, int prefer[], int rank[], int fiancee[], int next[]) {
  int i;
  int m;
  int temp;
  int w;
  for (i = 0; i < n; i++) {
    next[i] = 0;
  }
  for (i = 0; i < n; i++) {
    fiancee[i] = -1;
  }
  for (i = 1; i <= n; i++) {
    m = i;

    for (;;) {
      next[m - 1] = next[m - 1] + 1;

      w = prefer[m - 1 + (next[m - 1] - 1) * n];

      if (fiancee[w - 1] == -1) {
        fiancee[w - 1] = m;
        break;
      }

      if (rank[w - 1 + (m - 1) * n] < rank[w - 1 + (fiancee[w - 1] - 1) * n]) {
        temp = fiancee[w - 1];
        fiancee[w - 1] = m;
        m = temp;
      }
    }
  }
  return;
}

int mountain(int n, int x, int y) {
  int a;
  int b;
  int c;
  int value;
  if (n <= 0) {
    cerr << "\n";
    cerr << "MOUNTAIN(): Fatal error!\n";
    cerr << "  N <= 0.\n";
    cerr << "  N = " << n << "\n";
    exit(1);
  } else if (x < 0) {
    cerr << "\n";
    cerr << "MOUNTAIN(): Fatal error!\n";
    cerr << "  X < 0.\n";
    cerr << "  X = " << x << "\n";
    exit(1);
  } else if (2 * n < x) {
    cerr << "\n";
    cerr << "MOUNTAIN(): Fatal error!\n";
    cerr << "  2 * N < X.\n";
    cerr << "  X = " << x << "\n";
    cerr << "  N = " << n << "\n";
    exit(1);
  }
  if (y < 0) {
    value = 0;
  } else if (2 * n < x + y) {
    value = 0;
  } else if (((x + y) % 2) == 1) {
    value = 0;
  } else {
    a = 2 * n - x;
    b = n - (x + y) / 2;
    c = n - 1 - (x + y) / 2;
    value = i4_choose(a, b) - i4_choose(a, c);
  }
  return value;
}

int npart_enum(int n, int npart) {
  int* p;
  int value;

  if (n <= 0) {
    value = 0;
  } else if (npart <= 0 || n < npart) {
    value = 0;
  } else {
    p = npart_table(n, npart);

    value = p[n + npart * (n + 1)];

    delete[] p;
  }

  return value;
}

int* npart_rsf_lex_random(int n, int npart, int& seed) {
  int* a;
  int npartitions;
  int rank;

  npartitions = npart_enum(n, npart);

  rank = i4_uniform_ab(1, npartitions, seed);

  a = npart_rsf_lex_unrank(rank, n, npart);

  return a;
}

int npart_rsf_lex_rank(int n, int npart, int a[]) {
  int* b;
  bool check;
  int i;
  int ncopy;
  int npartcopy;
  int* p;
  int rank;
  check = part_rsf_check(n, npart, a);

  if (!check) {
    cerr << "\n";
    cerr << "NPART_RSF_LEX_RANK(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }
  p = npart_table(n, npart);
  b = new int[npart];

  for (i = 1; i <= npart; i++) {
    b[i - 1] = a[npart - i];
  }

  rank = 0;
  ncopy = n;
  npartcopy = npart;

  while (0 < ncopy && 0 < npartcopy) {
    if (b[npartcopy - 1] == 1) {
      ncopy = ncopy - 1;
      npartcopy = npartcopy - 1;
    } else {
      for (i = 0; i < npartcopy; i++) {
        b[i] = b[i] - 1;
      }
      rank = rank + p[ncopy - 1 + (npartcopy - 1) * (n + 1)];
      ncopy = ncopy - npartcopy;
    }
  }
  delete[] b;
  delete[] p;

  return rank;
}

void npart_rsf_lex_successor(int n, int npart, int a[], int& rank) {
  bool check;
  int d;
  int i;
  int j;
  if (rank == -1) {
    if (npart < 1) {
      cerr << "\n";
      cerr << "NPART_RSF_LEX_SUCCESSOR(): Fatal error!\n";
      cerr << "  NPART < 1.\n";
      exit(1);
    }

    for (i = 0; i < npart - 1; i++) {
      a[i] = 1;
    }
    a[npart - 1] = n - (npart - 1);

    rank = 0;
    return;
  }
  check = part_rsf_check(n, npart, a);

  if (!check) {
    cerr << "\n";
    cerr << "NPART_RSF_LEX_SUCCESSOR(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }
  i = 2;

  for (;;) {
    if (npart < i) {
      break;
    }

    if (a[npart - i] + 1 < a[npart - 1]) {
      break;
    }
    i = i + 1;
  }
  if (i == npart + 1) {
    for (i = 0; i < npart - 1; i++) {
      a[i] = 1;
    }
    a[npart - 1] = n - (npart - 1);

    rank = 0;
    return;
  } else {
    a[npart - i] = a[npart - i] + 1;
    d = -1;

    for (j = i - 1; 2 <= j; j--) {
      d = d + a[npart - j] - a[npart - i];
      a[npart - j] = a[npart - i];
    }
    a[npart - 1] = a[npart - 1] + d;
  }
  rank = rank + 1;

  return;
}

int* npart_rsf_lex_unrank(int rank, int n, int npart) {
  int* a;
  int i;
  int ncopy;
  int npartcopy;
  int npartitions;
  int* p;
  int rank_copy;
  if (n <= 0) {
    cerr << "\n";
    cerr << "NPART_RSF_LEX_UNRANK(): Fatal error!\n";
    cerr << "  The input N is illegal.\n";
    exit(1);
  }

  if (npart < 1 || n < npart) {
    cerr << "\n";
    cerr << "NPART_RSF_LEX_UNRANK(): Fatal error!\n";
    cerr << "  The input NPART is illegal.\n";
    exit(1);
  }

  npartitions = npart_enum(n, npart);

  if (rank < 0 || npartitions < rank) {
    cerr << "\n";
    cerr << "NPART_RSF_LEX_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }
  p = npart_table(n, npart);

  a = new int[npart];

  for (i = 0; i < npart; i++) {
    a[i] = 0;
  }

  rank_copy = rank;
  ncopy = n;
  npartcopy = npart;

  while (0 < ncopy) {
    if (rank_copy < p[ncopy - 1 + (npartcopy - 1) * (n + 1)]) {
      a[npart - npartcopy] = a[npart - npartcopy] + 1;
      ncopy = ncopy - 1;
      npartcopy = npartcopy - 1;
    } else {
      for (i = 1; i <= npartcopy; i++) {
        a[npart - i] = a[npart - i] + 1;
      }
      rank_copy = rank_copy - p[ncopy - 1 + (npartcopy - 1) * (n + 1)];
      ncopy = ncopy - npartcopy;
    }
  }
  return a;
}

void npart_sf_lex_successor(int n, int npart, int a[], int& rank) {
  bool check;
  int i;
  int indx;
  int temp;
  if (rank == -1) {
    i4vec_part2(n, npart, a);
    rank = 0;
    return;
  }
  check = part_sf_check(n, npart, a);

  if (!check) {
    cerr << "\n";
    cerr << "NPART_SF_LEX_SUCCESSOR(): Fatal error!\n";
    cerr << "  The partition is illegal.\n";
    exit(1);
  }
  for (i = npart; 1 <= i; i--) {
    if (1 < a[i - 1]) {
      indx = i;
      break;
    }
  }
  if (1 < indx) {
    a[indx - 1] = a[indx - 1] - 1;
    a[indx - 2] = a[indx - 2] + 1;
    indx = indx - 1;

    for (;;) {
      if (indx <= 1) {
        break;
      }

      if (a[indx - 1] <= a[indx - 2]) {
        break;
      }

      temp = a[indx - 1];
      a[indx - 1] = a[indx - 2];
      a[indx - 2] = temp;

      indx = indx - 1;
    }
    temp = 0;
    for (i = indx; i < npart; i++) {
      temp = temp + a[i];
    }
    i4vec_part2(temp, npart - indx, a + indx);

    rank = rank + 1;
  } else {
    i4vec_part2(n, npart, a);
    rank = 0;
  }

  return;
}

int* npart_table(int n, int npart) {
  int i;
  int j;
  int* p;

  p = new int[(n + 1) * (npart + 1)];

  p[0 + 0 * (n + 1)] = 1;
  for (i = 1; i <= n; i++) {
    p[i + 0 * (n + 1)] = 0;
  }

  for (i = 1; i <= n; i++) {
    for (j = 1; j <= npart; j++) {
      if (i < j) {
        p[i + j * (n + 1)] = 0;
      } else if (i < 2 * j) {
        p[i + j * (n + 1)] = p[i - 1 + (j - 1) * (n + 1)];
      } else {
        p[i + j * (n + 1)] = p[i - 1 + (j - 1) * (n + 1)] + p[i - j + j * (n + 1)];
      }
    }
  }
  return p;
}

int part_enum(int n) {
  int* p;
  int value;

  if (n < 0) {
    value = 0;
  } else {
    p = part_table(n);

    value = p[n];

    delete[] p;
  }
  return value;
}

bool part_rsf_check(int n, int npart, int a[]) {
  bool check;
  int i;

  check = true;

  if (n < 1) {
    check = false;
    return check;
  }

  if (npart < 1 || n < npart) {
    check = false;
    return check;
  }
  for (i = 0; i < npart; i++) {
    if (a[i] < 1 || n < a[i]) {
      check = false;
      return check;
    }
  }
  for (i = 1; i < npart; i++) {
    if (a[i] < a[i - 1]) {
      check = false;
      return check;
    }
  }
  if (i4vec_sum(npart, a) != n) {
    check = false;
    return check;
  }

  return check;
}

bool part_sf_check(int n, int npart, int a[]) {
  bool check;
  int i;

  check = true;

  if (n < 1) {
    check = false;
    return check;
  }

  if (npart < 1 || n < npart) {
    check = false;
    return check;
  }
  for (i = 0; i < npart; i++) {
    if (a[i] < 1 || n < a[i]) {
      check = false;
      return check;
    }
  }
  for (i = 1; i < npart; i++) {
    if (a[i - 1] < a[i]) {
      check = false;
      return check;
    }
  }
  if (i4vec_sum(npart, a) != n) {
    check = false;
    return check;
  }

  return check;
}

int* part_sf_conjugate(int n, int npart, int a[], int& npart2) {
  int* b;
  bool check;
  int i;
  int j;
  check = part_sf_check(n, npart, a);

  if (!check) {
    cerr << "\n";
    cerr << "PART_SF_CHECK(): Fatal error!\n";
    cerr << "  The partition is illegal.\n";
    exit(1);
  }

  npart2 = a[0];

  b = new int[n];

  for (i = 0; i < npart2; i++) {
    b[i] = 0;
  }

  for (i = 0; i < npart; i++) {
    for (j = 0; j < a[i]; j++) {
      b[j] = b[j] + 1;
    }
  }

  return b;
}

int part_sf_majorize(int n, int nparta, int a[], int npartb, int b[]) {
  bool check;
  int i;
  int result;
  int suma;
  int sumb;
  check = part_sf_check(n, nparta, a);

  if (!check) {
    cerr << "\n";
    cerr << "PART_SF_MAJORIZE(): Fatal error!\n";
    cerr << "  The partition is illegal.\n";
    exit(1);
  }

  check = part_sf_check(n, npartb, b);

  if (!check) {
    cerr << "\n";
    cerr << "PART_SF_MAJORIZE(): Fatal error!\n";
    cerr << "  The partition is illegal.\n";
    exit(1);
  }

  result = 0;
  suma = 0;
  sumb = 0;

  for (i = 0; i < i4_min(nparta, npartb); i++) {
    if (i < nparta) {
      suma = suma + a[i];
    }

    if (i < npartb) {
      sumb = sumb + b[i];
    }

    if (result == -1) {
      if (sumb < suma) {
        result = -2;
        return result;
      }
    } else if (result == 0) {
      if (suma < sumb) {
        result = -1;
      } else if (sumb < suma) {
        result = +1;
      }
    } else if (result == +1) {
      if (suma < sumb) {
        result = +2;
        return result;
      }
    }
  }

  return result;
}

void part_successor(int n, int& npart, int a[], int& rank) {
  int asum;
  bool check;
  int i;
  int ihi;
  int j;
  if (rank == -1) {
    for (i = 0; i < n; i++) {
      a[i] = 1;
    }
    npart = n;
    rank = 0;
    return;
  }
  check = part_sf_check(n, npart, a);

  if (!check) {
    cerr << "\n";
    cerr << "PART_SUCCESSOR(): Fatal error!\n";
    cerr << "  The partition is illegal.\n";
    exit(1);
  }
  ihi = npart - 1;

  for (i = ihi; 2 <= i; i--) {
    if (a[i - 1] < a[i - 2]) {
      asum = -1;
      for (j = i + 1; j <= npart; j++) {
        asum = asum + a[j - 1];
      }
      a[i - 1] = a[i - 1] + 1;
      for (j = i + 1; j <= npart; j++) {
        a[j - 1] = 0;
      }
      npart = i + asum;
      for (j = i + 1; j <= npart; j++) {
        a[j - 1] = 1;
      }
      rank = rank + 1;
      return;
    }
  }
  if (2 <= npart) {
    a[0] = a[0] + 1;
    for (j = 2; j <= npart; j++) {
      a[j - 1] = 0;
    }
    npart = n - a[0] + 1;
    for (j = 2; j <= npart; j++) {
      a[j - 1] = 1;
    }
    rank = rank + 1;
  } else if (npart == 1) {
    for (i = 0; i < n; i++) {
      a[i] = 1;
    }
    npart = n;
    rank = 0;
  }
  return;
}

int* part_table(int n) {
  int i;
  int j;
  int* p;
  int psum;
  int sign;
  int w;
  int wprime;

  p = new int[n + 1];

  p[0] = 1;

  if (n <= 0) {
    return p;
  }

  p[1] = 1;

  for (i = 2; i <= n; i++) {
    sign = 1;
    psum = 0;
    w = 1;
    j = 1;
    wprime = w + j;

    while (w < n) {
      if (0 <= i - w) {
        if (sign == 1) {
          psum = psum + p[i - w];
        } else {
          psum = psum - p[i - w];
        }
      }

      if (wprime <= i) {
        if (sign == 1) {
          psum = psum + p[i - wprime];
        } else {
          psum = psum - p[i - wprime];
        }
      }
      w = w + 3 * j + 1;
      j = j + 1;
      wprime = w + j;
      sign = -sign;
    }
    p[i] = psum;
  }
  return p;
}

int* partition_greedy(int n, int a[]) {
  int i;
  int* indx;
  int j;
  int sums[2];

  sums[0] = 0;
  sums[1] = 0;

  i4vec_sort_insert_d(n, a);

  indx = new int[n];

  for (i = 0; i < n; i++) {
    if (sums[0] < sums[1]) {
      j = 0;
    } else {
      j = 1;
    }
    indx[i] = j;
    sums[j] = sums[j] + a[i];
  }
  return indx;
}

int partn_enum(int n, int nmax) {
  int* p;
  int value;

  if (n <= 0) {
    value = 0;
  } else if (nmax <= 0 || n < nmax) {
    value = 0;
  } else {
    p = npart_table(n, nmax);

    value = p[n + nmax * (n + 1)];

    delete[] p;
  }

  return value;
}

bool partn_sf_check(int n, int nmax, int npart, int a[]) {
  int asum;
  bool check;
  int i;

  check = true;

  if (n < 1) {
    check = false;
    return check;
  }

  if (nmax < 1 || n < nmax) {
    check = false;
    return check;
  }

  if (npart < 1 || n < npart) {
    check = false;
    return check;
  }
  if (a[0] != nmax) {
    check = false;
    return check;
  }
  for (i = 0; i < npart; i++) {
    if (a[i] < 1 || n < a[i]) {
      check = false;
      return check;
    }
  }
  for (i = 1; i < npart; i++) {
    if (a[i - 1] < a[i]) {
      check = false;
      return check;
    }
  }
  asum = i4vec_sum(npart, a);

  if (asum != n) {
    check = false;
    return check;
  }

  return check;
}

void partn_successor(int n, int nmax, int& npart, int a[], int& rank) {
  bool check;
  int i;
  int index;
  int temp;
  if (rank == -1) {
    npart = n + 1 - nmax;
    a[0] = nmax;
    for (i = 1; i < npart; i++) {
      a[i] = 1;
    }
    rank = 0;
    return;
  }
  check = partn_sf_check(n, nmax, npart, a);

  if (!check) {
    cerr << "\n";
    cerr << "PARTN_SUCCESSOR(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }
  if (1 < npart) {
    if (a[npart - 2] < nmax) {
      a[npart - 1] = a[npart - 1] - 1;
      a[npart - 2] = a[npart - 2] + 1;
      index = npart - 1;

      for (;;) {
        if (index <= 1) {
          break;
        }

        if (a[index - 1] <= a[index - 2]) {
          break;
        }

        temp = a[index - 2];
        a[index - 2] = a[index - 1];
        a[index - 1] = temp;

        index = index - 1;
      }
      temp = 0;
      for (i = index; i < npart; i++) {
        temp = temp + a[i];
      }
      npart = index + temp;
      for (i = index; i < npart; i++) {
        a[i] = 1;
      }
      rank = rank + 1;
      return;
    }
  } else {
    npart = n + 1 - nmax;
    a[0] = nmax;
    for (i = 1; i < npart; i++) {
      a[i] = 1;
    }
    rank = 0;
    return;
  }

  return;
}

bool perm_check(int n, int p[]) {
  bool check;
  int i;
  int ifind;
  int iseek;

  check = true;

  if (n < 1) {
    check = false;
    return check;
  }

  for (i = 0; i < n; i++) {
    if (p[i] < 1 || n < p[i]) {
      check = false;
      return check;
    }
  }

  for (iseek = 1; iseek <= n; iseek++) {
    ifind = -1;
    for (i = 0; i < n; i++) {
      if (p[i] == iseek) {
        ifind = i;
        break;
      }
    }

    if (ifind == -1) {
      check = false;
      return check;
    }
  }
  return check;
}

int perm_enum(int n) {
  int value;

  value = i4_factorial(n);

  return value;
}

int* perm_inv(int n, int p[]) {
  bool check;
  int i;
  int* pinv;
  check = perm_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PERM_INV(): Fatal error!\n";
    cerr << "  Permutation is illegal.\n";
    exit(1);
  }

  pinv = new int[n];

  for (i = 0; i < n; i++) {
    pinv[p[i] - 1] = i + 1;
  }

  return pinv;
}

int perm_lex_rank(int n, int p[]) {
  bool check;
  int i;
  int j;
  int* pcopy;
  int rank;
  check = perm_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PERM_LEX_RANK(): Fatal error!\n";
    cerr << "  Permutation is illegal.\n";
    exit(1);
  }
  rank = 0;
  pcopy = new int[n];

  for (i = 0; i < n; i++) {
    pcopy[i] = p[i];
  }

  for (j = 0; j < n; j++) {
    rank = rank + (pcopy[j] - 1) * i4_factorial(n - 1 - j);
    for (i = j + 1; i < n; i++) {
      if (pcopy[j] < pcopy[i]) {
        pcopy[i] = pcopy[i] - 1;
      }
    }
  }
  delete[] pcopy;

  return rank;
}

void perm_lex_successor(int n, int p[], int& rank) {
  bool check;
  int i;
  int j;
  int temp;
  if (rank == -1) {
    i4vec_indicator1(n, p);
    rank = 0;
    return;
  }
  check = perm_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PERM_LEX_SUCCESSOR(): Fatal error!\n";
    cerr << "  Permutation is illegal.\n";
    exit(1);
  }
  i = n - 1;

  for (;;) {
    if (i <= 0) {
      break;
    }

    if (p[i - 1] <= p[i]) {
      break;
    }
    i = i - 1;
  }
  if (i == 0) {
    i4vec_indicator1(n, p);
    rank = 0;
  } else {
    j = n;
    while (p[j - 1] < p[i - 1]) {
      j = j - 1;
    }
    temp = p[i - 1];
    p[i - 1] = p[j - 1];
    p[j - 1] = temp;
    i4vec_reverse(n - i, p + i);

    rank = rank + 1;
  }

  return;
}

int* perm_lex_unrank(int rank, int n) {
  int d;
  int i;
  int j;
  int nperm;
  int* p;
  int rank_copy;
  if (n < 1) {
    cerr << "\n";
    cerr << "PERM_LEX_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  nperm = perm_enum(n);

  if (rank < 0 || nperm < rank) {
    cerr << "\n";
    cerr << "PERM_LEX_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }

  rank_copy = rank;

  p = new int[n];

  p[n - 1] = 1;

  for (j = 1; j <= n - 1; j++) {
    d = (rank_copy % i4_factorial(j + 1)) / i4_factorial(j);
    rank_copy = rank_copy - d * i4_factorial(j);
    p[n - j - 1] = d + 1;

    for (i = n - j + 1; i <= n; i++) {
      if (d < p[i - 1]) {
        p[i - 1] = p[i - 1] + 1;
      }
    }
  }
  return p;
}

int* perm_mul(int n, int p[], int q[]) {
  bool check;
  int i;
  int* r;
  check = perm_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PERM_MUL(): Fatal error!\n";
    cerr << "  Permutation is illegal.\n";
    exit(1);
  }

  check = perm_check(n, q);

  if (!check) {
    cerr << "\n";
    cerr << "PERM_MUL(): Fatal error!\n";
    cerr << "  Permutation is illegal.\n";
    exit(1);
  }
  r = new int[n];

  for (i = 0; i < n; i++) {
    r[i] = p[q[i] - 1];
  }

  return r;
}

int perm_parity(int n, int p[]) {
  int* a;
  int c;
  bool check;
  int i;
  int j;
  int parity;
  check = perm_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PERM_PARITY(): Fatal error!\n";
    cerr << "  Permutation is illegal.\n";
    exit(1);
  }

  a = new int[n];

  for (i = 0; i < n; i++) {
    a[i] = 0;
  }

  c = 0;

  for (j = 1; j <= n; j++) {
    if (a[j - 1] == 0) {
      c = c + 1;
      a[j - 1] = 1;
      i = j;

      while (p[i - 1] != j) {
        i = p[i - 1];
        a[i - 1] = 1;
      }
    }
  }

  parity = (n - c) % 2;

  delete[] a;

  return parity;
}

void perm_print(int n, int p[], string title) {
  int i;
  int ihi;
  int ilo;
  int inc = 20;

  if (s_len_trim(title) != 0) {
    cout << "\n";
    cout << title << "\n";

    for (ilo = 0; ilo < n; ilo = ilo + inc) {
      ihi = ilo + inc;
      if (n < ihi) {
        ihi = n;
      }
      cout << "\n";
      cout << "  ";
      for (i = ilo; i < ihi; i++) {
        cout << setw(4) << i;
      }
      cout << "\n";
      cout << "  ";
      for (i = ilo; i < ihi; i++) {
        cout << setw(4) << p[i];
      }
      cout << "\n";
    }
  } else {
    for (ilo = 0; ilo < n; ilo = ilo + inc) {
      ihi = ilo + inc;
      if (n < ihi) {
        ihi = n;
      }
      cout << "  ";
      for (i = ilo; i < ihi; i++) {
        cout << setw(4) << p[i];
      }
      cout << "\n";
    }
  }

  return;
}

int* perm_random(int n, int& seed) {
  int i;
  int j;
  int* p;
  int t;

  p = new int[n];

  for (i = 0; i < n; i++) {
    p[i] = i + 1;
  }

  for (i = 0; i < n - 1; i++) {
    j = i4_uniform_ab(i, n - 1, seed);

    t = p[i];
    p[i] = p[j];
    p[j] = t;
  }

  return p;
}

int perm_tj_rank(int n, int p[]) {
  bool check;
  int i;
  int j;
  int k;
  int rank;
  check = perm_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PERM_TJ_RANK(): Fatal error!\n";
    cerr << "  Permutation is illegal.\n";
    exit(1);
  }

  rank = 0;

  for (j = 2; j <= n; j++) {
    k = 1;
    i = 1;

    while (p[i - 1] != j) {
      if (p[i - 1] < j) {
        k = k + 1;
      }
      i = i + 1;
    }

    if ((rank % 2) == 0) {
      rank = j * rank + j - k;
    } else {
      rank = j * rank + k - 1;
    }
  }

  return rank;
}

void perm_tj_successor(int n, int p[], int& rank) {
  bool check;
  int d;
  bool done;
  int i;
  int m;
  int par;
  int* q;
  int st;
  int temp;
  if (rank == -1) {
    i4vec_indicator1(n, p);
    rank = 0;
    return;
  }
  check = perm_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PERM_TJ_SUCCESSOR(): Fatal error!\n";
    cerr << "  Permutation is illegal.\n";
    exit(1);
  }

  q = new int[n];

  st = 0;
  for (i = 0; i < n; i++) {
    q[i] = p[i];
  }
  done = false;
  m = n;

  while (1 < m && !done) {
    d = 1;
    while (q[d - 1] != m) {
      d = d + 1;
    }

    for (i = d; i < m; i++) {
      q[i - 1] = q[i];
    }

    par = perm_parity(m - 1, q);

    if (par == 1) {
      if (d == m) {
        m = m - 1;
      } else {
        temp = p[st + d - 1];
        p[st + d - 1] = p[st + d];
        p[st + d] = temp;
        done = true;
      }
    } else {
      if (d == 1) {
        m = m - 1;
        st = st + 1;
      } else {
        temp = p[st + d - 1];
        p[st + d - 1] = p[st + d - 2];
        p[st + d - 2] = temp;
        done = true;
      }
    }
  }
  if (m == 1) {
    i4vec_indicator1(n, p);
    rank = 0;
    delete[] q;
    return;
  }

  rank = rank + 1;

  delete[] q;

  return;
}

int* perm_tj_unrank(int rank, int n) {
  int i;
  int j;
  int k;
  int jhi;
  int nperm;
  int* p;
  int r1;
  int r2;
  if (n < 1) {
    cerr << "\n";
    cerr << "PERM_TJ_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  nperm = perm_enum(n);

  if (rank < 0 || nperm < rank) {
    cerr << "\n";
    cerr << "PERM_TJ_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }

  p = new int[n];

  p[0] = 1;
  r2 = 0;

  for (j = 2; j <= n; j++) {
    r1 = (rank * i4_factorial(j)) / i4_factorial(n);
    k = r1 - j * r2;

    if ((r2 % 2) == 0) {
      jhi = j - k;
    } else {
      jhi = k + 1;
    }

    for (i = j - 1; jhi <= i; i--) {
      p[i] = p[i - 1];
    }
    p[jhi - 1] = j;

    r2 = r1;
  }

  return p;
}

void perm_to_cycle(int n, int p[], int& ncycle, int t[], int index[]) {
  bool check;
  int i;
  int j;
  int nset;
  check = perm_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PERM_TO_CYCLE(): Fatal error!\n";
    cerr << "  Permutation is illegal.\n";
    exit(1);
  }
  ncycle = 0;
  for (i = 0; i < n; i++) {
    index[i] = 0;
  }
  for (i = 0; i < n; i++) {
    t[i] = 0;
  }
  nset = 0;
  for (i = 1; i <= n; i++) {
    if (0 < p[i - 1]) {
      ncycle = ncycle + 1;
      index[ncycle - 1] = 1;

      nset = nset + 1;
      t[nset - 1] = p[i - 1];
      p[i - 1] = -p[i - 1];

      for (;;) {
        j = t[nset - 1];

        if (p[j - 1] < 0) {
          break;
        }

        index[ncycle - 1] = index[ncycle - 1] + 1;

        nset = nset + 1;
        t[nset - 1] = p[j - 1];
        p[j - 1] = -p[j - 1];
      }
    }
  }
  for (i = 0; i < n; i++) {
    p[i] = -p[i];
  }

  return;
}

bool pruefer_check(int n, int p[]) {
  bool check;
  int i;

  check = true;

  if (n < 3) {
    check = false;
    return check;
  }

  for (i = 0; i < n - 2; i++) {
    if (p[i] < 1 || n < p[i]) {
      check = false;
      return check;
    }
  }
  return check;
}

int pruefer_enum(int n) {
  int value;

  if (n < 2) {
    value = 0;
  } else if (n == 2) {
    value = 1;
  } else {
    value = i4_power(n, n - 2);
  }

  return value;
}

int* pruefer_random(int n) {
  int i;
  int* p;

  p = new int[n - 2];

  for (i = 0; i < n - 2; i++) {
    p[i] = 1 + (rand() % n);
  }
  return p;
}

int pruefer_rank(int n, int p[]) {
  bool check;
  int i;
  int k;
  int rank;
  check = pruefer_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PRUEFER_RANK(): Fatal error!\n";
    cerr << "  Input array is illegal.\n";
    exit(1);
  }

  rank = 0;
  k = 1;
  for (i = n - 3; 0 <= i; i--) {
    rank = rank + k * (p[i] - 1);
    k = k * n;
  }

  return rank;
}

void pruefer_successor(int n, int p[], int& rank) {
  bool check;
  int i;
  int j;
  if (rank == -1) {
    for (i = 0; i < n - 2; i++) {
      p[i] = 1;
    }
    rank = 0;
    return;
  }
  check = pruefer_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PRUEFER_SUCCESSOR(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  j = n - 2;

  for (;;) {
    if (p[j - 1] != n) {
      break;
    }

    j = j - 1;

    if (j <= 0) {
      break;
    }
  }

  if (j != 0) {
    p[j - 1] = p[j - 1] + 1;
    for (i = j + 1; i <= n - 2; i++) {
      p[i - 1] = 1;
    }
    rank = rank + 1;
  } else {
    for (i = 0; i < n - 2; i++) {
      p[i] = 1;
    }
    rank = 0;
  }
  return;
}

void pruefer_to_tree(int n, int p[], int t[]) {
  bool check;
  int* d;
  int i;
  int j;
  int x;
  int y;
  check = pruefer_check(n, p);

  if (!check) {
    cerr << "\n";
    cerr << "PRUEFER_TO_TREE(): Fatal error!\n";
    cerr << "  The input array is illegal!\n";
    exit(1);
  }
  for (j = 0; j < n - 1; j++) {
    for (i = 0; i < 2; i++) {
      t[i + j * 2] = 0;
    }
  }

  d = new int[n];

  for (i = 0; i < n; i++) {
    d[i] = 1;
  }

  for (i = 0; i < n - 2; i++) {
    d[p[i] - 1] = d[p[i] - 1] + 1;
  }

  for (i = 1; i <= n - 1; i++) {
    x = n;
    while (d[x - 1] != 1) {
      x = x - 1;
    }

    if (i == n - 1) {
      y = 1;
    } else {
      y = p[i - 1];
    }

    d[x - 1] = d[x - 1] - 1;
    d[y - 1] = d[y - 1] - 1;

    t[0 + (i - 1) * 2] = x;
    t[1 + (i - 1) * 2] = y;
  }

  delete[] d;

  return;
}

int* pruefer_to_tree_new(int n, int p[]) {
  int* t;

  t = new int[2 * (n - 1)];

  pruefer_to_tree(n, p, t);

  return t;
}

int* pruefer_unrank(int rank, int n) {
  int i;
  int ncode;
  int* p;
  int rank_copy;
  if (n < 1) {
    cerr << "\n";
    cerr << "PRUEFER_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  if (n < 3) {
    p = NULL;
    return p;
  }

  ncode = pruefer_enum(n);

  if (rank < 0 || ncode < rank) {
    cerr << "\n";
    cerr << "PRUEFER_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }

  rank_copy = rank;
  p = new int[n - 2];

  for (i = n - 3; 0 <= i; i--) {
    p[i] = (rank_copy % n) + 1;
    rank_copy = (rank_copy - p[i] + 1) / n;
  }
  return p;
}

void queens(int n, int iarray[], int k, int& nstack, int istack[], int maxstack) {
  bool diag;
  int irow;
  int jcol;
  int ncan;
  bool row;

  ncan = 0;

  for (irow = 1; irow <= n; irow++) {
    row = false;

    for (jcol = 1; jcol <= k - 1; jcol++) {
      if (iarray[jcol - 1] == irow) {
        row = true;
      }
    }

    if (!row) {
      diag = false;

      for (jcol = 1; jcol <= k - 1; jcol++) {
        if (irow == iarray[jcol - 1] + k - jcol || irow == iarray[jcol - 1] - (k - jcol)) {
          diag = true;
        }
      }

      if (!diag) {
        ncan = ncan + 1;
        istack[nstack] = irow;
        nstack = nstack + 1;
      }
    }
  }

  istack[nstack] = ncan;
  nstack = nstack + 1;

  return;
}

double r8_choose(int n, int k) {
  double arg;
  double fack;
  double facn;
  double facnmk;
  double value;

  if (n < 0) {
    value = 0.0;
  } else if (k == 0) {
    value = 1.0;
  } else if (k == 1) {
    value = (double)(n);
  } else if (1 < k && k < n - 1) {
    arg = (double)(n + 1);
    facn = lgamma(arg);

    arg = (double)(k + 1);
    fack = lgamma(arg);

    arg = (double)(n - k + 1);
    facnmk = lgamma(arg);

    value = r8_nint(exp(facn - fack - facnmk));
  } else if (k == n - 1) {
    value = (double)(n);
  } else if (k == n) {
    value = 1.0;
  } else {
    value = 0.0;
  }

  return value;
}

int r8_nint(double x) {
  int value;

  if (x < 0.0) {
    value = -(int)(fabs(x) + 0.5);
  } else {
    value = (int)(fabs(x) + 0.5);
  }

  return value;
}

void r8vec_backtrack(int n, int maxstack, double stack[], double x[], int& indx, int& k, int& nstack, int ncan[]) {
  if (indx == 0) {
    k = 1;
    nstack = 0;
    indx = 2;
    return;
  }
  for (;;) {
    if (0 < ncan[k - 1]) {
      x[k - 1] = stack[nstack - 1];
      nstack = nstack - 1;

      ncan[k - 1] = ncan[k - 1] - 1;

      if (k != n) {
        k = k + 1;
        indx = 2;
      } else {
        indx = 1;
      }

      break;
    } else {
      k = k - 1;

      if (k <= 0) {
        indx = 3;
        break;
      }
    }
  }

  return;
}

double r8vec_dot_product(int n, double a1[], double a2[]) {
  int i;
  double value;

  value = 0.0;
  for (i = 0; i < n; i++) {
    value = value + a1[i] * a2[i];
  }
  return value;
}

bool rgf_check(int m, int f[]) {
  bool check;
  int fmax;
  int i;

  check = true;

  if (m <= 0) {
    check = false;
    return check;
  }

  fmax = 0;
  for (i = 0; i < m; i++) {
    if (f[i] <= 0 || fmax + 1 < f[i]) {
      check = false;
      return check;
    }
    fmax = i4_max(fmax, f[i]);
  }

  return check;
}

int rgf_enum(int m) {
  int* b;
  int i;
  int j;
  int value;

  if (m < 0) {
    value = 0;
  } else if (m == 0) {
    value = 1;
  } else {
    b = new int[m + 1];
    b[0] = 1;
    for (j = 1; j <= m; j++) {
      b[j] = 0;
      for (i = 0; i < j; i++) {
        b[j] = b[j] + i4_choose(j - 1, i) * b[i];
      }
    }
    value = b[m];

    delete[] b;
  }
  return value;
}

int* rgf_g_table(int m) {
  int* d;
  int i;
  int j;

  d = new int[(m + 1) * (m + 1)];

  for (j = 0; j <= m; j++) {
    d[0 + j * (m + 1)] = 1;
  }

  for (i = 1; i <= m; i++) {
    for (j = 0; j <= m; j++) {
      if (j <= m - i) {
        d[i + j * (m + 1)] = j * d[i - 1 + j * (m + 1)] + d[i - 1 + (j + 1) * (m + 1)];
      } else {
        d[i + j * (m + 1)] = 0;
      }
    }
  }
  return d;
}

int rgf_rank(int m, int f[]) {
  bool check;
  int* d;
  int i;
  int j;
  int rank;
  check = rgf_check(m, f);

  if (!check) {
    cerr << "\n";
    cerr << "RGF_RANK(): Fatal error!\n";
    cerr << "  The input array is illegal!\n";
    exit(1);
  }
  d = rgf_g_table(m);

  rank = 0;
  j = 1;
  for (i = 2; i <= m; i++) {
    rank = rank + (f[i - 1] - 1) * d[m - i + j * (m + 1)];
    j = i4_max(j, f[i - 1]);
  }

  delete[] d;

  return rank;
}

void rgf_successor(int m, int f[], int& rank) {
  bool check;
  int fmax;
  int i;
  int j;
  if (rank == -1) {
    for (i = 0; i < m; i++) {
      f[i] = 1;
    }
    rank = 0;
    return;
  }
  check = rgf_check(m, f);

  if (!check) {
    cerr << "\n";
    cerr << "RGF_SUCCESSOR(): Fatal error!\n";
    cerr << "  The input array is illegal!\n";
    exit(1);
  }
  for (i = m; 2 <= i; i--) {
    fmax = 1;
    for (j = 2; j < i; j++) {
      fmax = i4_max(fmax, f[j - 1]);
    }
    if (f[i - 1] != fmax + 1) {
      f[i - 1] = f[i - 1] + 1;
      for (j = i + 1; j <= m; j++) {
        f[j - 1] = 1;
      }
      rank = rank + 1;
      return;
    }
  }
  for (i = 0; i < m; i++) {
    f[i] = 1;
  }
  rank = 0;

  return;
}

void rgf_to_setpart(int m, int f[], int& nsub, int s[], int index[]) {
  bool check;
  int i;
  int j;
  int k;
  check = rgf_check(m, f);

  if (!check) {
    cerr << "\n";
    cerr << "RGF_TO_SETPART(): Fatal error!\n";
    cerr << "  The input array is illegal!\n";
    exit(1);
  }
  nsub = i4vec_max(m, f);
  for (i = 0; i < m; i++) {
    s[i] = 0;
  }
  for (i = 0; i < nsub; i++) {
    index[i] = 0;
  }
  k = 0;
  for (i = 1; i <= nsub; i++) {
    for (j = 1; j <= m; j++) {
      if (f[j - 1] == i) {
        k = k + 1;
        s[k - 1] = j;
      }
    }
    index[i - 1] = k;
  }
  return;
}

int* rgf_unrank(int rank, int m) {
  int* d;
  int* f;
  int i;
  int j;
  int nrgf;
  int rank_copy;
  if (m < 1) {
    cerr << "\n";
    cerr << "RGF_UNRANK(): Fatal error!\n";
    cerr << "  Input M is illegal.\n";
    exit(1);
  }

  nrgf = rgf_enum(m);

  if (rank < 0 || nrgf < rank) {
    cerr << "\n";
    cerr << "RGF_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }
  d = rgf_g_table(m);

  f = new int[m];

  rank_copy = rank;
  j = 1;
  f[0] = 1;

  for (i = 2; i <= m; i++) {
    if (j * d[m - i + j * (m + 1)] <= rank_copy) {
      f[i - 1] = j + 1;
      rank_copy = rank_copy - j * d[m - i + j * (m + 1)];
      j = j + 1;
    } else {
      f[i - 1] = 1 + (rank_copy / d[m - i + j * (m + 1)]);
      rank_copy = rank_copy % d[m - i + j * (m + 1)];
    }
  }

  delete[] d;

  return f;
}

int s_len_trim(string s) {
  int n;

  n = s.length();

  while (0 < n) {
    if (s[n - 1] != ' ') {
      return n;
    }
    n = n - 1;
  }

  return n;
}

bool setpart_check(int m, int nsub, int s[], int index[]) {
  bool check;
  int i;
  int imin;
  int j;

  check = true;
  if (m < 1) {
    check = false;
    return check;
  }
  if (nsub < 1) {
    check = false;
    return check;
  }
  imin = 0;
  for (i = 0; i < nsub; i++) {
    if (index[i] <= imin || m < index[i]) {
      check = false;
      return check;
    }
    imin = index[i];
  }
  for (i = 0; i < m; i++) {
    if (s[i] <= 0 || m < s[i]) {
      check = false;
      return check;
    }

    for (j = 0; j < i; j++) {
      if (s[j] == s[i]) {
        check = false;
        return check;
      }
    }
  }

  return check;
}

int setpart_enum(int m) {
  int* b;
  int i;
  int j;
  int value;

  if (m < 0) {
    value = 0;
  } else if (m == 0) {
    value = 1;
  } else {
    b = new int[m + 1];
    b[0] = 1;
    for (j = 1; j <= m; j++) {
      b[j] = 0;
      for (i = 0; i < j; i++) {
        b[j] = b[j] + i4_choose(j - 1, i) * b[i];
      }
    }
    value = b[m];

    delete[] b;
  }

  return value;
}

int* setpart_to_rgf(int m, int nsub, int s[], int index[]) {
  bool check;
  int* f;
  int i;
  int k;
  int khi;
  int klo;
  check = setpart_check(m, nsub, s, index);

  if (!check) {
    cerr << "\n";
    cerr << "SETPART_TO_RGF(): Fatal error!\n";
    cerr << "  The input array is illegal.\n";
    exit(1);
  }

  f = new int[m];

  khi = 0;
  for (i = 1; i <= nsub; i++) {
    klo = khi + 1;
    khi = index[i - 1];
    for (k = klo; k <= khi; k++) {
      f[s[k - 1] - 1] = i;
    }
  }
  return f;
}

int* stirling_numbers1(int m, int n) {
  int i;
  int j;
  int* s;

  s = new int[(m + 1) * (n + 1)];

  s[0 + 0 * (m + 1)] = 1;
  for (j = 1; j <= n; j++) {
    s[0 + j * (m + 1)] = 0;
  }

  for (i = 1; i <= m; i++) {
    s[i + 0 * (m + 1)] = 0;
  }
  for (i = 0; i <= i4_min(m, n - 1); i++) {
    s[i + (i + 1) * (m + 1)] = 0;
  }

  for (i = 1; i <= m; i++) {
    for (j = 1; j <= n; j++) {
      if (j <= i) {
        s[i + j * (m + 1)] = s[i - 1 + (j - 1) * (m + 1)] - (i - 1) * s[i - 1 + j * (m + 1)];
      } else {
        s[i + j * (m + 1)] = 0;
      }
    }
  }

  return s;
}

int* stirling_numbers2(int m, int n) {
  int i;
  int j;
  int* s;

  s = new int[(m + 1) * (n + 1)];

  s[0 + 0 * (m + 1)] = 1;
  for (j = 1; j <= n; j++) {
    s[0 + j * (m + 1)] = 0;
  }
  for (i = 1; i <= m; i++) {
    s[i + 0 * (m + 1)] = 0;
  }
  for (i = 0; i <= i4_min(m, n - 1); i++) {
    s[i + (i + 1) * (m + 1)] = 0;
  }

  for (i = 1; i <= m; i++) {
    for (j = 1; j <= n; j++) {
      if (j <= i) {
        s[i + j * (m + 1)] = j * s[i - 1 + j * (m + 1)] + s[i - 1 + (j - 1) * (m + 1)];
      } else {
        s[i + j * (m + 1)] = 0;
      }
    }
  }

  return s;
}

bool subset_check(int n, int t[]) {
  bool check;
  int i;

  check = true;

  if (n < 1) {
    check = false;
    return check;
  }

  for (i = 0; i < n; i++) {
    if (t[i] != 0 && t[i] != 1) {
      check = false;
      return check;
    }
  }
  return check;
}

int subset_colex_rank(int n, int t[]) {
  bool check;
  int i;
  int rank;
  check = subset_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_COLEX_RANK(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  rank = 0;

  for (i = 0; i < n; i++) {
    if (t[i] == 1) {
      rank = rank + i4_power(2, i);
    }
  }
  return rank;
}

void subset_colex_successor(int n, int t[], int& rank) {
  bool check;
  int i;
  if (rank == -1) {
    for (i = 0; i < n; i++) {
      t[i] = 0;
    }
    rank = 0;
    return;
  }
  check = subset_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_COLEX_SUCCESSOR(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  for (i = 0; i < n; i++) {
    if (t[i] == 0) {
      t[i] = 1;
      rank = rank + 1;
      return;
    } else {
      t[i] = 0;
    }
  }
  rank = 0;

  return;
}

int* subset_colex_unrank(int rank, int n) {
  int i;
  int nsub;
  int rank_copy;
  int* t;
  if (n < 1) {
    cerr << "\n";
    cerr << "SUBSET_COLEX_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  nsub = subset_enum(n);

  if (rank < 0 || nsub < rank) {
    cerr << "\n";
    cerr << "SUBSET_COLEX_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }

  rank_copy = rank;
  t = new int[n];

  for (i = 0; i < n; i++) {
    if ((rank_copy % 2) == 1) {
      t[i] = 1;
    } else {
      t[i] = 0;
    }
    rank_copy = rank_copy / 2;
  }
  return t;
}

int* subset_complement(int n, int a[]) {
  int* b;
  bool check;
  int i;
  check = subset_check(n, a);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_COMPLEMENT(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  b = new int[n];

  for (i = 0; i < n; i++) {
    b[i] = 1 - a[i];
  }
  return b;
}

int subset_distance(int n, int t1[], int t2[]) {
  bool check;
  int dist;
  int i;
  check = subset_check(n, t1);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_DISTANCE(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  check = subset_check(n, t2);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_DISTANCE(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  dist = 0;

  for (i = 0; i < n; i++) {
    if ((t1[i] == 0 && t2[i] != 0) || (t1[i] != 0 && t2[i] == 0)) {
      dist = dist + 1;
    }
  }
  return dist;
}

int subset_enum(int n) {
  int value;

  value = i4_power(2, n);

  return value;
}

int* subset_intersect(int n, int a[], int b[]) {
  int* c;
  bool check;
  int i;
  check = subset_check(n, a);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_INTERSECTION(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  check = subset_check(n, b);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_INTERSECTION(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  c = new int[n];

  for (i = 0; i < n; i++) {
    c[i] = i4_min(a[i], b[i]);
  }
  return c;
}

int subset_lex_rank(int n, int t[]) {
  bool check;
  int i;
  int rank;
  check = subset_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_LEX_RANK(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  rank = 0;

  for (i = 0; i < n; i++) {
    if (t[i] == 1) {
      rank = rank + i4_power(2, n - i - 1);
    }
  }

  return rank;
}

void subset_lex_successor(int n, int t[], int& rank) {
  bool check;
  int i;
  if (rank == -1) {
    for (i = 0; i < n; i++) {
      t[i] = 0;
    }
    rank = 0;
    return;
  }
  check = subset_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_LEX_SUCCESSOR(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  for (i = n - 1; 0 <= i; i--) {
    if (t[i] == 0) {
      t[i] = 1;
      rank = rank + 1;
      return;
    } else {
      t[i] = 0;
    }
  }
  rank = 0;

  return;
}

int* subset_lex_unrank(int rank, int n) {
  int i;
  int nsub;
  int rank_copy;
  int* t;
  if (n < 1) {
    cerr << "\n";
    cerr << "SUBSET_LEX_UNRANK(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  nsub = subset_enum(n);

  if (rank < 0 || nsub < rank) {
    cerr << "\n";
    cerr << "SUBSET_LEX_UNRANK(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }

  rank_copy = rank;
  t = new int[n];

  for (i = n - 1; 0 <= i; i--) {
    if ((rank_copy % 2) == 1) {
      t[i] = 1;
    } else {
      t[i] = 0;
    }
    rank_copy = rank_copy / 2;
  }

  return t;
}

int* subset_random(int n, int& seed) {
  int* s;

  s = i4vec_uniform_ab_new(n, 0, 1, seed);

  return s;
}

int* subset_union(int n, int a[], int b[]) {
  int* c;
  bool check;
  int i;
  check = subset_check(n, a);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_UNION(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  check = subset_check(n, b);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_UNION(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  c = new int[n];

  for (i = 0; i < n; i++) {
    c[i] = i4_max(a[i], b[i]);
  }

  return c;
}

int subset_weight(int n, int t[]) {
  bool check;
  int weight;
  check = subset_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_WEIGHT(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  weight = i4vec_sum(n, t);

  return weight;
}

int* subset_xor(int n, int a[], int b[]) {
  int* c;
  bool check;
  int i;
  check = subset_check(n, a);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_XOR(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  check = subset_check(n, b);

  if (!check) {
    cerr << "\n";
    cerr << "SUBSET_XOR(): Fatal error!\n";
    cerr << "  The subset is illegal.\n";
    exit(1);
  }

  c = new int[n];

  for (i = 0; i < n; i++) {
    c[i] = i4_max(a[i], b[i]) - i4_min(a[i], b[i]);
  }

  return c;
}

int subsetsum_swap(int n, int a[], int sum_desired, int index[]) {
  int i;
  int j;
  int nmove;
  int sum_achieved;
  sum_achieved = 0;

  for (i = 0; i < n; i++) {
    index[i] = 0;
  }
  i4vec_sort_insert_d(n, a);

  for (;;) {
    nmove = 0;

    for (i = 0; i < n; i++) {
      if (index[i] == 0) {
        if (sum_achieved + a[i] <= sum_desired) {
          index[i] = 1;
          sum_achieved = sum_achieved + a[i];
          nmove = nmove + 1;
          continue;
        }
      }

      if (index[i] == 0) {
        for (j = 0; j < n; j++) {
          if (index[j] == 1) {
            if (sum_achieved < sum_achieved + a[i] - a[j] && sum_achieved + a[i] - a[j] <= sum_desired) {
              index[j] = 0;
              index[i] = 1;
              nmove = nmove + 2;
              sum_achieved = sum_achieved + a[i] - a[j];
              break;
            }
          }
        }
      }
    }
    if (nmove <= 0) {
      break;
    }
  }

  return sum_achieved;
}

bool tableau_check(int n, int tab[]) {
  bool check;
  int i;
  int j;

  check = true;

  if (n < 1) {
    check = false;
    return check;
  }
  for (i = 0; i < 2; i++) {
    for (j = 0; j < n; j++) {
      if (tab[i + j * 2] < 1 || 2 * n < tab[i + j * 2]) {
        check = false;
        return check;
      }
    }
  }
  for (i = 0; i < 2; i++) {
    for (j = 1; j < n; j++) {
      if (tab[i + j * 2] <= tab[i + (j - 1) * 2]) {
        check = false;
        return check;
      }
    }
  }
  i = 1;
  for (j = 0; j < n; j++) {
    if (tab[i + j * 2] <= tab[i - 1 + j * 2]) {
      check = false;
      return check;
    }
  }
  return check;
}

int tableau_enum(int n) {
  int value;

  value = i4_choose(2 * n, n) / (n + 1);

  return value;
}

int* tableau_to_bal_seq(int n, int tab[]) {
  bool check;
  int i;
  int j;
  int* t;
  check = tableau_check(n, tab);

  if (!check) {
    cerr << "\n";
    cerr << "TABLEAU_TO_BAL_SEQ(): Fatal error!\n";
    cerr << "  The tableau is illegal.\n";
    exit(1);
  }

  t = new int[2 * n];

  for (i = 0; i < 2; i++) {
    for (j = 0; j < n; j++) {
      t[tab[i + j * 2] - 1] = i;
    }
  }

  return t;
}

bool tree_check(int n, int t[]) {
  bool check;
  int* d;
  int i;
  int j;
  int k;
  int x;
  int y;

  check = true;

  if (n < 1) {
    check = false;
    return check;
  }

  for (i = 0; i < 2; i++) {
    for (j = 0; j < n - 1; j++) {
      if (t[i + j * 2] < 1 || n < t[i + j * 2]) {
        check = false;
        return check;
      }
    }
  }
  d = edge_degree(n, n - 1, t);
  for (k = 1; k <= n - 1; k++) {
    x = 1;

    while (d[x - 1] != 1) {
      x = x + 1;
      if (n < x) {
        check = false;
        return check;
      }
    }
    j = 1;

    for (;;) {
      if (t[0 + (j - 1) * 2] == x) {
        y = t[1 + (j - 1) * 2];
        break;
      }

      if (t[1 + (j - 1) * 2] == x) {
        y = t[0 + (j - 1) * 2];
        break;
      }

      j = j + 1;

      if (n - 1 < j) {
        check = false;
        return check;
      }
    }
    t[0 + (j - 1) * 2] = -t[0 + (j - 1) * 2];
    t[1 + (j - 1) * 2] = -t[1 + (j - 1) * 2];

    d[x - 1] = d[x - 1] - 1;
    d[y - 1] = d[y - 1] - 1;
  }

  for (j = 0; j < n - 1; j++) {
    for (i = 0; i < 2; i++) {
      t[i + j * 2] = -t[i + j * 2];
    }
  }

  delete[] d;

  return check;
}

int tree_enum(int n) {
  int value;

  if (n < 1) {
    value = 0;
  } else if (n == 1) {
    value = 1;
  } else if (n == 2) {
    value = 1;
  } else {
    value = i4_power(n, n - 2);
  }
  return value;
}

int* tree_random(int n) {
  int* p;
  int rank;
  int* t;
  int tree_num;
  if (n < 1) {
    cerr << "\n";
    cerr << "tree_random(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }
  tree_num = tree_enum(n);
  rank = 1 + (rand() % tree_num);
  p = pruefer_unrank(rank, n);
  t = pruefer_to_tree_new(n, p);

  delete[] p;

  return t;
}

int tree_rank(int n, int t[]) {
  bool check;
  int* p;
  int rank;
  check = tree_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "TREE_RANK(): Fatal error!\n";
    cerr << "  The tree is illegal.\n";
    exit(1);
  }
  p = tree_to_pruefer(n, t);
  rank = pruefer_rank(n, p);

  delete[] p;

  return rank;
}

void tree_successor(int n, int t[], int& rank) {
  bool check;
  int i;
  int* p;
  if (rank == -1) {
    p = new int[n - 2];

    for (i = 0; i < n - 2; i++) {
      p[i] = 1;
    }
    pruefer_to_tree(n, p, t);
    rank = 0;
    delete[] p;
    return;
  }
  check = tree_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "TREE_SUCCESSOR(): Fatal error!\n";
    cerr << "  The tree is illegal.\n";
    exit(1);
  }
  p = tree_to_pruefer(n, t);
  pruefer_successor(n, p, rank);
  pruefer_to_tree(n, p, t);

  delete[] p;

  return;
}

int* tree_to_pruefer(int n, int t[]) {
  bool check;
  int* d;
  int i;
  int j;
  int k;
  int* p;
  int x;
  int y;
  check = tree_check(n, t);

  if (!check) {
    cerr << "\n";
    cerr << "TREE_TO_PRUEFER(): Fatal error!\n";
    cerr << "  The tree is illegal.\n";
    exit(1);
  }
  d = edge_degree(n, n - 1, t);

  p = new int[n - 2];

  for (j = 1; j <= n - 2; j++) {
    x = n;
    while (d[x - 1] != 1) {
      x = x - 1;
    }
    k = 1;

    for (;;) {
      if (t[0 + (k - 1) * 2] == x) {
        y = t[1 + (k - 1) * 2];
        break;
      }

      if (t[1 + (k - 1) * 2] == x) {
        y = t[0 + (k - 1) * 2];
        break;
      }
      k = k + 1;
    }
    p[j - 1] = y;
    d[x - 1] = d[x - 1] - 1;
    d[y - 1] = d[y - 1] - 1;

    t[0 + (k - 1) * 2] = -t[0 + (k - 1) * 2];
    t[1 + (k - 1) * 2] = -t[1 + (k - 1) * 2];
  }
  for (j = 0; j < n - 2; j++) {
    for (i = 0; i < 2; i++) {
      t[i + j * 2] = -t[i + j * 2];
    }
  }

  delete[] d;

  return p;
}

int* tree_unrank(int rank, int n) {
  int* p;
  int* t;
  int tree_num;
  if (n < 1) {
    cerr << "\n";
    cerr << "tree_unrank(): Fatal error!\n";
    cerr << "  Input N is illegal.\n";
    exit(1);
  }

  tree_num = tree_enum(n);

  if (rank < 0 || tree_num < rank) {
    cerr << "\n";
    cerr << "tree_unrank(): Fatal error!\n";
    cerr << "  The input rank is illegal.\n";
    exit(1);
  }
  p = pruefer_unrank(rank, n);
  t = pruefer_to_tree_new(n, p);

  delete[] p;

  return t;
}