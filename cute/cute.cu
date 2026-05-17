#include <cstdio>

#include <cute/layout.hpp>

using namespace cute;

namespace cute {

struct TikzColor_BWx8 {
  CUTE_HOST_DEVICE char const* operator()(int idx) const {
    static char const* color_map[8] = {
        "black!00", "black!40", "black!20", "black!60",
        "black!10", "black!50", "black!30", "black!70"};
    return color_map[idx % 8];
  }
};

template <class LayoutA, class TikzColorFn = TikzColor_BWx8>
CUTE_HOST_DEVICE
void print_latex(LayoutA const& layout_a, TikzColorFn color = {}) {
  CUTE_STATIC_ASSERT_V(rank(layout_a) <= Int<2>{});
  auto layout = append<2>(layout_a, Layout<_1, _0>{});

  printf("%% Layout: ");
  print(layout);
  printf("\n");
  printf("\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  auto [M, N] = product_each(shape(layout));
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int idx = layout(m, n);
      printf("\\node[fill=%s] at (%d,%d) {%d};\n", color(idx), m, n, idx);
    }
  }

  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n", int(M), int(N));
  for (int m = 0, n = -1; m < M; ++m) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, n, m);
  }
  for (int m = -1, n = 0; n < N; ++n) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, n, n);
  }

  printf("\\end{tikzpicture}\n"
         "\\end{document}\n");
}

} // namespace cute

template <class LayoutT>
void print_layout(char const* label, LayoutT const& layout) {
  printf("%s = ", label);
  print(layout);
  printf("\n");
}

template <class LayoutT>
void print_latex_block(char const* label, LayoutT const& layout) {
  printf("\n%% ===== %s =====\n", label);
  print_latex(layout);
  printf("\n");
}

int main() {
  // Basic layouts.
  auto row_major_2x3 = make_layout(
      make_shape(Int<2>{}, Int<3>{}),
      make_stride(Int<3>{}, Int<1>{}));

  auto col_major_2x3 = make_layout(
      make_shape(Int<2>{}, Int<3>{}),
      make_stride(Int<1>{}, Int<2>{}));

  auto identity_2x3 = make_identity_layout(make_shape(Int<2>{}, Int<3>{}));

  auto with_ones = make_layout(
      make_shape(Int<2>{}, Int<1>{}, Int<3>{}),
      make_stride(Int<3>{}, Int<0>{}, Int<1>{}));

  // Layout algebra.
  auto coalesced = coalesce(with_ones);
  auto filtered = filter(with_ones);
  auto composed = composition(row_major_2x3, identity_2x3);

  auto right_inv = right_inverse(row_major_2x3);
  auto left_inv = left_inverse(row_major_2x3);
  auto common_layout = max_common_layout(col_major_2x3, row_major_2x3);
  auto common_vector = max_common_vector(col_major_2x3, row_major_2x3);

  // Logical divide / product.
  auto line_12 = make_layout(Int<12>{});
  auto tile_3 = make_layout(Int<3>{});
  auto divided = logical_divide(line_12, tile_3);
  auto zipped_divided = zipped_divide(line_12, tile_3);
  auto product = logical_product(make_layout(Int<4>{}), tile_3);
  auto tiled_to_4x6 = tile_to_shape(row_major_2x3, make_shape(Int<4>{}, Int<6>{}));

  // Print results.
  print_layout("row_major_2x3", row_major_2x3);
  print_layout("col_major_2x3", col_major_2x3);
  print_layout("identity_2x3", identity_2x3);
  print_layout("with_ones", with_ones);
  print_layout("coalesced(with_ones)", coalesced);
  print_layout("filtered(with_ones)", filtered);
  print_layout("composition(row_major_2x3, identity_2x3)", composed);
  print_layout("right_inverse(row_major_2x3)", right_inv);
  print_layout("left_inverse(row_major_2x3)", left_inv);
  print_layout("max_common_layout(col_major_2x3, row_major_2x3)", common_layout);
  print_layout("logical_divide(line_12, tile_3)", divided);
  print_layout("zipped_divide(line_12, tile_3)", zipped_divided);
  print_layout("logical_product(line_4, tile_3)", product);
  print_layout("tile_to_shape(row_major_2x3, 4x6)", tiled_to_4x6);

  printf("row_major_2x3(1,2) = %d\n", row_major_2x3(1, 2));
  printf("col_major_2x3(1,2) = %d\n", col_major_2x3(1, 2));
  printf("max_common_vector(col_major_2x3, row_major_2x3) = ");
  print(common_vector);
  printf("\n");

  // LaTeX output: redirect stdout to a .tex file if you want to render it.
  print_latex_block("print_latex(row_major_2x3)", row_major_2x3);
  print_latex_block("print_latex(tile_to_shape(row_major_2x3, 4x6))", tiled_to_4x6);

  return 0;
}
