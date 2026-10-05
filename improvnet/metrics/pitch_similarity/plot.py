"""Dependency-free SVG diagnostic previews, never used for metric computation."""

import numpy as np


def save_matrix_plot(path, cross, reference_ssm, generated_ssm, alignment):
    size, margin, gap = 300, 50, 40
    width, height = 3 * (size + gap) + margin, size + 110
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#222">']
    for panel, (matrix, title) in enumerate(((cross, "Cross-similarity + alignment"), (reference_ssm, "Reference SSM"), (generated_ssm, "Comparison SSM"))):
        rows, cols = matrix.shape
        x, y = margin + panel * (size + gap), 45
        parts.append(f'<text x="{x}" y="20">{title} ({rows} × {cols})</text>')
        if matrix.size:
            # Preview-only block means bound SVG size. Scoring and NPZ retain
            # the original exact matrices; neither is resized or interpolated.
            row_edges = np.linspace(0, rows, min(rows, 80) + 1, dtype=int)
            col_edges = np.linspace(0, cols, min(cols, 80) + 1, dtype=int)
            for i, (low_r, high_r) in enumerate(zip(row_edges, row_edges[1:])):
                for j, (low_c, high_c) in enumerate(zip(col_edges, col_edges[1:])):
                    value = float(matrix[low_r:high_r, low_c:high_c].mean())
                    color = tuple(round(a * (1-value) + b * value) for a,b in zip((245,248,255),(20,60,150)))
                    xx, yy = x + size * low_c / cols, y + size * low_r / rows
                    w, h = size * (high_c-low_c) / cols, size * (high_r-low_r) / rows
                    parts.append(f'<rect x="{xx:.3f}" y="{yy:.3f}" width="{w+.05:.3f}" height="{h+.05:.3f}" fill="rgb{color}"/>')
            if panel == 0 and alignment:
                points = " ".join(f'{x+size*(j+.5)/cols:.3f},{y+size*(i+.5)/rows:.3f}' for i,j in alignment)
                parts.append(f'<polyline points="{points}" fill="none" stroke="#d62728" stroke-width="1.8"/>')
        else:
            parts.append(f'<text x="{x}" y="{y+size/2}">No pitched notes</text>')
        parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="none" stroke="#777"/>')
        parts.append(f'<text x="{x}" y="{y+size+20}">Note indices: 0 → {max(0,cols-1)} (row 0 at top)</text>')
    parts.append(f'<text x="{margin}" y="{height-12}">White = 0, dark blue = 1; red = retained alignment. Large matrices are block-averaged for preview only.</text>')
    parts.append('</g></svg>')
    with path.open("x") as handle:
        handle.write("\n".join(parts))
