# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.26.0",
#     "numpy>=2.0",
#     "plotly>=6.0",
# ]
# ///
#
# Finding the zeros of the Riemann zeta function: an interactive slide deck.
# Part 1 is the math (zeta = eta / (1 - 2^(1-s)), Euler's formula); Part 2
# animates eta's arrows spiralling into 0 at the first zeros; Part 3 is
# Riemann's own computation (the real Z(t), mirror pairs into cosines, Newton's
# method), then the intermediate value theorem and bisection.
#
# Present it as slides (arrow keys move; the charts have their own Play buttons):
#
#     uvx marimo run --sandbox finding_zeta_zeros.py
#
# Or open it as a notebook (code is hidden; "Show code" reveals it):
#
#     uvx marimo edit --sandbox finding_zeta_zeros.py
#
# It also runs in the browser with nothing installed (molab's WebAssembly mode);
# see README.md. The slide layout is stored inside this file (the layout_file
# data URI below), so the deck opens as slides wherever the file goes; after
# adding or removing cells, rerun make_slides_layout.py (cells named frag_*
# appear one step at a time).
#
# Sources for the history: C. L. Siegel, "Uber Riemanns Nachlass zur analytischen
# Zahlentheorie" (1932), English translation arXiv:1810.05198; G. Hiary,
# D. Ireland, M. Kyi, arXiv:2408.00187 (2024); D. Platt, T. Trudgian, Bull. LMS 53
# (2021), arXiv:2004.09765.

import marimo

__generated_with = "0.26.0"
app = marimo.App(
    width="medium",
    app_title="Finding the zeros of zeta",
    layout_file=(
        "data:application/json;base64,"
        "eyJ0eXBlIjoic2xpZGVzIiwiZGF0YSI6eyJjZWxscyI6W3t9LHt9LHt9LHt9LHt9"
        "LHt9LHsidHlwZSI6ImZyYWdtZW50In0seyJ0eXBlIjoiZnJhZ21lbnQifSx7InR5"
        "cGUiOiJmcmFnbWVudCJ9LHt9LHsidHlwZSI6ImZyYWdtZW50In0se30seyJ0eXBl"
        "IjoiZnJhZ21lbnQifSx7InR5cGUiOiJmcmFnbWVudCJ9LHt9LHsidHlwZSI6ImZy"
        "YWdtZW50In0se30se30se30se30se30se30se30se30se30se30seyJ0eXBlIjoi"
        "ZnJhZ21lbnQifSx7InR5cGUiOiJmcmFnbWVudCJ9LHsidHlwZSI6ImZyYWdtZW50"
        "In0se30se30seyJ0eXBlIjoiZnJhZ21lbnQifSx7InR5cGUiOiJmcmFnbWVudCJ9"
        "LHt9LHsidHlwZSI6ImZyYWdtZW50In0seyJ0eXBlIjoiZnJhZ21lbnQifSx7fSx7"
        "fSx7fSx7fSx7InR5cGUiOiJmcmFnbWVudCJ9LHt9LHsidHlwZSI6ImZyYWdtZW50"
        "In0se30se30se30seyJ0eXBlIjoiZnJhZ21lbnQifSx7InR5cGUiOiJmcmFnbWVu"
        "dCJ9LHt9LHt9LHt9LHsidHlwZSI6ImZyYWdtZW50In0seyJ0eXBlIjoiZnJhZ21l"
        "bnQifSx7InR5cGUiOiJmcmFnbWVudCJ9LHsidHlwZSI6ImZyYWdtZW50In1dLCJk"
        "ZWNrIjp7InRyYW5zaXRpb24iOiJzbGlkZSIsInZlcnRpY2FsQWxpZ24iOiJ0b3Ai"
        "fX19"
    ),
)


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:  # running in the browser (WebAssembly): fetch plotly
        import micropip

        await micropip.install("plotly")

    import functools
    import math

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return functools, go, make_subplots, math, mo, np


@app.cell(hide_code=True)
def _(math, np):
    # The core computations. Everything in the deck is built from these.

    TWO_PI = 2 * math.pi

    # The first 13 zeros on the critical line, s = 1/2 + it: every zero with t < 60.
    # (Computed with mpmath.zetazero; they agree with Odlyzko's published tables.)
    ALL_ZEROS = [14.134725141734694, 21.022039638771555, 25.010857580145689,
                 30.424876125859513, 32.93506158773919, 37.586178158825671,
                 40.918719012147495, 43.327073280915, 48.00515088116716,
                 49.773832477672302, 52.970321477714461, 56.446247697063395,
                 59.347044002602353]
    ZEROS = ALL_ZEROS[:10]
    GAMMA1 = ZEROS[0]  # 14.134725141734693...
    NOT_A_ZERO = 16.0


    def eta_terms(t, count):
        """The terms (-1)^(n-1) / n^s of eta, n = 1, ..., count, with s = 1/2 + it.

        1/n^s = (cos(t ln n) - i sin(t ln n)) / sqrt(n), by Euler's formula."""
        n = np.arange(1, count + 1)
        signs = np.where(n % 2 == 1, 1.0, -1.0)
        angle = t * np.log(n)
        return signs * (np.cos(angle) - 1j * np.sin(angle)) / np.sqrt(n)


    def eta_sums(t, count):
        """Running totals S_0 = 0, S_1, ..., S_count of eta's series (entry k is S_k)."""
        return np.concatenate([[0j], np.cumsum(eta_terms(t, count))])


    def zeta_sums(t, count):
        """Running totals of zeta's own series 1 + 1/2^s + 1/3^s + ... (every sign +)."""
        n = np.arange(1, count + 1)
        return np.concatenate([[0j], np.cumsum(np.exp(-1j * t * np.log(n)) / np.sqrt(n))])


    def eta_fast(t, start=60, rounds=60):
        """eta(1/2 + it) for one t or an array of t, accurately: eta's totals zigzag
        around the answer, so average neighbors of S_start, ..., S_(start + rounds)
        again and again down to one number (the Euler transform; Hasse 1930)."""
        ts = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
        n = np.arange(1, start + rounds + 1)
        signs = np.where(n % 2 == 1, 1.0, -1.0)
        totals = np.cumsum(signs * np.exp(-1j * ts * np.log(n)) / np.sqrt(n), axis=1)
        row = totals[:, start - 1:]  # S_start, ..., S_(start + rounds)
        for _ in range(rounds):
            row = (row[:, :-1] + row[:, 1:]) / 2
        return row[:, 0] if np.ndim(t) else complex(row[0, 0])


    def zeta_on_line(t):
        """zeta(1/2 + it) = eta(s) / (1 - 2^(1-s)), for one t or an array of t."""
        s = 0.5 + 1j * np.asarray(t, dtype=float)
        return eta_fast(t) / (1 - 2 ** (1 - s))


    def theta(t):
        """Riemann's angle (from Stirling's formula): turning zeta by theta makes it real."""
        return (t / 2) * np.log(t / TWO_PI) - t / 2 - np.pi / 8 + 1 / (48 * t)


    def theta_slope(t):
        """theta'(t), term by term: (1/2) ln(t/2pi) - 1/(48 t^2)."""
        return 0.5 * np.log(t / TWO_PI) - 1 / (48 * t * t)


    def z_from_eta(t):
        """Hardy's Z(t) = e^(i theta) zeta(1/2 + it), from eta's sums: a real number.
        Works for one t or an array of t; agrees with mpmath.siegelz to 1e-12 on [10, 60]."""
        return (np.exp(1j * theta(np.asarray(t, dtype=float))) * zeta_on_line(t)).real


    def crossing(f, lo, hi, halvings=50):
        """Bisection: f changes sign on [lo, hi]; halve the interval, keep the sign change."""
        f_lo = f(lo)
        for _ in range(halvings):
            mid = (lo + hi) / 2
            if (f(mid) > 0) == (f_lo > 0):
                lo, f_lo = mid, f(mid)
            else:
                hi = mid
        return (lo + hi) / 2
    return (
        ALL_ZEROS,
        GAMMA1,
        NOT_A_ZERO,
        TWO_PI,
        ZEROS,
        crossing,
        eta_fast,
        eta_sums,
        eta_terms,
        theta,
        theta_slope,
        z_from_eta,
        zeta_on_line,
        zeta_sums,
    )


@app.cell(hide_code=True)
def _(go, math, mo, np):
    # Chart styling (light or dark, following marimo's theme) and drawing helpers.

    _dark = mo.app_meta().theme == "dark"
    C = {
        "surface": "#1a1a19" if _dark else "#fcfcfb",
        "ink": "#f0efec" if _dark else "#0b0b0b",
        "ink2": "#c3c2b7" if _dark else "#52514e",
        "muted": "#898781",
        "grid": "#2c2c2a" if _dark else "#e1e0d9",
        "axis": "#383835" if _dark else "#c3c2b7",
        "blue": "#3987e5" if _dark else "#2a78d6",
        "orange": "#d95926" if _dark else "#eb6834",
        "aqua": "#199e70" if _dark else "#1baf7a",
    }
    FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


    def style(fig, height=460, legend="top", top=60, bottom=45):
        """Common look. ``top`` and ``bottom`` are margins in pixels: room for the
        caption (the figure title) and legend above, and for the player below."""
        legends = {
            "top": dict(orientation="h", x=0, y=1.0, xanchor="left", yanchor="bottom"),
            "right": dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"),
        }
        fig.update_layout(
            height=height,
            margin=dict(l=60, r=30, t=top, b=bottom),
            paper_bgcolor=C["surface"],
            plot_bgcolor=C["surface"],
            font=dict(family=FONT, size=14, color=C["ink2"]),
            title=dict(font=dict(size=15, color=C["ink"]), x=0.01, xanchor="left",
                       y=1, yanchor="top", pad=dict(t=12)),
            legend=legends[legend] | dict(font=dict(color=C["ink2"])),
            hoverlabel=dict(font=dict(family=FONT)),
        )
        fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["axis"], linecolor=C["axis"],
                         tickfont=dict(color=C["muted"]))
        fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["axis"], linecolor=C["axis"],
                         tickfont=dict(color=C["muted"]))
        fig.update_annotations(font=dict(color=C["ink2"], size=13))
        return fig


    def plane_window(fig, center, radius, row=None, col=None, xtitle=True):
        """Square window on the complex plane, equal scale on both axes."""
        k = "" if col in (None, 1) else str(col)
        fig.update_xaxes(range=[center.real - radius, center.real + radius],
                         constrain="domain", title_text="real part" if xtitle else None,
                         row=row, col=col)
        fig.update_yaxes(range=[center.imag - radius, center.imag + radius],
                         scaleanchor=f"x{k}", scaleratio=1, constrain="domain",
                         title_text="imaginary part", row=row, col=col)


    def _heads(pts, px_per_unit, max_head):
        """Arrowhead sizes: proportional to each segment's on-screen length, capped."""
        seg_px = np.abs(np.diff(pts)) * px_per_unit
        return np.concatenate([[0.0], np.clip(0.45 * seg_px, 0.0, max_head)])


    def arrows(points, color, width, px_per_unit, name, showlegend=True, hover=None,
               dash=None, max_head=12, xaxis=None, yaxis=None):
        """A path through the points, with an arrowhead at the end of every segment."""
        pts = np.asarray(points, dtype=complex)
        size = _heads(pts, px_per_unit, max_head)
        extra = {}
        if xaxis:
            extra = dict(xaxis=xaxis, yaxis=yaxis)
        return go.Scatter(
            x=pts.real, y=pts.imag, mode="lines+markers", name=name, showlegend=showlegend,
            line=dict(color=color, width=width, dash=dash),
            marker=dict(symbol="arrow", angleref="previous", size=size, standoff=size / 2,
                        color=color, line=dict(width=0)),
            customdata=None if hover is None else hover[0],
            hovertemplate="<extra></extra>" if hover is None else hover[1],
            **extra,
        )


    def arrow_frame(points, px_per_unit, max_head=12, color=None):
        """The changing parts of an arrows() trace, for an animation frame. Frames carry
        only what changes; plotly keeps the rest of the trace's styling."""
        pts = np.asarray(points, dtype=complex)
        size = _heads(pts, px_per_unit, max_head)
        update = dict(x=pts.real, y=pts.imag, marker=dict(size=size, standoff=size / 2))
        if color:
            update["line"] = dict(color=color)
            update["marker"]["color"] = color
        return update


    def dot(z, color, name, symbol="circle", size=10, showlegend=True, **extra):
        return go.Scatter(
            x=[z.real], y=[z.imag], mode="markers", name=name, showlegend=showlegend,
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=2, color=color)),
            hovertemplate=f"{name}<br>%{{x:.6f}} + (%{{y:.6f}}) i<extra></extra>", **extra,
        )


    def fmt_c(z, places=4):
        """a + bi, as plain text with a real minus sign."""
        re_part = f"{z.real:.{places}f}".replace("-", "−")
        sign = "+" if z.imag >= 0 else "−"
        return f"{re_part} {sign} {abs(z.imag):.{places}f}i"


    def fmt_r(x, places=4, plus=False):
        """A real number as plain text with a real minus sign."""
        return (f"{x:+.{places}f}" if plus else f"{x:.{places}f}").replace("-", "−")


    def tex_c(z, places=4):
        """a + bi, as TeX."""
        sign = "+" if z.imag >= 0 else "-"
        return rf"{z.real:.{places}f} {sign} {abs(z.imag):.{places}f}\,i"


    def add_player(fig, duration, step_names=(), step_labels=(), prefix="", active=0,
                   from_start=False, quiet_ticks=False):
        """Play / Pause buttons (and a frame slider) in the bottom margin. Call after
        style(), which sets the margins this positions against."""
        plot_px = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b

        def below(px):
            return -px / plot_px

        def _run(from_current):
            return dict(frame=dict(duration=duration, redraw=True), transition=dict(duration=0),
                        fromcurrent=from_current, mode="immediate")

        buttons = [dict(label="▶ Play", method="animate", args=[None, _run(not from_start)]),
                   dict(label="❚❚ Pause", method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           transition=dict(duration=0), mode="immediate")])]
        if not from_start:
            buttons.append(dict(label="↺ Start over", method="animate", args=[None, _run(False)]))
        menus = [dict(type="buttons", direction="left", showactive=False, x=0, y=below(34),
                      xanchor="left", yanchor="top", pad=dict(t=0, r=10),
                      bgcolor=C["surface"], bordercolor=C["axis"],
                      font=dict(color=C["ink"], size=13), buttons=buttons)]
        sliders = []
        if step_names:
            sliders = [dict(
                active=active, x=0.42, len=0.5, y=below(24), yanchor="top", pad=dict(t=0, b=0),
                currentvalue=dict(prefix=prefix, visible=True, xanchor="left",
                                  font=dict(size=14, color=C["ink"])),
                font=dict(color="rgba(0,0,0,0)" if quiet_ticks else C["muted"], size=10),
                bordercolor=C["axis"], tickcolor=C["axis"],
                steps=[dict(method="animate", label=lab,
                            args=[[name], dict(mode="immediate", transition=dict(duration=0),
                                               frame=dict(duration=0, redraw=True))])
                       for name, lab in zip(step_names, step_labels)])]
        fig.update_layout(updatemenus=menus, sliders=sliders)
        return fig
    return (C, add_player, arrow_frame, arrows, dot, fmt_c, fmt_r, plane_window, style,
            tex_c)


@app.cell(hide_code=True)
def title_slide(mo):
    mo.md(r"""
    # Finding the zeros of the zeta function

    The Riemann hypothesis says every non-trivial zero of the zeta function lies on
    the line $s = \tfrac12 + it$. The first one is at

    $$t = 14.134725\ldots$$

    Let's find it ourselves, with nothing fancier than sines, cosines and a lot of
    adding.

    1. **The math:** rewrite zeta so it works on that line, and turn every term into an arrow.
    2. **Watching it happen:** add the arrows and see them spiral into $0$.
    3. **How Riemann did it,** by hand, in the 1850s, and how zeros are pinned down today.
    """)
    return


@app.cell(hide_code=True)
def part1(mo):
    mo.md(r"""
    # Part 1 · The math

    Rewrite zeta so it works on the line $s = \tfrac12 + it$, then turn every term
    into an arrow.
    """)
    return


@app.cell(hide_code=True)
def two_series(mo):
    mo.md(r"""
    ## Zeta and its cousin, eta

    Eta is zeta with every other sign flipped:

    $$\zeta(s) = \frac{1}{1^s} + \frac{1}{2^s} + \frac{1}{3^s} + \frac{1}{4^s} + \cdots \qquad\qquad \eta(s) = \frac{1}{1^s} - \frac{1}{2^s} + \frac{1}{3^s} - \frac{1}{4^s} + \cdots$$
    """)
    return


@app.cell(hide_code=True)
def frag_subtract(mo):
    mo.md(r"""
    **Subtract them.** The odd terms cancel, and the even terms double:

    $$\zeta(s) - \eta(s) = \frac{2}{2^s} + \frac{2}{4^s} + \frac{2}{6^s} + \cdots$$
    """)
    return


@app.cell(hide_code=True)
def frag_factor(mo):
    mo.md(r"""
    **Pull out** $\frac{2}{2^s} = 2^{1-s}$, since $4^s = 2^s \cdot 2^s$, $6^s = 2^s \cdot 3^s$, and so on:

    $$\zeta(s) - \eta(s) = \frac{2}{2^s}\left(\frac{1}{1^s} + \frac{1}{2^s} + \frac{1}{3^s} + \cdots\right) = 2^{1-s}\,\zeta(s)$$
    """)
    return


@app.cell(hide_code=True)
def frag_result(mo):
    mo.md(r"""
    **Solve for zeta:** $\;\zeta(s)\left(1 - 2^{1-s}\right) = \eta(s)$, so

    $$\boxed{\;\zeta(s) = \frac{\eta(s)}{1 - 2^{1-s}}\;}$$
    """)
    return


@app.cell(hide_code=True)
def why_eta(mo):
    mo.md(r"""
    ## Why bother with eta?

    On the line $s = \tfrac12 + it$, zeta's own series **never settles down**: its
    running totals drift further and further from any answer. Eta's series does
    settle. (We'll watch both in Part 2.)

    <small>*The fine print: subtracting two endless sums is only safe where both add
    up to a number, which needs the real part of $s$ above $1$. Eta's series works
    for any real part above $0$, and Riemann's extension of zeta agrees with the
    formula there, so on our line this formula is how zeta is computed.*</small>
    """)
    return


@app.cell(hide_code=True)
def frag_same_zeros(mo):
    mo.md(r"""
    **Same zeros.** On the line, $2^{1-s} = 2^{1/2 - it}$ always has size
    $\sqrt 2 \approx 1.414$, so the bottom $1 - 2^{1-s}$ is never $0$. So

    $$\zeta\!\left(\tfrac12 + it\right) = 0 \quad\Longleftrightarrow\quad \eta\!\left(\tfrac12 + it\right) = 0.$$

    Hunting for zeros of zeta is the same as hunting for zeros of eta.
    """)
    return


@app.cell(hide_code=True)
def arrows_math(mo):
    mo.md(r"""
    ## Every term is an arrow

    Write $s = \tfrac12 + it$ and split the power in two. Since $n = e^{\ln n}$,
    we have $n^{it} = e^{it\ln n}$, so

    $$\frac{1}{n^s} = \frac{1}{n^{1/2}} \cdot \frac{1}{n^{it}} = \frac{1}{\sqrt n}\, e^{-it\ln n}.$$
    """)
    return


@app.cell(hide_code=True)
def frag_euler(mo):
    mo.md(r"""
    **Euler's formula,** $e^{ix} = \cos x + i\sin x$, with $x = -t\ln n$. Cosine is
    even and sine is odd, so the minus sign moves outside the sine:

    $$e^{-it\ln n} = \cos(-t\ln n) + i\sin(-t\ln n) = \cos(t\ln n) - i\sin(t\ln n)$$
    """)
    return


@app.cell(hide_code=True)
def frag_arrow_box(mo):
    mo.md(r"""
    $$\boxed{\;\frac{1}{n^{1/2 + it}} = \frac{1}{\sqrt n}\Big(\cos(t\ln n) - i\,\sin(t\ln n)\Big)\;}$$

    An **arrow of length $1/\sqrt n$, turned clockwise by $t\ln n$ radians.**
    """)
    return


@app.cell(hide_code=True)
def zero_needs(mo):
    mo.md(r"""
    ## What a zero needs

    Eta's $n$-th term is that arrow with the sign $(-1)^{n-1}$ in front, which flips
    every second arrow around:

    $$\eta\!\left(\tfrac12 + it\right) = \sum_{n \ge 1} \frac{(-1)^{n-1}}{\sqrt n}\Big(\cos(t\ln n) - i\,\sin(t\ln n)\Big)$$
    """)
    return


@app.cell(hide_code=True)
def frag_two_sums(mo):
    mo.md(r"""
    So a zero needs **two** real sums to vanish at the same $t$:

    $$\sum_{n \ge 1} \frac{(-1)^{n-1}\cos(t\ln n)}{\sqrt n} = 0 \qquad\text{and}\qquad \sum_{n \ge 1} \frac{(-1)^{n-1}\sin(t\ln n)}{\sqrt n} = 0$$
    """)
    return


@app.cell(hide_code=True)
def part2(mo):
    mo.md(r"""
    # Part 2 · Watching it happen

    Turn one term into an arrow, then add the arrows one at a time and see where
    they go.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    one_n = mo.ui.slider(1, 30, value=2, label="term $n$", show_value=True, full_width=True)
    return (one_n,)


@app.cell(hide_code=True)
def one_term(C, GAMMA1, add_player, arrow_frame, arrows, dot, go, math, mo, np, one_n,
             plane_window, style, tex_c):
    def rotation_figure(t, n, turn_frames=40, settle_frames=10):
        """The arrow for term n turning clockwise by t ln n, shrinking, then flipping."""
        turn = t * math.log(n)
        length = 1 / math.sqrt(n)
        sign = 1 if n % 2 else -1
        ppu = 400 / 2.6
        states = []
        for k in range(turn_frames + 1):
            done = turn * k / turn_frames
            states.append((done, 1.0, f"turning clockwise: {done:.2f} of {turn:.2f} radians "
                                       f"({done / (2 * math.pi):.2f} turns)"))
        for k in range(1, settle_frames + 1):
            states.append((turn, 1 + (length - 1) * k / settle_frames,
                           f"shrinking to length 1/√{n} = {length:.4f}"))
        if sign < 0:
            for k in range(1, settle_frames + 1):
                states.append((turn, length * (1 - 2 * k / settle_frames),
                               f"flipping: eta's sign for n = {n} is −1"))
        final = sign * length * complex(math.cos(turn), -math.sin(turn))
        states[-1] = (turn, sign * length, f"eta's term {n} = {final.real:.4f} "
                                           f"{'+' if final.imag >= 0 else '−'} {abs(final.imag):.4f}i")

        def trail(done):
            angles = np.linspace(0, done, max(2, int(done * 12)))
            return (0.25 + 0.05 * angles / (2 * math.pi)) * np.exp(-1j * angles)

        def tip(done, scale):
            return scale * complex(math.cos(done), -math.sin(done))

        last_done, last_scale = states[-1][:2]
        circle = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
        fig = go.Figure(data=[
            go.Scatter(x=circle.real, y=circle.imag, mode="lines", name="radius 1",
                       line=dict(color=C["muted"], width=1, dash="dot"), hoverinfo="skip"),
            dot(0j, C["ink"], "0", symbol="circle-open", size=8, showlegend=False),
            go.Scatter(x=trail(last_done).real, y=trail(last_done).imag, mode="lines",
                       name="angle turned", line=dict(color=C["ink2"], width=1.5),
                       hoverinfo="skip"),
            arrows([0, tip(last_done, last_scale)], C["blue"] if last_scale > 0 else C["orange"],
                   4, ppu, "the arrow", max_head=16),
        ])
        fig.frames = [go.Frame(name=str(i), traces=[2, 3], layout=dict(title_text=label), data=[
            dict(x=trail(done).real, y=trail(done).imag),
            arrow_frame([0, tip(done, scale)], ppu, max_head=16,
                        color=C["blue"] if scale > 0 else C["orange"]),
        ]) for i, (done, scale, label) in enumerate(states)]
        fig.update_layout(title_text=states[-1][2])
        plane_window(fig, 0j, 1.3, xtitle=False)
        style(fig, height=495, legend="right", top=40, bottom=75)
        return add_player(fig, duration=60, from_start=True)


    def steps(t, n):
        turn = t * math.log(n)
        term = complex(math.cos(turn), -math.sin(turn)) / math.sqrt(n)
        sign = 1 if n % 2 else -1
        return mo.md(rf"""
    - turn: $t\ln {n} = {turn:.4f}$ rad, or ${turn / (2 * math.pi):.2f}$ turns
    - length: $1/\sqrt{{{n}}} = {1 / math.sqrt(n):.4f}$
    - $\dfrac{{1}}{{{n}^s}} = {tex_c(term)}$
    - eta's sign $(-1)^{{{n - 1}}} = {sign:+d}$, so eta's term is ${tex_c(sign * term)}$
    """)


    mo.vstack([
        mo.md(rf"## One term, animated &nbsp; <small>($t = {GAMMA1:.6f}\ldots$, the first zero)</small>"),
        mo.hstack([
            mo.vstack([
                mo.md(r"""
    Pick $n$, then press **▶ Play** under the picture. The arrow turns clockwise by
    $t\ln n$ (the spiral in the middle counts the full turns), shrinks to length
    $1/\sqrt n$, and flips if eta's sign is $-1$.
    """),
                one_n,
                steps(GAMMA1, one_n.value),
            ], gap=1),
            rotation_figure(GAMMA1, one_n.value),
        ], widths=[1, 1.7], align="start", gap=1.5),
    ])
    return


@app.cell(hide_code=True)
def _(NOT_A_ZERO, ZEROS, mo):
    _ordinal = ["1st", "2nd", "3rd"] + [f"{k}th" for k in range(4, 11)]
    T_CHOICES = {f"{o} zero: t = {z:.6f}...": z for o, z in zip(_ordinal, ZEROS)}
    T_CHOICES[f"not a zero: t = {NOT_A_ZERO:g}"] = NOT_A_ZERO
    pick_t = mo.ui.dropdown(T_CHOICES, value=list(T_CHOICES)[0], label="t")
    return T_CHOICES, pick_t


@app.cell(hide_code=True)
def first_30(C, add_player, arrow_frame, arrows, dot, eta_sums, eta_terms, fmt_c, go, math,
             mo, pick_t, plane_window, style):
    def first_arrows_figure(t, count=30, grow=4):
        """Eta's first arrows added tip to tail, one at a time (each grows into place)."""
        sums = eta_sums(t, count)
        terms = eta_terms(t, count)
        xs, ys = sums.real, sums.imag
        center = complex((xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2)
        radius = 0.55 * max(xs.max() - xs.min(), ys.max() - ys.min()) + 0.06
        ppu = 420 / (2 * radius)

        def tip(n, f):
            return sums[n - 1] + f * (sums[n] - sums[n - 1])

        def caption(n):
            turn = t * math.log(n)
            return (f"<b>term {n}</b> = (−1)<sup>{n - 1}</sup>/√{n} · (cos {turn:.2f} − i sin {turn:.2f})"
                    f" = {fmt_c(complex(terms[n - 1]))}<br><b>total S<sub>{n}</sub></b> = "
                    f"{fmt_c(complex(sums[n]))}, at distance {abs(sums[n]):.4f} from 0")

        fig = go.Figure(data=[
            arrows(sums[:1], C["muted"], 1.5, ppu, "earlier terms"),
            arrows([sums[0], sums[1]], C["orange"], 3.5, ppu, "the new term", max_head=15),
            arrows([0, sums[1]], C["blue"], 3, ppu, "the running total", max_head=15),
            dot(0j, C["ink"], "0", symbol="circle-open", size=12, showlegend=False),
        ])
        frames = []
        for n in range(1, count + 1):
            for g in range(1, grow + 1):  # the last growth step is the finished arrow
                f = g / grow
                frames.append(go.Frame(
                    name=str(n) if g == grow else f"{n}.{g}", traces=[0, 1, 2],
                    layout=dict(title_text=caption(n)),
                    data=[arrow_frame(sums[:n], ppu),
                          arrow_frame([sums[n - 1], tip(n, f)], ppu, max_head=15),
                          arrow_frame([0, tip(n, f)], ppu, max_head=15)]))
        fig.frames = frames
        fig.update_layout(title_text=caption(1))
        plane_window(fig, center, radius, xtitle=False)
        style(fig, height=540, legend="right", top=70, bottom=85)
        return add_player(fig, duration=110, step_names=[str(n) for n in range(1, count + 1)],
                          step_labels=[str(n) for n in range(1, count + 1)], prefix="terms added: ")


    mo.vstack([
        mo.md("## Adding the first 30 arrows"),
        mo.hstack([
            mo.vstack([
                mo.md(r"""
    Each arrow starts where the last one ended.

    **Orange:** the newest term.

    **Blue:** the running total, from $0$.

    Press **▶ Play** under the picture, or drag the slider. Try the other zeros too.
    """),
                pick_t,
            ], gap=1),
            first_arrows_figure(pick_t.value),
        ], widths=[1, 2.6], align="start", gap=1.5),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    compare_zeta = mo.ui.checkbox(label="also show zeta's own sum (every sign +)")
    return (compare_zeta,)


@app.cell(hide_code=True)
def keep_adding(C, add_player, arrow_frame, arrows, compare_zeta, dot, eta_fast, eta_sums, fmt_c,
                go, make_subplots, mo, np, pick_t, plane_window, style, zeta_sums):
    _N_MAX = 5000
    _IDX = np.unique(np.concatenate([np.arange(1, 101), np.geomspace(100, _N_MAX, 250).astype(int)]))


    def closing_in_figure(t, show_zeta):
        """The running total S_N for N = 30 ... 5000, riding a circle of radius 1/(2 sqrt N)."""
        sums = eta_sums(t, _N_MAX)
        target = eta_fast(t)
        is_zero = abs(target) < 1e-6
        steps = np.unique(np.geomspace(30, _N_MAX, 70).round().astype(int))
        radius = 1.35 * max(abs(sums[30] - target), 0.5 / np.sqrt(30))
        ppu = 300 / (2 * radius)
        ring = np.exp(1j * np.linspace(0, 2 * np.pi, 72))

        def circle(n):
            return target + 0.5 / np.sqrt(n) * ring

        def caption(n):
            s = complex(sums[n])
            if is_zero:
                return (f"<b>N = {n}</b>: S<sub>N</sub> = {fmt_c(s, 5)}, distance from 0: "
                        f"{abs(s):.5f}   (1/(2√N) = {0.5 / np.sqrt(n):.5f})")
            return (f"<b>N = {n}</b>: S<sub>N</sub> = {fmt_c(s, 5)}, closing in on "
                    f"η = {fmt_c(target, 3)}, not on 0")

        n0 = steps[0]
        fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.12)
        fig.add_trace(go.Scatter(x=circle(n0).real, y=circle(n0).imag, mode="lines",
                                 name="radius 1/(2√N)", hoverinfo="skip",
                                 line=dict(color=C["ink2"], width=1.2, dash="dot")), row=1, col=1)
        fig.add_trace(arrows([0, sums[n0]], C["blue"], 3, ppu, "running total S<sub>N</sub>",
                             max_head=14), row=1, col=1)
        fig.add_trace(go.Scatter(x=[n0], y=[abs(sums[n0])], mode="markers", showlegend=False,
                                 marker=dict(size=11, color=C["blue"],
                                             line=dict(width=2, color=C["surface"])),
                                 hoverinfo="skip"), row=1, col=2)
        if is_zero:
            fig.add_trace(dot(0j, C["ink"], "0, the target", symbol="cross-thin", size=14),
                          row=1, col=1)
        else:
            fig.add_trace(dot(target, C["orange"], "η, the target", symbol="x-thin", size=14),
                          row=1, col=1)
        fig.add_trace(go.Scatter(x=_IDX, y=0.5 / np.sqrt(_IDX), mode="lines", name="1/(2√N)",
                                 line=dict(color=C["ink2"], width=1.2, dash="dash"),
                                 hovertemplate="1/(2√N) = %{y:.5f}<extra></extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=_IDX, y=np.abs(sums[_IDX]), mode="lines", name="distance from 0",
                                 line=dict(color=C["blue"], width=2.5),
                                 hovertemplate="N = %{x}: %{y:.5f}<extra></extra>"), row=1, col=2)
        if show_zeta:
            fig.add_trace(go.Scatter(x=_IDX, y=np.abs(zeta_sums(t, _N_MAX)[_IDX]), mode="lines",
                                     name="zeta's own total", line=dict(color=C["orange"], width=2),
                                     hovertemplate="zeta, N = %{x}: %{y:.3f}<extra></extra>"),
                          row=1, col=2)
        fig.frames = [go.Frame(name=str(n), traces=[0, 1, 2], layout=dict(title_text=caption(n)),
                               data=[dict(x=circle(n).real, y=circle(n).imag),
                                     arrow_frame([0, sums[n]], ppu, max_head=14),
                                     dict(x=[n], y=[abs(sums[n])], xaxis="x2", yaxis="y2")])
                      for n in steps]
        fig.update_layout(title_text=caption(n0))
        plane_window(fig, target, radius, row=1, col=1, xtitle=False)
        fig.update_xaxes(type="log", dtick=1, range=[0, np.log10(_N_MAX)], row=1, col=2)
        fig.update_yaxes(type="log", dtick=1, range=[-2.5, 1.2 if show_zeta else 0.7],
                         title_text="distance from 0", row=1, col=2)
        style(fig, height=405, legend="top", top=62, bottom=85)
        return add_player(fig, duration=170, step_names=[str(n) for n in steps],
                          step_labels=[str(n) for n in steps], prefix="N = ", quiet_ticks=True)


    mo.vstack([
        mo.md(r"""
    ## Keep adding: closing in on $0$

    After a few dozen terms each new arrow nearly cancels the last, so the running
    total (blue) stays on a shrinking circle of radius about $1/(2\sqrt N)$. Press **▶ Play**.
    """),
        mo.hstack([pick_t, compare_zeta], justify="start", align="center", gap=3),
        closing_in_figure(pick_t.value, compare_zeta.value),
    ])
    return


@app.cell(hide_code=True)
def all_zeros(C, NOT_A_ZERO, ZEROS, eta_sums, functools, go, mo, np, pick_t, style):
    _N_MAX = 100_000
    _IDX = np.unique(np.concatenate([np.arange(1, 101), np.geomspace(100, _N_MAX, 300).astype(int)]))


    @functools.lru_cache(maxsize=None)
    def _distances(t):
        return np.abs(eta_sums(t, _N_MAX))[_IDX]


    def convergence_figure(selected):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=_IDX, y=0.5 / np.sqrt(_IDX), mode="lines", name="1/(2√N)",
                                 line=dict(color=C["ink2"], width=1.5, dash="dash"),
                                 hovertemplate="1/(2√N) at N = %{x}: %{y:.5f}<extra></extra>"))
        shown_grey = False
        for t in ZEROS + [NOT_A_ZERO]:
            if t == selected:
                continue
            is_zero = t != NOT_A_ZERO
            fig.add_trace(go.Scatter(
                x=_IDX, y=_distances(t), mode="lines",
                name="the other zeros" if is_zero else f"t = {t:g} (not a zero)",
                legendgroup="others" if is_zero else "nonzero",
                showlegend=(not is_zero) or not shown_grey,
                line=dict(color=C["muted"] if is_zero else C["orange"], width=1 if is_zero else 2),
                opacity=0.7 if is_zero else 1.0,
                hovertemplate=f"t = {t:.4f}<br>N = %{{x}}<br>|S_N| = %{{y:.5f}}<extra></extra>",
            ))
            shown_grey = shown_grey or is_zero
        fig.add_trace(go.Scatter(
            x=_IDX, y=_distances(selected), mode="lines", name=f"t = {selected:.4f} (picked)",
            line=dict(color=C["blue"], width=2.5),
            hovertemplate=f"t = {selected:.4f}<br>N = %{{x}}<br>|S_N| = %{{y:.5f}}<extra></extra>",
        ))
        fig.update_xaxes(type="log", dtick=1, title_text="number of terms N")
        fig.update_yaxes(type="log", dtick=1, title_text="distance from 0", range=[-3.5, 1.0])
        return style(fig, height=395, top=35, bottom=55)


    mo.vstack([
        mo.md(r"""
    ## Every zero tells the same story

    Distance from $0$ after $N$ terms. Every zero lands on the same line,
    $1/(2\sqrt N)$, heading for $0$. At $t = 16$ the spiral closes in on
    $\eta(\tfrac12 + 16i) \approx 2.54$ instead: **not a zero**.
    """),
        pick_t,
        convergence_figure(pick_t.value),
    ])
    return


@app.cell(hide_code=True)
def part3(mo):
    mo.md(r"""
    # Part 3 · How Riemann did it

    Riemann had no computer, yet around 1859 he found the first zero to be about
    $14.14$. Carl Siegel found how in Riemann's notes, and published it in 1932.
    """)
    return


@app.cell(hide_code=True)
def make_real(mo):
    mo.md(r"""
    ## Step 1: make zeta real

    Riemann's **mirror formula**: on our line, zeta is its own mirror image (complex
    conjugate) turned by $-2\theta$, where $\theta$ is a real angle:

    $$\zeta\!\left(\tfrac12 + it\right) = e^{-2i\theta(t)}\;\overline{\zeta\!\left(\tfrac12 + it\right)}, \qquad \theta(t) \approx \frac t2 \ln\frac{t}{2\pi} - \frac t2 - \frac\pi8 + \frac{1}{48t}$$

    Turn zeta by $+\theta$: $\;Z(t) = e^{i\theta}\,\zeta$. Why is that real? Write $\zeta$ for $\zeta(\tfrac12 + it)$.
    """)
    return


@app.cell(hide_code=True)
def frag_real_conj(mo):
    mo.md(r"""
    **1.** Multiply the mirror formula by $e^{2i\theta}$: $\qquad \overline{\zeta} = e^{2i\theta}\,\zeta$

    **2.** Take the mirror image of $Z$ ($\theta$ is real, so $\overline{e^{i\theta}} = e^{-i\theta}$): $\qquad \overline{Z} = e^{-i\theta}\,\overline{\zeta}$
    """)
    return


@app.cell(hide_code=True)
def frag_real_sub(mo):
    mo.md(r"""
    **3.** Put 1 into 2:

    $$\overline{Z} = e^{-i\theta}\cdot e^{2i\theta}\,\zeta = e^{i\theta}\,\zeta = Z$$
    """)
    return


@app.cell(hide_code=True)
def frag_real_done(mo):
    mo.md(r"""
    If $Z = a + bi$, then $\overline{Z} = Z$ says $a - bi = a + bi$, so $b = 0$: **$Z$ is real.**
    Turning never changes length, so $Z = 0$ exactly when $\zeta = 0$.
    """)
    return


@app.cell(hide_code=True)
def z_stays_real(C, ZEROS, add_player, arrow_frame, arrows, fmt_c, fmt_r, go, make_subplots,
                 math, mo, np, plane_window, style, theta, zeta_on_line):
    def turning_figure(t_from=10.0, t_to=35.0, step=0.2):
        """zeta(1/2 + it) wanders, but turned by theta(t) it always lands on the real axis."""
        ts = np.round(np.arange(t_from, t_to + 1e-9, step), 3)
        zetas = zeta_on_line(ts)
        turned = (np.exp(1j * theta(ts)) * zetas).real
        curve_ts = np.arange(t_from, t_to + 1e-9, 0.05)
        curve = (np.exp(1j * theta(curve_ts)) * zeta_on_line(curve_ts)).real
        radius = 1.1 * float(np.abs(zetas).max())
        ppu = 280 / (2 * radius)

        def line(t):
            d = complex(math.cos(theta(t)), -math.sin(theta(t)))  # direction at angle -theta
            return [-1.5 * radius * d, 1.5 * radius * d]

        def caption(k):
            return (f"<b>t = {ts[k]:.2f}</b>: ζ = {fmt_c(zetas[k], 3)}, θ(t) = {fmt_r(theta(ts[k]), 3)}, "
                    f"turned: Z = {fmt_r(turned[k], 3, plus=True)}")

        z0 = complex(zetas[0])
        fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.1)
        l0 = line(ts[0])
        fig.add_trace(go.Scatter(x=[l0[0].real, l0[1].real], y=[l0[0].imag, l0[1].imag],
                                 mode="lines", name="the line at angle −θ", hoverinfo="skip",
                                 line=dict(color=C["muted"], width=1.2, dash="dash")), row=1, col=1)
        fig.add_trace(arrows([0, z0.conjugate()], C["muted"], 2, ppu, "its mirror image",
                             dash="dot"), row=1, col=1)
        fig.add_trace(arrows([0, z0], C["blue"], 3.5, ppu, "ζ(½ + it)", max_head=14), row=1, col=1)
        fig.add_trace(arrows([0, complex(turned[0], 0)], C["aqua"], 3.5, ppu,
                             "Z(t) = e<sup>iθ</sup>ζ", max_head=14), row=1, col=1)
        fig.add_trace(go.Scatter(x=[ts[0]], y=[turned[0]], mode="markers", showlegend=False,
                                 marker=dict(size=11, color=C["aqua"],
                                             line=dict(width=2, color=C["surface"])),
                                 hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(x=curve_ts, y=curve, mode="lines", showlegend=False,
                                 line=dict(color=C["aqua"], width=2),
                                 hovertemplate="t = %{x:.2f}<br>Z = %{y:.4f}<extra></extra>"),
                      row=1, col=2)
        inside = [z for z in ZEROS if t_from <= z <= t_to]
        fig.add_trace(go.Scatter(x=inside, y=[0] * len(inside), mode="markers", showlegend=False,
                                 marker=dict(size=8, color=C["ink"]),
                                 hovertemplate="zero at t = %{x:.4f}<extra></extra>"), row=1, col=2)
        frames = []
        for k, t in enumerate(ts):
            z, lk = complex(zetas[k]), line(t)
            frames.append(go.Frame(name=f"{t:.3f}", traces=[0, 1, 2, 3, 4],
                                   layout=dict(title_text=caption(k)), data=[
                dict(x=[lk[0].real, lk[1].real], y=[lk[0].imag, lk[1].imag]),
                arrow_frame([0, z.conjugate()], ppu),
                arrow_frame([0, z], ppu, max_head=14),
                arrow_frame([0, complex(turned[k], 0)], ppu, max_head=14),
                dict(x=[t], y=[turned[k]], xaxis="x2", yaxis="y2"),
            ]))
        fig.frames = frames
        fig.update_layout(title_text=caption(0))
        plane_window(fig, 0j, radius, row=1, col=1, xtitle=False)
        fig.update_xaxes(range=[t_from, t_to], row=1, col=2)
        fig.update_yaxes(title_text="Z(t)", row=1, col=2)
        style(fig, height=450, legend="top", top=62, bottom=85)
        return add_player(fig, duration=90, step_names=[f"{t:.3f}" for t in ts],
                          step_labels=[f"{t:.2f}" for t in ts], prefix="t = ", quiet_ticks=True)


    mo.vstack([
        mo.md(r"""
    ## Watching $Z$ stay real

    Zeta (blue) wanders, but always sits on the dashed line at angle $-\theta$. Turned by
    $\theta$ it lands on the real axis (aqua), passing through $0$ at each zero. Press **▶ Play**.
    """),
        turning_figure(),
    ])
    return


@app.cell(hide_code=True)
def pairs_math(mo):
    mo.md(r"""
    ## Step 2: arrows pair up into cosines

    Riemann replaced zeta's endless sum by its first $N = \lfloor\sqrt{t/2\pi}\,\rfloor$
    arrows, their mirror images, and a small correction:

    $$\zeta\!\left(\tfrac12 + it\right) \approx \sum_{n=1}^{N} \frac{e^{-it\ln n}}{\sqrt n} \;+\; e^{-2i\theta(t)} \sum_{n=1}^{N} \frac{e^{+it\ln n}}{\sqrt n} \;+\; \text{correction}$$
    """)
    return


@app.cell(hide_code=True)
def frag_pairs_turn(mo):
    mo.md(r"""
    **Turn by $\theta$.** Term $n$ and its mirror become a pair, with $\varphi = \theta(t) - t\ln n$:

    $$e^{i\theta}\cdot\frac{e^{-it\ln n}}{\sqrt n} = \frac{e^{i\varphi}}{\sqrt n} \qquad\text{and}\qquad e^{i\theta}\cdot e^{-2i\theta}\,\frac{e^{it\ln n}}{\sqrt n} = \frac{e^{-i\varphi}}{\sqrt n}$$
    """)
    return


@app.cell(hide_code=True)
def frag_pairs_cos(mo):
    mo.md(r"""
    **Add the pair.** The vertical parts cancel and the horizontal parts add:

    $$e^{i\varphi} + e^{-i\varphi} = (\cos\varphi + i\sin\varphi) + (\cos\varphi - i\sin\varphi) = 2\cos\varphi$$
    """)
    return


@app.cell(hide_code=True)
def rs_formula(mo):
    mo.md(r"""
    ## The Riemann–Siegel formula

    Adding up the pairs:

    $$Z(t) \approx 2\sum_{n=1}^{N} \frac{\cos\!\big(\theta(t) - t\ln n\big)}{\sqrt n} + R(t), \qquad N = \Big\lfloor\sqrt{t/2\pi}\,\Big\rfloor$$
    """)
    return


@app.cell(hide_code=True)
def frag_rs_correction(mo):
    mo.md(r"""
    The first term of Riemann's correction, with $p = \sqrt{t/2\pi} - N$ (the part after the decimal point):

    $$R(t) \approx (-1)^{N-1}\Big(\frac{t}{2\pi}\Big)^{-1/4}\, \frac{\cos\!\big(2\pi(p^2 - p - \tfrac1{16})\big)}{\cos(2\pi p)}$$
    """)
    return


@app.cell(hide_code=True)
def frag_rs_one(mo):
    mo.md(r"""
    Near $t = 14$, $\sqrt{t/2\pi} \approx 1.5$, so $N = 1$: **just one cosine.**

    $$Z(t) \approx 2\cos\theta(t) + R(t)$$
    """)
    return


@app.cell(hide_code=True)
def _(TWO_PI, math, theta):
    # Riemann's formula, with his first correction term.

    def terms_needed(t):
        """N = floor(sqrt(t / 2 pi)): how many cosines the formula keeps."""
        return int(math.sqrt(t / TWO_PI))


    def correction(t):
        """R(t) = (-1)^(N-1) (t/2pi)^(-1/4) cos(2pi(p^2 - p - 1/16)) / cos(2pi p)."""
        a = math.sqrt(t / TWO_PI)
        m = int(a)
        p = a - m
        if abs(math.cos(TWO_PI * p)) < 1e-9:  # removable 0/0 at p = 1/4, 3/4
            p += 1e-6
        coeff = math.cos(TWO_PI * (p * p - p - 1 / 16)) / math.cos(TWO_PI * p)
        return (-1) ** (m - 1) * (t / TWO_PI) ** -0.25 * coeff


    def riemann_z(t):
        """Z(t) by Riemann's formula: N cosines plus the correction."""
        th = theta(t)
        cosines = sum(2 * math.cos(th - t * math.log(n)) / math.sqrt(n)
                      for n in range(1, terms_needed(t) + 1))
        return cosines + correction(t)
    return (riemann_z,)


@app.cell(hide_code=True)
def _(crossing, math, riemann_z, theta):
    ONE_COSINE_ZERO = crossing(lambda t: math.cos(theta(t)), 14.0, 15.0)  # 14.5179
    CORRECTED_ZERO = crossing(riemann_z, 14.0, 14.5)                      # 14.1372
    return CORRECTED_ZERO, ONE_COSINE_ZERO


@app.cell(hide_code=True)
def _(math, theta, theta_slope):
    # Solving 2 cos theta(t) + R = 0 the way you would by hand. At the zero,
    # sqrt(t/2pi) is almost exactly 3/2, so p = 1/2 and R has a closed form.
    R_HALF = math.sqrt(2 / 3) * math.sin(math.pi / 8)  # (9/4)^(-1/4) * C0(1/2)
    THETA_STAR = -math.acos(-R_HALF / 2)               # the angle between theta(14) and theta(14.5)
    NEWTON = [13.0]  # a rough first guess, so the first tangent step is visible
    for _ in range(3):
        NEWTON.append(float(NEWTON[-1] + (THETA_STAR - theta(NEWTON[-1])) / theta_slope(NEWTON[-1])))
    return NEWTON, R_HALF, THETA_STAR


@app.cell(hide_code=True)
def riemann_equation(R_HALF, mo):
    mo.md(rf"""
    ## The equation for the first zero

    Set the one-cosine formula equal to zero:

    $$2\cos\theta(t) + R(t) = 0$$

    Near the zero, $\sqrt{{t/2\pi}}$ is almost exactly $\tfrac32$, so $p = \tfrac12$ and the
    correction has a simple value:

    $$R = \Big(\tfrac94\Big)^{{-1/4}} \cdot \frac{{\cos\!\big(2\pi(\tfrac14 - \tfrac12 - \tfrac1{{16}})\big)}}{{\cos \pi}} = \sqrt{{\tfrac23}}\,\sin\frac{{\pi}}{{8}} = {R_HALF:.4f}$$

    <small>Siegel reports that Riemann used this formula, with its correction series, for his
    calculations; the notes don't spell out every step, so this is the calculation the formula
    leads to.</small>
    """)
    return


@app.cell(hide_code=True)
def frag_eq_angle(R_HALF, THETA_STAR, mo, theta):
    mo.md(rf"""
    **So the angle is known.** $\cos\theta = -R/2 = {-R_HALF / 2:.4f}$. The angle with that
    cosine lying between $\theta(14) = {theta(14):.4f}$ and $\theta(14.5) = {theta(14.5):.4f}$ is

    $$\theta^* = {THETA_STAR:.4f}$$

    All that's left: find the $t$ with $\theta(t) = \theta^*$.
    """)
    return


@app.cell(hide_code=True)
def newton_slide(C, NEWTON, THETA_STAR, add_player, fmt_r, go, mo, np, style, theta,
                 theta_slope):
    def newton_figure():
        """Newton's method on theta(t) = theta*: follow the tangent line, not the curve."""
        grid = np.linspace(12.8, 14.5, 200)

        def tangent(k):
            tk, tn = NEWTON[k], NEWTON[k + 1]
            span = np.array([min(tk, tn) - 0.25, max(tk, tn) + 0.25])
            return span, theta(tk) + theta_slope(tk) * (span - tk)

        def caption(k):
            return (f"<b>step {k + 1}</b>: the tangent at t = {NEWTON[k]:.4f} reaches θ* at "
                    f"t = {NEWTON[k + 1]:.4f}")

        span0, line0 = tangent(0)
        fig = go.Figure(data=[
            go.Scatter(x=grid, y=theta(grid), mode="lines", name="θ(t)",
                       line=dict(color=C["ink"], width=2.5),
                       hovertemplate="t = %{x:.3f}<br>θ = %{y:.4f}<extra></extra>"),
            go.Scatter(x=[grid[0], grid[-1]], y=[THETA_STAR, THETA_STAR], mode="lines",
                       name=f"θ* = {fmt_r(THETA_STAR)}", line=dict(color=C["orange"], width=1.5),
                       hoverinfo="skip"),
            go.Scatter(x=span0, y=line0, mode="lines", name="tangent line",
                       line=dict(color=C["blue"], width=2, dash="dash"), hoverinfo="skip"),
            go.Scatter(x=[NEWTON[0]], y=[theta(NEWTON[0])], mode="markers", name="current guess",
                       marker=dict(size=11, color=C["blue"]), hoverinfo="skip"),
            go.Scatter(x=[NEWTON[1], NEWTON[1]], y=[THETA_STAR, theta(NEWTON[1])],
                       mode="lines+markers", name="next guess",
                       line=dict(color=C["orange"], width=1.2, dash="dot"),
                       marker=dict(size=[11, 0], color=C["orange"]), hoverinfo="skip"),
        ])
        frames = []
        for k in range(len(NEWTON) - 1):
            span, line = tangent(k)
            tk, tn = NEWTON[k], NEWTON[k + 1]
            frames.append(go.Frame(name=str(k), traces=[2, 3, 4], layout=dict(title_text=caption(k)),
                                   data=[dict(x=span, y=line), dict(x=[tk], y=[theta(tk)]),
                                         dict(x=[tn, tn], y=[THETA_STAR, theta(tn)])]))
        fig.frames = frames
        fig.update_layout(title_text=caption(0))
        fig.update_xaxes(range=[12.8, 14.5])
        fig.update_yaxes(range=[-2.35, -1.4], title_text="θ(t)")
        style(fig, height=250, legend="right", top=40, bottom=75)
        return add_player(fig, duration=1600, step_names=[str(k) for k in range(len(NEWTON) - 1)],
                          step_labels=[f"t{k + 1}" for k in range(len(NEWTON) - 1)],
                          prefix="step: ")


    mo.vstack([
        mo.md(r"""
    ## Solving it by hand: Newton's method

    At a guess $t_k$, follow the **tangent line** of $\theta$ instead of the curve; where it reaches $\theta^*$ is the next guess:

    $$t_{k+1} = t_k + \frac{\theta^* - \theta(t_k)}{\theta'(t_k)}, \qquad \theta'(t) = \frac12 \ln\frac{t}{2\pi} - \frac{1}{48t^2}\quad\text{(differentiate $\theta$ term by term)}$$
    """),
        newton_figure(),
    ])
    return


@app.cell(hide_code=True)
def frag_newton_steps(NEWTON, THETA_STAR, TWO_PI, mo, theta, theta_slope):
    _t0, _t1, _t2, _t3 = NEWTON
    mo.md(rf"""
    From $t_0 = 13$: $\;t_1 = 13 + \frac{{{THETA_STAR - theta(_t0):.4f}}}{{{theta_slope(_t0):.4f}}} = {_t1:.4f}$, $\;t_2 = {_t2:.4f}$, $\;t_3 = {_t3:.4f}$: each step roughly doubles the
    correct digits. In Riemann's variable $x = t/2\pi$, that's $x = {_t3 / TWO_PI:.5f} \approx \tfrac94$
    (his notes: $2.250466$).
    """)
    return


@app.cell(hide_code=True)
def how_close(C, CORRECTED_ZERO, GAMMA1, ONE_COSINE_ZERO, go, make_subplots, mo, style):
    def number_line_figure():
        lo, hi = 14.130, 14.145
        fig = make_subplots(rows=2, cols=1, row_heights=[0.28, 0.72], vertical_spacing=0.16)
        fig.add_trace(go.Scatter(x=[lo, hi, hi, lo, lo], y=[-1, -1, 1, 1, -1], fill="toself",
                                 mode="lines", line=dict(width=0), fillcolor="rgba(137,135,129,0.18)",
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[ONE_COSINE_ZERO], y=[0], mode="markers", showlegend=False,
                                 marker=dict(size=13, color=C["blue"]),
                                 hovertemplate=f"one cosine alone: {ONE_COSINE_ZERO:.4f}<extra></extra>"),
                      row=1, col=1)
        fig.add_annotation(x=ONE_COSINE_ZERO, y=0, ax=0, ay=-32, showarrow=True, arrowhead=0,
                           arrowwidth=1, arrowcolor=C["muted"],
                           text=f"one cosine alone: {ONE_COSINE_ZERO:.4f}", row=1, col=1)
        fig.add_annotation(x=(lo + hi) / 2, y=0, ax=0, ay=-32, showarrow=True, arrowhead=0,
                           arrowwidth=1, arrowcolor=C["muted"],
                           text="everything else (zoomed in below)", row=1, col=1)
        close = [
            ("true value", GAMMA1, C["ink"], "diamond", 45, f"{GAMMA1:.6f}…"),
            ("Gram, 1895 (next in print)", 14.135, C["muted"], "circle", -45, "14.135"),
            ("one cosine + correction (ours)", CORRECTED_ZERO, C["aqua"], "circle", 92,
             f"{CORRECTED_ZERO:.4f}"),
            ("Riemann's notes, Siegel's reading (1932)", 14.1386, C["orange"], "circle", -92, "14.1386"),
            ("Riemann's notes, 2024 re-reading", 14.1401, C["orange"], "square", 45, "14.1401"),
        ]
        for name, x0, color, symbol, ay, shown in close:
            fig.add_trace(go.Scatter(x=[x0], y=[0], mode="markers", showlegend=False,
                                     marker=dict(size=13, color=color, symbol=symbol,
                                                 line=dict(width=2, color=C["surface"])),
                                     hovertemplate=f"{name}: {shown}<extra></extra>"), row=2, col=1)
            fig.add_annotation(x=x0, y=0, ax=0, ay=ay, showarrow=True, arrowhead=0, arrowwidth=1,
                               arrowcolor=C["muted"], text=f"<b>{shown}</b><br>{name}", row=2, col=1)
        fig.update_xaxes(range=[14.0, 14.6], dtick=0.1, row=1, col=1)
        fig.update_xaxes(range=[lo, hi], dtick=0.005, row=2, col=1)
        fig.update_yaxes(range=[-1, 1], showticklabels=False, showgrid=False, zeroline=True,
                         zerolinewidth=2)
        fig.update_yaxes(title_text="14.0 to 14.6", row=1, col=1)
        fig.update_yaxes(title_text="zoomed in", row=2, col=1)
        return style(fig, height=425, top=20, bottom=40)


    mo.vstack([
        mo.md(r"""
    ## How close did Riemann get?

    One cosine alone misses by almost $0.4$. With the correction, it lands within
    $0.003$ of the truth, right next to the value in Riemann's own notes. Nothing
    closer appeared in print until Gram, in 1895.
    """),
        number_line_figure(),
    ])
    return


@app.cell(hide_code=True)
def beyond(ALL_ZEROS, C, TWO_PI, crossing, go, mo, np, riemann_z, style, z_from_eta):
    def beyond_figure(t_to=60.0):
        grid = np.arange(10.0, t_to + 1e-9, 0.04)
        exact = z_from_eta(grid)
        approx = np.array([riemann_z(t) for t in grid])
        brackets = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1)
                    if (approx[i] > 0) != (approx[i + 1] > 0)]
        found = [crossing(riemann_z, lo, hi) for lo, hi in brackets]
        gap = float(np.abs(exact - approx).max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=exact, mode="lines", name="true Z(t), by computer",
                                 line=dict(color=C["muted"], width=7), opacity=0.55,
                                 hovertemplate="t = %{x:.2f}<br>Z = %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=grid, y=approx, mode="lines", name="Riemann's formula",
                                 line=dict(color=C["blue"], width=2),
                                 hovertemplate="t = %{x:.2f}<br>formula: %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=ALL_ZEROS, y=[0] * len(ALL_ZEROS), mode="markers", name="the zeros",
            marker=dict(size=10, color=C["aqua"], line=dict(width=2, color=C["surface"])),
            customdata=np.column_stack([found, np.array(found) - np.array(ALL_ZEROS)]),
            hovertemplate=("true zero %{x:.4f}<br>Riemann's formula: %{customdata[0]:.4f}"
                           "<br>off by %{customdata[1]:+.4f}<extra></extra>")))
        # An m-th cosine joins when sqrt(t/2pi) reaches m, at t = 2 pi m^2.
        for m, label in ((2, "second cosine joins"), (3, "third cosine joins")):
            fig.add_vline(x=TWO_PI * m * m, line=dict(color=C["ink2"], width=1, dash="dot"),
                          annotation_text=label, annotation_position="top right")
        fig.update_xaxes(title_text="t")
        fig.update_yaxes(title_text="Z(t)")
        return (style(fig, height=395, top=35, bottom=50), gap,
                max(abs(f - z) for f, z in zip(found, ALL_ZEROS)))


    _fig, _gap, _worst = beyond_figure()
    mo.vstack([
        mo.md(rf"""
    ## Beyond the first zero

    A new cosine joins each time $\sqrt{{t/2\pi}}$ reaches a whole number: the second at
    $t = 8\pi \approx 25.13$, the third at $18\pi \approx 56.55$. The formula stays within
    ${_gap:.3f}$ of the true $Z$, and each of its 13 zeros is within ${_worst:.4f}$ of a
    true one. Hover over the dots to compare.
    """),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def ivt(mo):
    mo.md(r"""
    ## The intermediate value theorem

    If $f$ is **continuous** on $[a, b]$ (its graph has no jumps), and $f(a) < 0 < f(b)$,
    then $f(c) = 0$ for some $c$ between $a$ and $b$.

    *A continuous path from below the axis to above it has to cross the axis.*
    """)
    return


@app.cell(hide_code=True)
def frag_ivt_why(mo):
    mo.md(r"""
    **Why we need $Z$, not $\zeta$.** Zeta is complex: it can swing around $0$ without
    touching it, and "changing sign" means nothing for a complex number. $Z(t)$ is real
    and continuous, so a sign change forces it through $0$, and $Z = 0$ exactly when
    $\zeta = 0$.
    """)
    return


@app.cell(hide_code=True)
def frag_ivt_example(C, fmt_r, go, mo, np, style, z_from_eta):
    def ivt_figure(a=14.0, b=14.5):
        grid = np.linspace(13.85, 14.65, 60)
        za, zb = float(z_from_eta(a)), float(z_from_eta(b))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=z_from_eta(grid), mode="lines", name="Z(t)",
                                 line=dict(color=C["ink"], width=2.5),
                                 hovertemplate="t = %{x:.3f}<br>Z = %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[a], y=[za], mode="markers+text",
                                 text=[f"Z({a:g}) = {fmt_r(za)}"], textposition="bottom right",
                                 name=f"Z({a:g}) < 0", marker=dict(size=11, color=C["orange"]),
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[b], y=[zb], mode="markers+text",
                                 text=[f"Z({b:g}) = {fmt_r(zb, plus=True)}"], textposition="top left",
                                 name=f"Z({b:g}) > 0", marker=dict(size=11, color=C["blue"]),
                                 hoverinfo="skip"))
        fig.update_traces(textfont=dict(color=C["ink2"], size=13))
        fig.update_yaxes(title_text="Z(t)", zeroline=True, zerolinewidth=2)
        return style(fig, height=205, legend="right", top=15, bottom=30), za, zb


    _fig, _za, _zb = ivt_figure()
    mo.vstack([
        mo.md(rf"""
    **Example.** $Z(14) = {_za:.4f} < 0$ and $Z(14.5) = {_zb:+.4f} > 0$, so there is a zero
    between $14$ and $14.5$, *exactly on* the critical line.
    """),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(np, z_from_eta):
    # Z(t), computed from eta's sums, on [10, 50]; its sign changes bracket the zeros.
    HUNT_GRID = np.arange(10.0, 50.0 + 1e-9, 0.025)
    HUNT_Z = z_from_eta(HUNT_GRID)
    _coarse = np.arange(10.0, 50.0 + 1e-9, 0.1)
    _signs = z_from_eta(_coarse) > 0
    HUNT_BRACKETS = [(float(_coarse[i]), float(_coarse[i + 1])) for i in range(len(_coarse) - 1)
                     if _signs[i] != _signs[i + 1]]
    return HUNT_BRACKETS, HUNT_GRID, HUNT_Z


@app.cell(hide_code=True)
def hunt(C, HUNT_BRACKETS, HUNT_GRID, HUNT_Z, NOT_A_ZERO, add_player, go, make_subplots, math,
         mo, np, pick_t, style, z_from_eta):
    def hunt_figure(t_pick, halvings=30, samples=41):
        lo0, hi0 = min(HUNT_BRACKETS, key=lambda b: abs((b[0] + b[1]) / 2 - t_pick))
        steps = []
        lo, hi = lo0, hi0
        f_lo = z_from_eta(lo)
        for k in range(halvings + 1):
            steps.append((lo, hi))
            mid = (lo + hi) / 2
            f_mid = z_from_eta(mid)
            if (f_mid > 0) == (f_lo > 0):
                lo, f_lo = mid, f_mid
            else:
                hi = mid

        def view(k):
            lo, hi = steps[k]
            w = hi - lo
            ts = np.linspace(lo - 0.5 * w, hi + 0.5 * w, samples)
            zs = z_from_eta(ts)
            top = 1.3 * float(np.abs(zs).max())
            mid = (lo + hi) / 2
            digits = max(1, math.ceil(-math.log10(w)) + 1)
            title = (f"<b>after {k} halving{'s' if k != 1 else ''}</b>: "
                     f"{lo:.{digits}f} < t < {hi:.{digits}f}   (width {w:.1e})")
            right = dict(xaxis="x2", yaxis="y2")
            data = [dict(x=[lo, hi, hi, lo, lo], y=[-top, -top, top, top, -top], **right),
                    dict(x=ts, y=zs, **right), dict(x=[ts[0], ts[-1]], y=[0, 0], **right),
                    dict(x=[mid], y=[float(z_from_eta(mid))], **right)]
            return data, ts, top, title

        views = [view(k) for k in range(halvings + 1)]
        data0, ts0, top0, title0 = views[0]
        fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.1)
        fig.add_trace(go.Scatter(x=HUNT_GRID, y=HUNT_Z, mode="lines", showlegend=False,
                                 line=dict(color=C["ink"], width=2),
                                 hovertemplate="t = %{x:.2f}<br>Z = %{y:.4f}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[(lo0 + hi0) / 2], y=[0], mode="markers", showlegend=False,
                                 marker=dict(size=12, color=C["orange"], symbol="circle-open",
                                             line=dict(width=2.5, color=C["orange"])),
                                 hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(**data0[0], fill="toself", mode="lines", line=dict(width=0),
                                 fillcolor="rgba(42,120,214,0.12)", name="the zero is in here",
                                 hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(**data0[1], mode="lines", name="Z(t)", showlegend=False,
                                 line=dict(color=C["ink"], width=2.5),
                                 hovertemplate="t = %{x}<br>Z = %{y:.3e}<extra></extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(**data0[2], mode="lines", showlegend=False,
                                 line=dict(color=C["axis"], width=1), hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(**data0[3], mode="markers", name="next check: the midpoint",
                                 marker=dict(size=11, color=C["orange"],
                                             line=dict(width=2, color=C["surface"])),
                                 hovertemplate="midpoint %{x}<extra></extra>"), row=1, col=2)
        fig.frames = [go.Frame(name=str(k), data=data, traces=[2, 3, 4, 5], layout={
            "xaxis2": {"range": [ts[0], ts[-1]]}, "yaxis2": {"range": [-top, top]},
            "title": {"text": title}}) for k, (data, ts, top, title) in enumerate(views)]
        fig.update_layout(title_text=title0)
        fig.update_yaxes(title_text="Z(t), t from 10 to 50", row=1, col=1)
        fig.update_xaxes(range=[ts0[0], ts0[-1]], row=1, col=2)
        fig.update_yaxes(range=[-top0, top0], exponentformat="e", title_text="zoomed in",
                         row=1, col=2)
        style(fig, height=405, legend="top", top=62, bottom=85)
        return add_player(fig, duration=650, step_names=[str(k) for k in range(halvings + 1)],
                          step_labels=[str(k) for k in range(halvings + 1)], prefix="halvings: ")


    _note = (r" ($t = 16$ isn't a zero, so this hunts the nearest one.)"
             if pick_t.value == NOT_A_ZERO else "")
    mo.vstack([
        mo.md(rf"""
    ## Finding zeros today: halving

    The general method is **bisection**. On an interval where $Z$ changes sign there is a
    zero (the intermediate value theorem). Check the midpoint, keep the half that still
    changes sign, repeat: about $3.3$ halvings per decimal digit. Press **▶ Play**.{_note}
    """),
        pick_t,
        hunt_figure(pick_t.value),
    ])
    return


@app.cell(hide_code=True)
def summary(mo):
    mo.md(r"""
    ## The whole story

    - Zeta's own series fails on the line, but $\zeta = \eta / (1 - 2^{1-s})$ works, with
      the same zeros.
    """)
    return


@app.cell(hide_code=True)
def frag_story_arrows(mo):
    mo.md(r"""
    - **Euler's formula** turns each term into an arrow of length $1/\sqrt n$, turned
      clockwise by $t\ln n$. At a zero, eta's arrows spiral into $0$, about
      $1/(2\sqrt N)$ away after $N$ terms.
    """)
    return


@app.cell(hide_code=True)
def frag_story_riemann(mo):
    mo.md(r"""
    - **Riemann** turned zeta into the real $Z(t)$ and paired each arrow with its mirror
      image, leaving real cosines. Near $t = 14$ there's just one: solving
      $2\cos\theta(t) + R = 0$ with Newton's method gives $14.137$.
    """)
    return


@app.cell(hide_code=True)
def frag_story_ivt(mo):
    mo.md(r"""
    - **Sign changes** of the real $Z(t)$ prove a zero is there (the intermediate value
      theorem), and halving pins it down to any precision.
    """)
    return


@app.cell(hide_code=True)
def frag_story_end(mo):
    mo.md(r"""
    In 1950 Alan Turing ran Riemann's formula on one of the first electronic computers; its
    descendants have confirmed the first $12{,}363{,}153{,}437{,}138$ zeros all lie on the
    line (Platt and Trudgian, 2021). Proving none were *missed* takes Turing's way of
    counting them. And trillions of checks still don't prove the Riemann hypothesis,
    which is about all infinitely many.

    <small>Sources: C. L. Siegel, "Über Riemanns Nachlass zur analytischen
    Zahlentheorie" (1932); G. Hiary, D. Ireland, M. Kyi, arXiv:2408.00187 (2024), who
    read Riemann's value as $2.250466 \times 2\pi$; D. Platt, T. Trudgian, Bull. LMS 53 (2021).</small>
    """)
    return


if __name__ == "__main__":
    app.run()
