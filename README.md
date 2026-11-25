# Riemann-Code

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/justinstevens42/Riemann-Code/blob/main/finding_zeta_zeros.py/wasm?mode=read)

Interactive slides on **finding the zeros of the Riemann zeta function**, built as a
[marimo](https://marimo.io) notebook. The first zero sits at s = ½ + 14.134725…i; the deck
finds it with nothing fancier than sines, cosines and a lot of adding, then shows how
Riemann found it by hand in the 1850s.

1. **The math.** Zeta's own series fails on the critical line, so rewrite it as
   ζ(s) = η(s) / (1 − 2^(1−s)); Euler's formula then turns every term into an arrow.
2. **Watching it happen.** Animations of one term turning, the first 30 arrows added
   tip to tail, the running total closing in on 0, and the first ten zeros side by side.
3. **How Riemann did it.** Why Z(t) = e^(iθ)ζ is real, how mirror pairs of arrows become
   cosines (the Riemann–Siegel formula), the equation for the first zero solved with
   Newton's method, how close Riemann's own value was, and finally the intermediate value
   theorem and bisection, the way zeros are pinned down today.

## Run it in your browser

Click **Open in molab** above. It runs entirely in your browser (WebAssembly): nothing to
install and no account needed. It takes about 15 seconds to start. Use the arrow keys to
move between slides and reveal each step; the animated charts have their own **▶ Play**
buttons, and molab's full-screen button gives the slides more room.

## Run it on your computer

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Download this repository (green **Code** button, then **Download ZIP**) or clone it.
3. In a terminal, from the repository folder:

   ```sh
   uvx marimo run --sandbox finding_zeta_zeros.py
   ```

It opens in your browser as slides. The first run takes a minute while it installs its
packages into a throwaway environment; nothing is installed globally. To browse it as a
notebook instead (code hidden unless you click "Show code"):

```sh
uvx marimo edit --sandbox finding_zeta_zeros.py
```

## Files

- `finding_zeta_zeros.py`: the deck. Its slide layout is stored inside the file, so the
  file works on its own, wherever it goes.
- `make_slides_layout.py`: rebuilds that layout after cells are added or removed (cells
  named `frag_*` appear one step at a time). Run `uv run python make_slides_layout.py`.

## Sources

- C. L. Siegel, "Über Riemanns Nachlass zur analytischen Zahlentheorie" (1932); English
  translation: [arXiv:1810.05198](https://arxiv.org/abs/1810.05198).
- G. Hiary, D. Ireland, M. Kyi, "A method for verifying the generalized Riemann
  hypothesis", [arXiv:2408.00187](https://arxiv.org/abs/2408.00187) (2024), for the
  re-reading of Riemann's value.
- D. Platt, T. Trudgian, "The Riemann hypothesis is true up to 3·10¹²", Bull. LMS 53
  (2021), [arXiv:2004.09765](https://arxiv.org/abs/2004.09765).
