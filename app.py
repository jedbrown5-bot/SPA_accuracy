"""Week 10 — Confusion-Matrix Explorer (SPA317 / SPA442, Accuracy Assessment).

Edit a confusion matrix and watch overall accuracy, producer's and user's
accuracy, kappa, quantity and allocation disagreement, and error-adjusted area
with 95% confidence intervals update live. Rows = MAP, columns = REFERENCE.

Informative extras:
  - a live plain-language "Reading this matrix" diagnosis (weakest class and why,
    biggest confusion, whether the error is amount or placement, what to fix);
  - a per-class error table (correct / omitted / over-claimed, with producer's
    and user's accuracy);
  - a "show the working" panel that builds every number from the current matrix;
  - the matrix with its margins: totals, producer's and user's accuracy, omission
    and commission at the end of every row and column;
  - a "Quantity vs allocation" tab: a next-step walkthrough that starts at the
    confusion matrix and highlights, on the matrix itself, the cells feeding
    each number (diagonal, a class's row, its column, the pairing), then splits
    the disagreement: pairs are allocation, leftovers are quantity;
  - an "Agreement by luck" tab: random label allocation with your class totals,
    the chance baseline that kappa corrects against.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

import accuracy_metrics as A

# ---- house palette -------------------------------------------------------
PAPER = '#fbfaf7'; INK = '#1a1a1a'; INK_SOFT = '#555555'; FAINT = '#dddddd'
WATER = '#1d4e89'; FOREST = '#2e7d32'; BURNT = '#7a2e2e'; PASTURE = '#7cb342'
URBAN = '#9aa0a8'; BARE = '#e08a1e'
GOOD = '#2e7d32'; WARN = '#e08a1e'; BAD = '#b23b3b'

SCHEMES = {
    'Hastings — 5 class (real map)': dict(
        names=['Water', 'Forest', 'Burnt', 'Pasture', 'Urban'],
        cols=[WATER, FOREST, BURNT, PASTURE, URBAN],
        M=[[47, 0, 0, 1, 0], [1, 78, 6, 4, 1], [0, 11, 34, 2, 1],
           [1, 5, 1, 41, 6], [0, 1, 0, 5, 28]],
        weights=[0.10, 0.46, 0.13, 0.23, 0.08]),
    'Synthetic — 3 class (teaching)': dict(
        names=['Water', 'Forest', 'Pasture'],
        cols=[WATER, FOREST, PASTURE],
        M=[[34, 1, 2], [1, 28, 5], [2, 3, 24]],
        weights=[0.37, 0.34, 0.29]),
}

def diagnose(M, names, oa, q, a):
    """Plain-language reading of the current matrix, as a list of markdown lines."""
    M = np.asarray(M, float)
    n = len(names)
    diag = np.diag(M)
    rt = M.sum(1)
    ct = M.sum(0)
    lines = []
    lines.append(
        f"**Overall accuracy is {oa*100:.0f}%** for the scene: of every checked point, that is "
        f"the share the map got right.")

    # weakest link: the lowest producer's or user's accuracy across all classes
    cand = []
    for k in range(n):
        if ct[k] > 0:
            cand.append((diag[k] / ct[k], k, 'producer'))
        if rt[k] > 0:
            cand.append((diag[k] / rt[k], k, 'user'))
    if cand:
        val, k, kind = min(cand, key=lambda x: x[0])
        if kind == 'producer':
            miss = int(ct[k] - diag[k])
            lines.append(
                f"**Weakest link: {names[k]} is being missed.** Its producer's "
                f"accuracy is {val*100:.0f}%: {miss} of the {int(ct[k])} points that "
                f"are really {names[k]} were labelled something else (omission).")
        else:
            over = int(rt[k] - diag[k])
            lines.append(
                f"**Weakest link: {names[k]} is being over-claimed.** Its user's "
                f"accuracy is {val*100:.0f}%: {over} of the {int(rt[k])} points the map "
                f"calls {names[k]} are really something else (commission).")

    # biggest single confusion (largest off-diagonal cell)
    off = M.copy()
    np.fill_diagonal(off, 0)
    if off.max() > 0:
        i, j = np.unravel_index(int(np.argmax(off)), off.shape)
        lines.append(
            f"**Biggest single confusion:** {int(off[i, j])} points the map calls "
            f"{names[i]} are really {names[j]}.")

    # quantity vs allocation: what to fix
    if (q + a) > 1e-9:
        if a >= q:
            lines.append(
                f"**Most of the error is placement, not amount** (allocation "
                f"{a*100:.0f}% vs quantity {q*100:.0f}%). The class totals are about "
                f"right but pixels sit in the wrong spots, so reach for a better feature "
                f"or a cleanup filter, not a different total.")
        else:
            lines.append(
                f"**Most of the error is amount, not placement** (quantity "
                f"{q*100:.0f}% vs allocation {a*100:.0f}%). A class total is off: the "
                f"classifier is producing too much or too little of it.")
    else:
        lines.append("**No disagreement at all:** this is a perfect map, every point on the diagonal.")
    return lines



@st.cache_data(show_spinner=False)
def _simulate_chance(M_tuple, n_sims=1000):
    """Overall accuracy of n_sims random maps sharing this matrix's marginals."""
    Mm = np.array(M_tuple, float)
    rt = Mm.sum(1).astype(int)
    ct = Mm.sum(0).astype(int)
    ref = np.repeat(np.arange(len(ct)), ct)
    lab = np.repeat(np.arange(len(rt)), rt)
    rng = np.random.default_rng(42)
    out = np.empty(n_sims)
    for s in range(n_sims):
        rng.shuffle(lab)
        out[s] = float((lab == ref).mean())
    return out



# cell styles for the step-by-step matrix (background AND text colour set, so
# every highlight stays readable on both the light and the dark theme)
QA_DIAG = 'background-color:#e6f2e6;color:#14501a;font-weight:700'
QA_ERR = 'background-color:#f7e9e9;color:#b23b3b;font-weight:700'
QA_PAIR = 'background-color:#fdeed3;color:#8a5a10;font-weight:700'
QA_TOT = 'background-color:#f2efe9;color:#1a1a1a'
QA_TOT_HL = 'background-color:#f2efe9;color:#b23b3b;font-weight:700'
QA_DIMMED = 'color:#9aa0a8'


def qa_matrix_view(Mv, names_v, cell_css=None, dim_others=False,
                   row_total_css=None, col_total_css=None):
    # render the confusion matrix (with totals) as a styled table, highlighting
    # exactly the cells a walkthrough step is using
    n_ = len(names_v)
    rt_v = Mv.sum(1).astype(int)
    ct_v = Mv.sum(0).astype(int)
    ridx = [f'map: {c}' for c in names_v] + ['column total']
    rcols = [f'ref: {c}' for c in names_v] + ['row total']
    dfm = pd.DataFrame('', index=ridx, columns=rcols, dtype=object)
    for i in range(n_):
        for j in range(n_):
            dfm.iloc[i, j] = f'{int(Mv[i, j])}'
        dfm.iloc[i, n_] = f'{rt_v[i]}'
    for j in range(n_):
        dfm.iloc[n_, j] = f'{ct_v[j]}'
    dfm.iloc[n_, n_] = f'{int(Mv.sum())}'
    cell_css = cell_css or {}
    row_total_css = row_total_css or {}
    col_total_css = col_total_css or {}

    def _sty(d):
        s = pd.DataFrame('', index=d.index, columns=d.columns)
        s.iloc[n_, :] = QA_TOT
        s.iloc[:, n_] = QA_TOT
        for i in range(n_):
            for j in range(n_):
                if (i, j) in cell_css:
                    s.iloc[i, j] = cell_css[(i, j)]
                elif dim_others:
                    s.iloc[i, j] = QA_DIMMED
        for i, css in row_total_css.items():
            s.iloc[i, n_] = css
        for j, css in col_total_css.items():
            s.iloc[n_, j] = css
        return s

    st.dataframe(dfm.style.apply(_sty, axis=None), width='stretch')


st.set_page_config(page_title='Confusion-Matrix Explorer', layout='wide')

st.markdown('## Confusion-Matrix Explorer')
st.caption('Week 10 · Accuracy Assessment · rows are the **map**, columns are the '
           '**reference**. Edit any cell and watch the numbers move.')


# ---- state ---------------------------------------------------------------
def load_scheme(key):
    sc = SCHEMES[key]
    st.session_state.scheme = key
    st.session_state.M = np.array(sc['M'], float)
    st.session_state.weights = list(sc['weights'])

if 'scheme' not in st.session_state:
    load_scheme('Hastings — 5 class (real map)')

with st.sidebar:
    st.header('Scene')
    sk = st.selectbox('Class scheme', list(SCHEMES.keys()),
                      index=list(SCHEMES).index(st.session_state.scheme))
    if sk != st.session_state.scheme:
        load_scheme(sk)
        st.rerun()
    names = SCHEMES[st.session_state.scheme]['names']
    cols = SCHEMES[st.session_state.scheme]['cols']
    n = len(names)

    if st.button('Reset matrix', width='stretch'):
        load_scheme(st.session_state.scheme); st.rerun()

    st.divider()
    st.subheader('Teaching presets')
    if st.button('Majority-class trap', width='stretch',
                 help='Label almost everything one dominant class'):
        Z = np.zeros((n, n)); Z[0] = [85 if j == 0 else round(15 / (n - 1)) for j in range(n)]
        st.session_state.M = Z; st.rerun()
    if st.button('Pure allocation error', width='stretch',
                 help='Right amount of each class, wrong places (swap)'):
        Z = np.full((n, n), 4.0); np.fill_diagonal(Z, 30.0)
        st.session_state.M = Z; st.rerun()
    if st.button('Pure quantity error', width='stretch',
                 help='One class systematically over-mapped'):
        Z = np.zeros((n, n)); np.fill_diagonal(Z, 30.0)
        if n >= 2:
            Z[0, 1] = 25   # map over-claims class 0 at the expense of class 1
        st.session_state.M = Z; st.rerun()

    st.divider()
    st.subheader('Area estimation')
    total_area = st.number_input('Total mapped area (ha)', 1000, 10_000_000,
                                 40000, step=1000)
    st.caption('Map-area weight of each class (strata). Should sum to 1.')
    w = []
    for k in range(n):
        w.append(st.number_input(names[k], 0.0, 1.0,
                                 float(st.session_state.weights[k]), step=0.01,
                                 key=f'w{k}'))
    wsum = sum(w) or 1.0
    weights = np.array(w) / wsum


tab_explore, tab_steps, tab_chance = st.tabs(
    ['Explore the matrix', 'Quantity vs allocation: step by step',
     'Agreement by luck: kappa'])

with tab_explore:
    # ---- editable matrix -----------------------------------------------------
    left, right = st.columns([1.05, 1.0])

    with left:
        st.markdown('#### The matrix (edit any cell)')
        df = pd.DataFrame(st.session_state.M.astype(int),
                          index=[f'map: {c}' for c in names],
                          columns=[f'ref: {c}' for c in names])
        edited = st.data_editor(df, width='stretch',
                                column_config={c: st.column_config.NumberColumn(
                                    min_value=0, step=1, format='%d') for c in df.columns})
        M = edited.to_numpy(float)
        st.session_state.M = M
        N = M.sum()
        if N == 0:
            st.warning('Add some points to the matrix.'); st.stop()

        oa, q, a = A.disagreement(M)
        kap = A.kappa(M)

        # the matrix with its margins: totals, producer's/user's, omission/commission
        pa = A.producers_accuracy(M) * 100
        ua = A.users_accuracy(M) * 100
        rt_m = M.sum(1)
        ct_m = M.sum(0)
        st.markdown("##### With its margins")
        ridx = [f'map: {c}' for c in names] + ['column total', "Producer's %", 'Omission %']
        rcols = [f'ref: {c}' for c in names] + ['row total', "User's %", 'Commission %']
        ann = pd.DataFrame('', index=ridx, columns=rcols, dtype=object)
        for i in range(n):
            for j in range(n):
                ann.iloc[i, j] = f'{int(M[i, j])}'
            ann.iloc[i, n] = f'{int(rt_m[i])}'
            ann.iloc[i, n + 1] = f'{ua[i]:.0f}' if rt_m[i] > 0 else 'n/a'
            ann.iloc[i, n + 2] = f'{100 - ua[i]:.0f}' if rt_m[i] > 0 else 'n/a'
        for j in range(n):
            ann.iloc[n, j] = f'{int(ct_m[j])}'
            ann.iloc[n + 1, j] = f'{pa[j]:.0f}' if ct_m[j] > 0 else 'n/a'
            ann.iloc[n + 2, j] = f'{100 - pa[j]:.0f}' if ct_m[j] > 0 else 'n/a'
        ann.iloc[n, n] = f'{int(N)}'

        def _margin_style(dfa):
            # every styled cell sets BOTH background and text colour, so the table
            # stays readable whether the app runs on the light or the dark theme
            sty = pd.DataFrame('', index=dfa.index, columns=dfa.columns)
            sty.iloc[n, :] = 'background-color:#f2efe9;color:#1a1a1a'
            sty.iloc[:, n] = 'background-color:#f2efe9;color:#1a1a1a'
            sty.iloc[n + 1, :n] = f'background-color:#e8eef6;color:{WATER};font-weight:600'
            sty.iloc[n + 2, :n] = f'background-color:#f7e9e9;color:{BAD}'
            sty.iloc[:n, n + 1] = f'background-color:#e9f2e9;color:{FOREST};font-weight:600'
            sty.iloc[:n, n + 2] = f'background-color:#f7e9e9;color:{BAD}'
            for i in range(n):
                sty.iloc[i, i] = 'background-color:#e6f2e6;color:#14501a;font-weight:700'
            return sty

        st.dataframe(ann.style.apply(_margin_style, axis=None), width='stretch')
        st.caption("Down a reference column: producer's and omission (did the map find the real thing). "
                   "Across a map row: user's and commission (can you trust the label). "
                   "The green diagonal is the agreement.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Overall accuracy', f'{oa*100:.1f}%',
                  help='Of all the checked points, the share the map got right. The headline; it can hide a failing class behind a dominant one.')
        c2.metric('Kappa', f'{kap:.2f}',
                  help='Overall agreement corrected against a random-labelling baseline. Legacy: report it if expected, lean on the others.')
        c3.metric('Quantity disagr.', f'{q*100:.1f}%',
                  help='The map has the wrong AMOUNT of a class: its total is off, too much or too little.')
        c4.metric('Allocation disagr.', f'{a*100:.1f}%',
                  help='The amounts are right but sit in the wrong PLACES: a swap across the scene.')

        # disagreement stacked bar: label inside when the segment is wide enough,
        # otherwise below the bar in the segment's own colour (never clipped)
        figb, axb = plt.subplots(figsize=(6.2, 1.5)); figb.patch.set_facecolor(PAPER)
        segs = [('agreement', oa * 100, FOREST), ('quantity', q * 100, WATER),
                ('allocation', a * 100, BARE)]
        edge = 0.0; last_below = -99.0; below_row = 0
        for label, val, colr in segs:
            if val > 0.3:
                axb.barh(0, val, left=edge, color=colr, edgecolor='white')
                centre = edge + val / 2
                if val >= 16:
                    axb.text(centre, 0, f'{label} {val:.0f}%', ha='center', va='center',
                             color=(INK if colr == BARE else 'white'), fontsize=9,
                             fontweight='bold')
                else:
                    below_row = 1 if (centre - last_below) < 20 and below_row == 0 else 0
                    ytxt = -0.78 - 0.42 * below_row
                    axb.plot([centre, centre], [-0.42, ytxt + 0.12], color=colr, lw=1)
                    axb.text(centre, ytxt, f'{label} {val:.1f}%', ha='center', va='top',
                             color=colr, fontsize=8.5, fontweight='bold')
                    last_below = centre
            edge += val
        axb.set_xlim(0, 100); axb.set_ylim(-1.75, 0.65); axb.axis('off')
        axb.set_title('agreement + quantity + allocation = 100%', fontsize=9,
                      color=INK, loc='left')
        st.pyplot(figb, width='stretch')

    with right:
        st.markdown('#### Per-class accuracy')
        pa = A.producers_accuracy(M) * 100
        ua = A.users_accuracy(M) * 100
        figp, axp = plt.subplots(figsize=(6.0, 0.55 * n + 0.6)); figp.patch.set_facecolor(PAPER)
        y = np.arange(n)[::-1]
        axp.barh(y + 0.18, pa, height=0.34, color=WATER, label="Producer's")
        axp.barh(y - 0.18, ua, height=0.34, color=FOREST, label="User's")
        rt = M.sum(1)
        for k in range(n):
            axp.text(pa[k] + 1, y[k] + 0.18, f'{pa[k]:.0f}', va='center', fontsize=8, color=WATER)
            # a class the map never claims has no user's accuracy (0/0), so say n/a
            ua_lab = f'{ua[k]:.0f}' if rt[k] > 0 else 'n/a'
            axp.text(ua[k] + 1, y[k] - 0.18, ua_lab, va='center', fontsize=8, color=FOREST)
        axp.set_yticks(y); axp.set_yticklabels(names, fontsize=9)
        axp.set_xlim(0, 108); axp.set_xlabel('accuracy (%)', fontsize=9)
        for s in axp.spines.values(): s.set_color(FAINT)
        axp.legend(fontsize=8, loc='lower right', framealpha=0.9)
        axp.set_facecolor('white')
        st.pyplot(figp, width='stretch')
        st.caption("Producer's = 1 − omission (did the map find real class j). "
                   "User's = 1 − commission (can you trust a map label). "
                   "This chart uses the raw sample counts; the area table below shows "
                   "the Olofsson-adjusted producer's accuracy, weighted by mapped area, "
                   "so the two can differ.")

        # explicit per-class error counts: correct, omitted, over-claimed
        diagc = np.diag(M).astype(int)
        ctc = M.sum(0).astype(int)
        rtc = M.sum(1).astype(int)
        err_tbl = pd.DataFrame({
            'Class': names,
            'Correct': diagc,
            'Omitted (missed)': (ctc - diagc),
            "Producer's %": [f'{pa[k]:.0f}' if ctc[k] > 0 else 'n/a' for k in range(n)],
            'Over-claimed': (rtc - diagc),
            "User's %": [f'{ua[k]:.0f}' if rtc[k] > 0 else 'n/a' for k in range(n)],
        })
        st.dataframe(err_tbl, width='stretch', hide_index=True)
        st.caption('Omitted = real class the map missed (down the reference column). '
                   'Over-claimed = points the map wrongly labelled this class (across the map row).')


    # ---- plain-language reading ----------------------------------------------
    st.markdown('#### Reading this matrix')
    st.caption('A plain-language diagnosis that updates as you edit the matrix, the way a reviewer would read it.')
    with st.container(border=True):
        for ln in diagnose(M, names, oa, q, a):
            st.markdown('- ' + ln)


    # ---- show the working ----------------------------------------------------
    with st.expander('Show the working: how each number is calculated from this matrix'):
        diag = np.diag(M).astype(int)
        rows_t = M.sum(1).astype(int)
        cols_t = M.sum(0).astype(int)
        Ni = int(N)

        st.markdown('##### Overall accuracy: the diagonal over the total')
        st.code(f'correct  = {" + ".join(str(d) for d in diag)} = {diag.sum()}\n'
                f'checked  = {Ni}\n'
                f'overall accuracy = {diag.sum()} / {Ni} = {oa*100:.1f}%', language=None)

        st.markdown("##### Producer's and user's accuracy: same cell, two denominators")
        wk = st.selectbox('Work through one class', names, index=min(1, n - 1),
                          key='working_class')
        kk = names.index(wk)
        st.code(
            f'diagonal cell for {wk}: {diag[kk]}\n\n'
            f"producer's accuracy (down the REFERENCE column)\n"
            f'  column total = {" + ".join(str(int(v)) for v in M[:, kk])} = {cols_t[kk]}\n'
            f"  PA = {diag[kk]} / {cols_t[kk]} = "
            + (f'{diag[kk]/cols_t[kk]*100:.1f}%   (omission = {100-diag[kk]/cols_t[kk]*100:.1f}%)'
               if cols_t[kk] else 'n/a (no reference points in this class)') + '\n\n'
            f"user's accuracy (across the MAP row)\n"
            f'  row total = {" + ".join(str(int(v)) for v in M[kk, :])} = {rows_t[kk]}\n'
            f"  UA = {diag[kk]} / {rows_t[kk]} = "
            + (f'{diag[kk]/rows_t[kk]*100:.1f}%   (commission = {100-diag[kk]/rows_t[kk]*100:.1f}%)'
               if rows_t[kk] else 'n/a (the map never claims this class)'), language=None)

        st.markdown('##### Kappa: correcting against a chance baseline (report it, do not lean on it)')
        E = float((M.sum(1) * M.sum(0)).sum() / (N * N))
        st.code(
            'expected agreement from the marginals\n'
            f'  E = sum(row total x column total) / N^2\n'
            f'    = ({" + ".join(f"{r}x{c}" for r, c in zip(rows_t, cols_t))}) / {Ni}^2 = {E:.3f}\n'
            f'kappa = (OA - E) / (1 - E) = ({oa:.3f} - {E:.3f}) / (1 - {E:.3f}) = {kap:.2f}',
            language=None)

        st.markdown('##### Quantity and allocation: why the map is wrong, not just how much')
        qg = np.abs(rows_t - cols_t)
        st.code(
            'per-class quantity mismatch |map row total - reference column total|\n'
            + '\n'.join(f'  {names[g]:<10s} |{rows_t[g]} - {cols_t[g]}| = {qg[g]}'
                        for g in range(n)) + '\n'
            f'quantity   Q = (sum of mismatches) / 2 / N = {qg.sum()} / 2 / {Ni} = {q*100:.1f}%\n'
            f'total disagreement D = 1 - OA = {(1-oa)*100:.1f}%\n'
            f'allocation A = D - Q = {(1-oa)*100:.1f}% - {q*100:.1f}% = {a*100:.1f}%',
            language=None)
        st.caption('Quantity: the map has the wrong amount of a class. Allocation: the '
                   'right amount in the wrong places. Pontius and Millones (2011), '
                   'equation 7: agreement + quantity + allocation = 100%.')

    # ---- area estimation -----------------------------------------------------
    st.markdown('#### Error-adjusted area with 95% confidence intervals')
    st.caption('Olofsson et al. (2014) stratified estimator. The naive pixel count '
               '(area weight × total) is biased; the adjusted estimate corrects it '
               'using the sample, and the interval shows how sure you are.')

    res = A.olofsson_area(M, weights, total_area)
    naive = weights * total_area
    tbl = pd.DataFrame({
        'Class': names,
        'Naive count (ha)': np.round(naive).astype(int),
        'Adjusted (ha)': np.round(res['area']).astype(int),
        '± 95% CI (ha)': np.round(res['ci95']).astype(int),
        "Producer's %": np.round(res['pa'] * 100, 1),
        "User's %": np.round(res['ua'] * 100, 1),
    })
    st.dataframe(tbl, width='stretch', hide_index=True)

    pick = st.selectbox('Show one class as a chart', names, index=min(2, n - 1))
    j = names.index(pick)
    figa, axa = plt.subplots(figsize=(8.5, 1.9)); figa.patch.set_facecolor(PAPER)
    axa.barh(1, naive[j], height=0.42, color=BARE, edgecolor='white')
    axa.text(naive[j], 1, f'  naive {naive[j]:,.0f} ha', va='center', fontsize=9, color=INK)
    axa.errorbar(res['area'][j], 0.2, xerr=res['ci95'][j], fmt='o', color=WATER,
                 ecolor=WATER, elinewidth=2.2, capsize=7, markersize=9)
    axa.text(res['area'][j], -0.35,
             f"adjusted {res['area'][j]:,.0f} ± {res['ci95'][j]:,.0f} ha (95% CI)",
             ha='center', va='top', fontsize=9, color=WATER, fontweight='bold')
    axa.set_ylim(-0.7, 1.5); axa.set_yticks([])
    axa.set_xlim(0, max(naive[j], res['area'][j] + res['ci95'][j]) * 1.15)
    axa.set_xlabel(f'{pick} area (ha)', fontsize=9)
    for s in axa.spines.values(): s.set_color(FAINT)
    axa.spines['left'].set_visible(False); axa.set_facecolor('white')
    st.pyplot(figa, width='stretch')

    st.caption(f"Area-weighted overall accuracy {res['oa_w']*100:.1f}%. "
               "An area estimate without its confidence interval is not defensible.")


# ---- quantity vs allocation: step-by-step walkthrough --------------------
with tab_steps:
    st.markdown('#### Quantity vs allocation, one step at a time')
    st.caption('Runs on whatever matrix is on the Explore tab right now. Every step '
               'highlights, on the matrix itself, exactly the cells it is using. '
               'Rows are the map, columns are the reference.')

    if 'qa_step' not in st.session_state:
        st.session_state.qa_step = 1
    QA_STEPS = 7
    st.session_state.qa_step = min(st.session_state.qa_step, QA_STEPS)
    b1, b2, b3, _sp = st.columns([0.16, 0.22, 0.18, 0.44])
    if b1.button('Back', width='stretch', disabled=st.session_state.qa_step <= 1):
        st.session_state.qa_step -= 1
        st.rerun()
    if b2.button('Next step', width='stretch', type='primary',
                 disabled=st.session_state.qa_step >= QA_STEPS):
        st.session_state.qa_step += 1
        st.rerun()
    if b3.button('Start again', width='stretch', disabled=st.session_state.qa_step <= 1):
        st.session_state.qa_step = 1
        st.rerun()
    step = st.session_state.qa_step
    st.progress(step / QA_STEPS, text=f'Step {step} of {QA_STEPS}')

    diag_q = np.diag(M).astype(int)
    rt_q = M.sum(1).astype(int)
    ct_q = M.sum(0).astype(int)
    Nq = int(N)
    correct_q = int(diag_q.sum())
    wrong_q = Nq - correct_q
    comm_q = rt_q - diag_q
    omis_q = ct_q - diag_q
    pairs_q = np.minimum(comm_q, omis_q)
    left_q = comm_q - omis_q
    A_pts = int(pairs_q.sum())
    Q_pts = int(np.maximum(left_q, 0).sum())

    if step in (3, 4, 5, 6):
        wk_qa = st.selectbox('Class to follow through the steps', names,
                             index=min(2, n - 1), key='qa_class')
        kk = names.index(wk_qa)
        row_off = [int(M[kk, j]) for j in range(n) if j != kk]
        col_off = [int(M[i, kk]) for i in range(n) if i != kk]

    if step == 1:
        st.markdown('##### Step 1 · Start at the confusion matrix')
        st.markdown(
            'Every number in the split comes off this one table. Each **row** is what the '
            'map said, each **column** is what the reference said, and a cell counts the '
            'sample points that landed there. The plan: find what is wrong, then split it '
            'into wrong **amount** (quantity) and wrong **place** (allocation).')
        qa_matrix_view(M, names)
        st.caption('The margins are the row totals (how often the map claimed each class) '
                   'and the column totals (how often each class was really there). The '
                   'whole calculation uses nothing but these cells and totals.')

    elif step == 2:
        st.markdown('##### Step 2 · The diagonal is right; the rest is what we must explain')
        st.markdown(
            'Green cells are agreement: map and reference say the same thing. Every red '
            'cell is a confusion, points the map calls one class that are really another. '
            'The split has to account for every point in the red cells.')
        css = {(i, i): QA_DIAG for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j and M[i, j] > 0:
                    css[(i, j)] = QA_ERR
        qa_matrix_view(M, names, cell_css=css, dim_others=True)
        st.code(f'correct (green diagonal) = {" + ".join(str(d) for d in diag_q)} = {correct_q}\n'
                f'wrong (red cells)        = {Nq} - {correct_q} = {wrong_q}', language=None)

    elif step == 3:
        st.markdown(f"##### Step 3 · Across {wk_qa}'s map row: its over-claims (commission)")
        st.markdown(
            f'Read across the **map: {wk_qa}** row. The green diagonal cell is the {wk_qa} '
            f'the map got right. Every red cell in that row is a point the map **called '
            f'{wk_qa} but is really something else**: an over-claim. Their sum is the row '
            f'total minus the diagonal.')
        css = {(kk, kk): QA_DIAG}
        for j in range(n):
            if j != kk:
                css[(kk, j)] = QA_ERR
        qa_matrix_view(M, names, cell_css=css, dim_others=True,
                       row_total_css={kk: QA_TOT_HL})
        st.code(f'over-claimed ({wk_qa}) = {" + ".join(str(v) for v in row_off)} '
                f'= {int(comm_q[kk])}\n'
                f'check: row total - diagonal = {int(rt_q[kk])} - {int(diag_q[kk])} '
                f'= {int(comm_q[kk])}', language=None)
        if comm_q[kk] > 0:
            rr = M[kk].copy()
            rr[kk] = -1
            jmax = int(np.argmax(rr))
            if M[kk, jmax] > 0:
                st.markdown(f'The biggest over-claim: **{int(M[kk, jmax])} points the map '
                            f'calls {wk_qa} are really {names[jmax]}**. This is the same '
                            f"count behind user's accuracy: 1 minus commission.")

    elif step == 4:
        st.markdown(f"##### Step 4 · Down {wk_qa}'s reference column: its misses (omission)")
        st.markdown(
            f'Now read down the **ref: {wk_qa}** column. Every red cell in the column is a '
            f'point that **really is {wk_qa} but the map called something else**: a miss. '
            f'Their sum is the column total minus the diagonal. Do this row-and-column '
            f'reading for every class and you get the table below.')
        css = {(kk, kk): QA_DIAG}
        for i in range(n):
            if i != kk:
                css[(i, kk)] = QA_ERR
        qa_matrix_view(M, names, cell_css=css, dim_others=True,
                       col_total_css={kk: QA_TOT_HL})
        st.code(f'missed ({wk_qa}) = {" + ".join(str(v) for v in col_off)} '
                f'= {int(omis_q[kk])}\n'
                f'check: column total - diagonal = {int(ct_q[kk])} - {int(diag_q[kk])} '
                f'= {int(omis_q[kk])}', language=None)
        t4 = pd.DataFrame({
            'Class': names,
            'Over-claimed (row)': comm_q,
            'Missed (column)': omis_q,
        })
        st.dataframe(t4, width='stretch', hide_index=True)
        st.caption(f'Both columns sum to the same {wrong_q} wrong points, because every '
                   'red cell is an over-claim for its row class and a miss for its '
                   'column class at the same time.')

    elif step == 5:
        st.markdown('##### Step 5 · Pair up what cancels: that is allocation')
        st.markdown(
            f'{wk_qa} over-claimed **{int(comm_q[kk])}** (across its row) and missed '
            f'**{int(omis_q[kk])}** (down its column), both highlighted in amber. Pair them '
            f'up as far as they go: **{int(pairs_q[kk])} pairs**. Each pair leaves '
            f"{wk_qa}'s total exactly right, the pixels just sit in the wrong places. A "
            f'swap. Swaps are **allocation disagreement**, and you count the pairs for '
            f'every class.')
        css = {(kk, kk): QA_DIAG}
        for j in range(n):
            if j != kk:
                css[(kk, j)] = QA_PAIR
        for i in range(n):
            if i != kk:
                css[(i, kk)] = QA_PAIR
        qa_matrix_view(M, names, cell_css=css, dim_others=True)
        fig3, ax3 = plt.subplots(figsize=(8.2, 0.95 * n + 0.7))
        fig3.patch.set_facecolor(PAPER)
        xmax = max(1, int(max(comm_q.max(), omis_q.max())))
        for g in range(n):
            y0 = (n - 1 - g) * 1.15
            m = int(pairs_q[g])
            for yy, tot, lab in ((y0 + 0.19, int(comm_q[g]), 'over-claimed'),
                                 (y0 - 0.19, int(omis_q[g]), 'missed')):
                if m > 0:
                    ax3.barh(yy, m, height=0.3, color=BARE, edgecolor='white')
                if tot > m:
                    ax3.barh(yy, tot - m, left=m, height=0.3, color=WATER, edgecolor='white')
                ax3.text(tot + xmax * 0.02, yy, f'{lab} {tot}', va='center',
                         fontsize=8.2, color=INK)
            ax3.text(-xmax * 0.03, y0, names[g], ha='right', va='center',
                     fontsize=9.5, color=INK, fontweight='bold')
        ax3.set_xlim(0, xmax * 1.45)
        ax3.set_ylim(-0.8, (n - 1) * 1.15 + 0.8)
        ax3.set_yticks([])
        ax3.set_facecolor('white')
        for s in ax3.spines.values():
            s.set_color(FAINT)
        ax3.spines['left'].set_visible(False)
        hnd = [plt.Rectangle((0, 0), 1, 1, color=BARE),
               plt.Rectangle((0, 0), 1, 1, color=WATER)]
        ax3.legend(hnd, ['pairs up and cancels: allocation', 'left over: quantity'],
                   fontsize=8, loc='upper right', framealpha=0.95)
        st.pyplot(fig3, width='stretch')
        st.code('pairs that cancel per class: '
                + ', '.join(f'{names[g]} {int(pairs_q[g])}' for g in range(n)) + '\n'
                f'allocation points = {" + ".join(str(int(v)) for v in pairs_q)} = {A_pts}\n'
                f'allocation disagreement = {A_pts} / {Nq} = {A_pts / Nq * 100:.1f}%',
                language=None)

    elif step == 6:
        st.markdown('##### Step 6 · The leftover is quantity')
        st.markdown(
            f'Whatever could not pair is a real **amount** problem. {wk_qa} over-claimed '
            f'{int(comm_q[kk])} and missed {int(omis_q[kk])}, so after pairing it has '
            f'**{abs(int(left_q[kk]))} left over**'
            + (', too many on the map.' if left_q[kk] > 0 else
               (', too few on the map.' if left_q[kk] < 0 else
                ': nothing left over, its total is exactly right.'))
            + ' And every surplus point in one class is a deficit point in another, the '
              'same wrong point seen from two sides, so it is counted **once**, not twice. '
              'That is why the textbook formula sums the per-class gaps and halves them.')
        desc = []
        for g in range(n):
            if left_q[g] > 0:
                desc.append(f'{int(left_q[g])} too many')
            elif left_q[g] < 0:
                desc.append(f'{int(-left_q[g])} too few')
            else:
                desc.append('balanced')
        t6 = pd.DataFrame({
            'Class': names,
            'Over-claimed': comm_q,
            'Missed': omis_q,
            'Pairs cancelled': pairs_q,
            'Leftover': desc,
        })
        st.dataframe(t6, width='stretch', hide_index=True)
        surplus = int(np.maximum(left_q, 0).sum())
        deficit = int(np.maximum(-left_q, 0).sum())
        st.code(f'surpluses (too many) = {surplus}    deficits (too few) = {deficit}\n'
                f'same points from two sides, so count once: quantity points = {Q_pts}\n'
                f'quantity disagreement = {Q_pts} / {Nq} = {Q_pts / Nq * 100:.1f}%\n'
                f'(textbook form: sum the gaps {" + ".join(str(int(abs(v))) for v in left_q)} '
                f'= {int(np.abs(left_q).sum())}, then halve = {Q_pts})', language=None)

    else:
        st.markdown('##### Step 7 · Put it back together')
        st.markdown(
            f'Every one of the {wrong_q} wrong points is now explained: **{A_pts} are '
            f'allocation** (right amount, wrong place) and **{Q_pts} are quantity** (wrong '
            f'amount). Add the agreement back on and the three pieces cover every checked '
            f'point.')
        fig5, ax5 = plt.subplots(figsize=(8.0, 1.7))
        fig5.patch.set_facecolor(PAPER)
        edge = 0.0
        last_below = -1e9
        brow = 0
        for lab, val, colr, txtc in [('agreement', correct_q, FOREST, 'white'),
                                     ('quantity', Q_pts, WATER, 'white'),
                                     ('allocation', A_pts, BARE, INK)]:
            if val > 0:
                ax5.barh(0, val, left=edge, color=colr, edgecolor='white')
                pct = val / Nq * 100
                centre = edge + val / 2
                if pct >= 14:
                    ax5.text(centre, 0, f'{lab} {pct:.0f}%', ha='center',
                             va='center', color=txtc, fontsize=9, fontweight='bold')
                else:
                    # below-bar labels stagger onto a second row when the previous
                    # below-bar label is close enough to collide
                    brow = 1 if (centre - last_below) < 0.24 * Nq and brow == 0 else 0
                    ytxt = -0.8 - 0.62 * brow
                    ax5.plot([centre, centre], [-0.42, ytxt + 0.1], color=colr, lw=1)
                    ax5.text(centre, ytxt, f'{lab} {pct:.1f}%', ha='center',
                             va='top', fontsize=8.4, color=colr, fontweight='bold')
                    last_below = centre
            edge += val
        ax5.set_xlim(0, Nq)
        ax5.set_ylim(-2.4, 0.7)
        ax5.axis('off')
        st.pyplot(fig5, width='stretch')
        st.code(f'agreement  {correct_q / Nq * 100:.1f}%\n'
                f'quantity   {Q_pts / Nq * 100:.1f}%\n'
                f'allocation {A_pts / Nq * 100:.1f}%\n'
                f'total      {(correct_q + Q_pts + A_pts) / Nq * 100:.0f}%', language=None)
        if A_pts >= Q_pts:
            st.markdown(
                '**The reading:** most of the error is placement, not amount. The class '
                'totals are close to right, so reach for better features or a cleanup '
                'filter, not a different total.')
        else:
            st.markdown(
                '**The reading:** most of the error is amount. A class total is off, the '
                'classifier is producing too much or too little of something, and that is '
                'what needs fixing first.')

# ---- agreement by luck: the kappa baseline -------------------------------
GRID = 30


def _exact_counts(props, total):
    """Integer counts per class summing to total (largest remainder)."""
    raw = np.asarray(props, float) * total
    base = np.floor(raw).astype(int)
    short = int(total - base.sum())
    if short > 0:
        order = np.argsort(-(raw - base))
        base[order[:short]] += 1
    return base


with tab_chance:
    st.markdown('#### Agreement by luck: what kappa corrects for')
    st.markdown(
        'A map with no skill still gets pixels right by coincidence, and kappa asks how '
        'much. Top row: the truth on the ground, then a **trained map** built to match your '
        'confusion matrix, its errors sitting along the patch edges the way a real '
        'classifier fails, then where it is right. Bottom row: the same truth, a **random '
        'map** with zero skill, the right amount of each class scattered completely at '
        'random, then where it flukes a correct pixel anyway. Kappa measures how far the '
        'top row sits above the bottom row.')

    rt_c = M.sum(1)
    ct_c = M.sum(0)
    Nc = int(N)
    Pe = float((rt_c * ct_c).sum()) / (Nc * Nc)

    npix = GRID * GRID
    ref_counts = _exact_counts(ct_c / Nc, npix)
    map_counts = _exact_counts(rt_c / Nc, npix)

    # a patchy reference landscape: smooth a random field, then cut it into class
    # bands so the class areas match the reference (column) proportions exactly
    rng_ref = np.random.default_rng(7)
    f = np.kron(rng_ref.normal(size=(6, 6)), np.ones((5, 5)))
    for _ in range(4):
        f = (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) + f) / 5
    order = np.argsort(f.ravel())
    ref_grid = np.empty(npix, int)
    pos = 0
    for k in range(n):
        ref_grid[order[pos:pos + ref_counts[k]]] = k
        pos += ref_counts[k]
    ref_grid = ref_grid.reshape(GRID, GRID)

    if 'chance_rolls' not in st.session_state:
        st.session_state.chance_rolls = 0
    if st.button('Roll a new random map', width='stretch'):
        st.session_state.chance_rolls += 1
    rng_roll = np.random.default_rng(1000 + st.session_state.chance_rolls)
    rand_flat = np.repeat(np.arange(n), map_counts)
    rng_roll.shuffle(rand_flat)
    rand_grid = rand_flat.reshape(GRID, GRID)

    match = rand_grid == ref_grid
    hit = int(match.sum())

    # the trained map: the truth mislabelled at your matrix's own rates, with the
    # errors pushed to the patch edges the way a real classifier fails
    rng_t = np.random.default_rng(11)
    refflat = ref_grid.ravel()
    edge_ct = np.zeros_like(ref_grid)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.roll(np.roll(ref_grid, dy, 0), dx, 1)
        edge_ct = edge_ct + (nb != ref_grid)
    edgeflat = edge_ct.ravel().astype(float)
    trained = refflat.copy()
    for j in range(n):
        idx = np.where(refflat == j)[0]
        colj = M[:, j].astype(float)
        if idx.size == 0 or colj.sum() <= 0:
            continue
        cnts_j = _exact_counts(colj / colj.sum(), idx.size)
        n_wrong = int(idx.size - cnts_j[j])
        if n_wrong <= 0:
            continue
        order = np.argsort(-(edgeflat[idx] + rng_t.random(idx.size)))
        wrong_lbls = np.concatenate([np.full(int(cnts_j[i]), i)
                                     for i in range(n) if i != j])
        rng_t.shuffle(wrong_lbls)
        trained[idx[order[:n_wrong]]] = wrong_lbls[:n_wrong]
    trained_grid = trained.reshape(GRID, GRID)
    match_t = trained_grid == ref_grid
    hit_t = int(match_t.sum())

    cmap = ListedColormap(cols)
    skill_cmap = ListedColormap(['#e3e0da', GOOD])
    luck_cmap = ListedColormap(['#e3e0da', BARE])
    figm, axm = plt.subplots(2, 3, figsize=(11.5, 7.8))
    figm.patch.set_facecolor(PAPER)
    panels = [
        (0, 0, ref_grid, cmap, n - 1, 'Reference: the truth', INK, False),
        (0, 1, trained_grid, cmap, n - 1, 'Your trained map: skill', INK, False),
        (0, 2, match_t.astype(int), skill_cmap, 1,
         f'Correct: {hit_t} of {npix} ({hit_t / npix * 100:.0f}%)', GOOD, True),
        (1, 0, ref_grid, cmap, n - 1, 'Reference: the same truth', INK, False),
        (1, 1, rand_grid, cmap, n - 1, 'A random map: zero skill', INK, False),
        (1, 2, match.astype(int), luck_cmap, 1,
         f'Correct by luck: {hit} of {npix} ({hit / npix * 100:.0f}%)', BARE, True),
    ]
    for r, c, img, cm, vmx, title, tcol, bold in panels:
        axm[r, c].imshow(img, cmap=cm, vmin=0, vmax=vmx, interpolation='nearest')
        axm[r, c].set_title(title, fontsize=10, color=tcol,
                            fontweight='bold' if bold else 'normal')
        axm[r, c].set_xticks([])
        axm[r, c].set_yticks([])
        for s in axm[r, c].spines.values():
            s.set_color(FAINT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[k]) for k in range(n)]
    figm.legend(handles, names, loc='lower center', ncol=n, fontsize=9, frameon=False)
    figm.subplots_adjust(bottom=0.075, top=0.95, wspace=0.06, hspace=0.12,
                         left=0.02, right=0.98)
    st.pyplot(figm, width='stretch')
    st.caption(f'The reference uses your matrix\'s reference (column) totals; the trained and '
               f'random maps both use its map (row) totals, so all the class amounts agree. '
               f'The trained map mislabels each reference class at the rates in your matrix. '
               f'Roll again and only the random map re-scatters. Skill lands about '
               f'{oa * 100:.0f}%, luck lands about {Pe * 100:.0f}%, and kappa measures the gap.')

    lcol, rcol2 = st.columns([1.0, 1.15])

    with lcol:
        st.markdown('##### The numbers')
        m1, m2 = st.columns(2)
        m1.metric('This random roll', f'{hit / npix * 100:.0f}%',
                  help='The green pixels over the whole grid. Pure coincidence.')
        m2.metric('Expected by luck (Pe)', f'{Pe * 100:.0f}%',
                  help='Sum over classes of (row total x column total) / N squared: what random '
                       'labelling with your class totals scores on average.')
        m3, m4 = st.columns(2)
        m3.metric('Your map (Po)', f'{oa * 100:.0f}%',
                  help='Your overall accuracy, the observed agreement.')
        m4.metric('Kappa', f'{kap:.2f}',
                  help='(Po - Pe) / (1 - Pe): how far above luck your map sits, rescaled so 1 is '
                       'perfect and 0 is no better than luck.')
        st.code(f'kappa = (Po - Pe) / (1 - Pe)\n'
                f'      = ({oa:.2f} - {Pe:.2f}) / (1 - {Pe:.2f}) = {kap:.2f}', language=None)

    with rcol2:
        st.markdown('##### A thousand random rolls')
        sims = _simulate_chance(tuple(map(tuple, M.astype(int).tolist())))
        figh, axh = plt.subplots(figsize=(6.4, 2.7))
        figh.patch.set_facecolor(PAPER)
        axh.hist(sims * 100, bins=24, color='#c9cdd3', edgecolor='white')
        top = axh.get_ylim()[1]
        axh.axvline(Pe * 100, color=BARE, lw=2)
        luck_left = Pe * 100 > 18
        axh.text(Pe * 100 - 1.5 if luck_left else Pe * 100 + 1.5, top * 0.95,
                 f'luck: {Pe * 100:.0f}%', color=BARE, fontsize=9, fontweight='bold',
                 va='top', ha='right' if luck_left else 'left')
        axh.axvline(oa * 100, color=FOREST, lw=2)
        if abs(oa - Pe) > 0.02:
            map_right = oa * 100 < 78
            axh.text(oa * 100 + 1.5 if map_right else oa * 100 - 1.5, top * 0.80,
                     f'your map: {oa * 100:.0f}%', color=FOREST, fontsize=9,
                     fontweight='bold', va='top', ha='left' if map_right else 'right')
        axh.set_xlim(0, 100)
        axh.set_yticks([])
        axh.set_xlabel('overall accuracy of a random map (%)', fontsize=9)
        for s in axh.spines.values():
            s.set_color(FAINT)
        axh.spines['left'].set_visible(False)
        axh.set_facecolor('white')
        st.pyplot(figh, width='stretch')

    with st.container(border=True):
        st.markdown(
            f"**The reading.** Luck alone is worth about {Pe * 100:.0f}% here, so of your map's "
            f"{oa * 100:.0f}%, only the gap above the luck line is evidence of skill. Kappa rescales "
            f"that gap by the room left above luck, which is why kappa always sits at or below "
            f"overall accuracy. Now load the majority-class trap preset in the sidebar and come "
            f"back: with one dominant class the random map is nearly all one colour, the luck "
            f"line climbs on its own, and kappa collapses even though overall accuracy still "
            f"looks healthy. That sensitivity to class totals is also why the field moved past "
            f"kappa: the baseline moves with the scene, not with the map's skill.")
