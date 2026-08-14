"""Week 10 — Confusion-Matrix Explorer (SPA317 / SPA442, Accuracy Assessment).

Edit a confusion matrix and watch overall accuracy, producer's and user's
accuracy, kappa, quantity and allocation disagreement, and error-adjusted area
with 95% confidence intervals update live. Rows = MAP, columns = REFERENCE.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

st.set_page_config(page_title='Confusion-Matrix Explorer', layout='wide')

st.markdown(
    f"<h2 style='color:{INK};margin-bottom:0'>Confusion-Matrix Explorer</h2>"
    f"<p style='color:{INK_SOFT};margin-top:2px'>Week 10 · Accuracy Assessment · "
    f"rows are the <b>map</b>, columns are the <b>reference</b>. Edit any cell and "
    f"watch the numbers move.</p>", unsafe_allow_html=True)


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

    if st.button('Reset matrix', use_container_width=True):
        load_scheme(st.session_state.scheme); st.rerun()

    st.divider()
    st.subheader('Teaching presets')
    if st.button('Majority-class trap', use_container_width=True,
                 help='Label almost everything one dominant class'):
        Z = np.zeros((n, n)); Z[0] = [85 if j == 0 else round(15 / (n - 1)) for j in range(n)]
        st.session_state.M = Z; st.rerun()
    if st.button('Pure allocation error', use_container_width=True,
                 help='Right amount of each class, wrong places (swap)'):
        Z = np.full((n, n), 4.0); np.fill_diagonal(Z, 30.0)
        st.session_state.M = Z; st.rerun()
    if st.button('Pure quantity error', use_container_width=True,
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


# ---- editable matrix -----------------------------------------------------
left, right = st.columns([1.05, 1.0])

with left:
    st.markdown('#### The matrix (edit any cell)')
    df = pd.DataFrame(st.session_state.M.astype(int),
                      index=[f'map: {c}' for c in names],
                      columns=[f'ref: {c}' for c in names])
    edited = st.data_editor(df, use_container_width=True,
                            column_config={c: st.column_config.NumberColumn(
                                min_value=0, step=1, format='%d') for c in df.columns})
    M = edited.to_numpy(float)
    st.session_state.M = M
    N = M.sum()
    if N == 0:
        st.warning('Add some points to the matrix.'); st.stop()

    oa, q, a = A.disagreement(M)
    kap = A.kappa(M)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Overall accuracy', f'{oa*100:.1f}%')
    c2.metric('Kappa', f'{kap:.2f}', help='Reportable but weak — see the disagreement split')
    c3.metric('Quantity disagr.', f'{q*100:.1f}%', help='Wrong amount of a class')
    c4.metric('Allocation disagr.', f'{a*100:.1f}%', help='Right amount, wrong place')

    # disagreement stacked bar
    figb, axb = plt.subplots(figsize=(6.2, 1.1)); figb.patch.set_facecolor(PAPER)
    axb.barh(0, oa * 100, color=GOOD, edgecolor='white')
    axb.barh(0, q * 100, left=oa * 100, color=WARN, edgecolor='white')
    axb.barh(0, a * 100, left=(oa + q) * 100, color=BAD, edgecolor='white')
    axb.text(oa * 50, 0, f'agreement {oa*100:.0f}%', ha='center', va='center',
             color='white', fontsize=9, fontweight='bold')
    if a > 0.04:
        axb.text(oa * 100 + q * 100 + a * 50, 0, f'allocation {a*100:.0f}%',
                 ha='center', va='center', color='white', fontsize=8.5, fontweight='bold')
    axb.set_xlim(0, 100); axb.set_ylim(-0.6, 0.6); axb.axis('off')
    axb.set_title('agreement + quantity + allocation = 100%', fontsize=9,
                  color=INK, loc='left')
    st.pyplot(figb, use_container_width=True)

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
    st.pyplot(figp, use_container_width=True)
    st.caption("Producer's = 1 − omission (did the map find real class j). "
               "User's = 1 − commission (can you trust a map label). "
               "This chart uses the raw sample counts; the area table below shows "
               "the Olofsson-adjusted producer's accuracy, weighted by mapped area, "
               "so the two can differ.")


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
st.dataframe(tbl, use_container_width=True, hide_index=True)

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
st.pyplot(figa, use_container_width=True)

st.caption(f"Area-weighted overall accuracy {res['oa_w']*100:.1f}%. "
           "An area estimate without its confidence interval is not defensible.")
