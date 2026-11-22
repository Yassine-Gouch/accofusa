"""US Accidents - interactive dashboard (Streamlit + DuckDB, all 7.5M rows).
Run locally:  pip install -r requirements.txt  ->  streamlit run app.py
The Parquet file is downloaded automatically from Hugging Face on first start,
so nothing big has to be uploaded with the app (only app.py + requirements.txt)."""
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="US Accidents Dashboard", page_icon="🚗", layout="wide")

DATA_URL = "https://huggingface.co/datasets/yassinegouch/accidentofusa/resolve/main/accidents.parquet"
INFRA = ["Traffic_Signal", "Junction", "Crossing", "Stop", "Railway", "Station", "Give_Way"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEV_COLORS = {"1": "#22c55e", "2": "#3b82f6", "3": "#f59e0b", "4": "#ef4444"}
PLACE = "City || ', ' || State"


def show(fig):
    """st.plotly_chart that works on old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


@st.cache_resource(show_spinner="First start: downloading the dataset (~120 MB)...")
def get_con():
    local = Path("accidents.parquet")
    if not local.exists():
        local = Path(tempfile.gettempdir()) / "accidents.parquet"
    if not local.exists():
        part = str(local) + ".part"
        urllib.request.urlretrieve(DATA_URL, part)
        os.replace(part, local)
    con = duckdb.connect()
    con.execute("SET memory_limit='700MB'; SET threads=2;")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{local.as_posix()}')")
    return con


@st.cache_data(show_spinner=False, max_entries=400)
def _q(sql, key):
    cur = get_con().cursor()
    cur.execute(sql, json.loads(key))
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def Q(sql, params=()):
    return _q(sql, json.dumps(list(params), default=str))


# ---------------- Sidebar (filters) ----------------
st.sidebar.title("🚗 US Accidents")
meta = Q("SELECT min(Year) AS a, max(Year) AS b, count(*) AS n FROM t")
y_min, y_max, total = int(meta.a[0]), int(meta.b[0]), int(meta.n[0])
all_states = Q("SELECT DISTINCT State FROM t WHERE State IS NOT NULL ORDER BY 1").State.tolist()
years = st.sidebar.slider("Years", y_min, y_max, (y_min, y_max))
states = st.sidebar.multiselect("States (empty = all)", all_states)
sevs = st.sidebar.multiselect("Severity", ["1", "2", "3", "4"], default=["1", "2", "3", "4"]) or ["1", "2", "3", "4"]

W = "Year BETWEEN ? AND ? AND list_contains(?, Severity)"
P = [years[0], years[1], [int(s) for s in sevs]]
if states:
    W += " AND list_contains(?, State)"
    P.append(states)


def F(sql, extra_where="", extra_params=()):
    """Run `sql` where {w} is replaced by the sidebar filters (+ optional extra condition)."""
    return Q(sql.format(w=W + extra_where), P + list(extra_params))


kpi = F("SELECT count(*) AS n, avg(Severity) AS avg_sev, 100*avg((Severity>=3)::INT) AS severe, "
        f"count(DISTINCT {PLACE}) AS cities FROM t WHERE {{w}}").iloc[0]
if kpi.n == 0:
    st.warning("No data for these filters.")
    st.stop()
st.sidebar.caption(f"{int(kpi.n):,} accidents selected out of {total:,}")

tabs = st.tabs(["Overview", "City comparison", "Time", "Weather", "Infrastructure", "Map", "Model"])

# ---------------- Overview ----------------
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Accidents", f"{int(kpi.n):,}")
    c[1].metric("Avg severity", f"{kpi.avg_sev:.2f}")
    c[2].metric("Severe (3-4)", f"{kpi.severe:.1f}%")
    c[3].metric("Cities", f"{int(kpi.cities):,}")
    a, b = st.columns(2)
    s = F("SELECT Severity, count(*) AS n FROM t WHERE {w} GROUP BY 1 ORDER BY 1")
    s["Severity"] = s.Severity.astype(str)
    with a:
        show(px.bar(s, x="Severity", y="n", color="Severity", color_discrete_map=SEV_COLORS, title="Severity distribution"))
    st_all = F("SELECT State, count(*) AS n FROM t WHERE {w} GROUP BY 1 ORDER BY n DESC")
    with b:
        show(px.bar(st_all.head(10), x="n", y="State", orientation="h", title="Top 10 states").update_yaxes(autorange="reversed"))
    show(px.choropleth(st_all, locations="State", locationmode="USA-states", color="n", scope="usa",
                       color_continuous_scale="Reds", title="Accidents by state"))

# ---------------- City comparison ----------------
with tabs[1]:
    ranked = F(f"SELECT {PLACE} AS place, count(*) AS n FROM t WHERE {{w}} GROUP BY 1 ORDER BY n DESC LIMIT 500").place.tolist()
    pick = st.multiselect("Choose 2 to 6 cities to compare", ranked, default=ranked[:3], max_selections=6)
    if len(pick) < 2:
        st.info("Select at least two cities.")
    else:
        cw, cp = f" AND ({PLACE}) IN (SELECT unnest(?))", [pick]
        inf = ", ".join(f"100*avg({c}::INT) AS {c}" for c in INFRA)
        summ = F(f"SELECT {PLACE} AS City, count(*) AS Accidents, avg(Severity) AS avg_sev, 100*avg((Severity>=3)::INT) AS severe, "
                 f"mode(Hour) AS peak, 100*avg((Sunrise_Sunset='Night')::INT) AS night, {inf} FROM t WHERE {{w}} GROUP BY 1 ORDER BY Accidents DESC", cw, cp)
        show_t = pd.DataFrame({"City": summ.City, "Accidents": summ.Accidents, "Avg severity": summ.avg_sev.round(2),
                               "Severe % (3-4)": summ.severe.round(1), "Peak hour": summ.peak.astype(int),
                               "Night %": summ.night.round(1), "Traffic signal %": summ.Traffic_Signal.round(1),
                               "Junction %": summ.Junction.round(1)}).set_index("City")
        table(show_t)
        a, b = st.columns(2)
        with a:
            show(px.bar(summ, x="City", y="Accidents", color="City", title="Total accidents"))
        sv = F(f"SELECT {PLACE} AS Place, Severity, count(*) AS n FROM t WHERE {{w}} GROUP BY 1,2", cw, cp)
        sv["Severity"] = sv.Severity.astype(str)
        sv["pct"] = sv.n / sv.groupby("Place").n.transform("sum") * 100
        with b:
            show(px.bar(sv, x="Place", y="pct", color="Severity", color_discrete_map=SEV_COLORS, title="Severity mix (%)",
                        labels={"pct": "% of city accidents", "Place": ""}))
        dim = st.radio("Compare by", ["Hour", "Weekday", "Month", "Year"], horizontal=True)
        pr = F(f"SELECT {PLACE} AS Place, {dim} AS x, count(*) AS n FROM t WHERE {{w}} GROUP BY 1,2 ORDER BY 1,2", cw, cp)
        pr["pct"] = pr.n / pr.groupby("Place").n.transform("sum") * 100
        if dim == "Weekday":
            pr["x"] = pr.x.map(dict(enumerate(DAYS)))
        if dim == "Month":
            pr["x"] = pr.x.map(dict(enumerate(MONTHS, 1)))
        show(px.line(pr, x="x", y="pct", color="Place", markers=True, labels={"x": dim, "pct": "% of city accidents"},
                     title=f"Accidents by {dim.lower()} (% of each city's total, so sizes are comparable)"))
        im = summ.melt(id_vars="City", value_vars=INFRA, var_name="Feature", value_name="%")
        show(px.bar(im, x="Feature", y="%", color="City", barmode="group", title="Road infrastructure at accident location (%)"))

# ---------------- Time ----------------
with tabs[2]:
    def by(col):
        return F(f"SELECT {col} AS x, count(*) AS n FROM t WHERE {{w}} GROUP BY 1 ORDER BY 1")
    a, b = st.columns(2)
    h, w_, m = by("Hour"), by("Weekday"), by("Month")
    w_["x"], m["x"] = w_.x.map(dict(enumerate(DAYS))), m.x.map(dict(enumerate(MONTHS, 1)))
    with a:
        show(px.bar(h, x="x", y="n", title="By hour (rush hours ~7-8 AM and 4-5 PM)", labels={"x": "Hour"}))
        show(px.bar(m, x="x", y="n", title="By month", labels={"x": "Month"}))
    heat = F("SELECT Weekday, Hour, count(*) AS n FROM t WHERE {w} GROUP BY 1,2").pivot(index="Weekday", columns="Hour", values="n")
    heat.index = [DAYS[i] for i in heat.index]
    with b:
        show(px.bar(w_, x="x", y="n", title="By day of week", labels={"x": "Day"}))
        show(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd", title="Weekday x hour heatmap"))

# ---------------- Weather ----------------
with tabs[3]:
    n = st.slider("Top weather conditions", 5, 20, 10)
    wd = F("WITH f AS (SELECT Weather_Condition AS w, Severity FROM t WHERE {w} AND Weather_Condition IS NOT NULL), "
           f"top AS (SELECT w FROM f GROUP BY 1 ORDER BY count(*) DESC LIMIT {int(n)}) "
           "SELECT w AS Weather, Severity, count(*) AS n FROM f WHERE w IN (SELECT w FROM top) GROUP BY 1,2")
    wd["Severity"] = wd.Severity.astype(str)
    tot = wd.groupby("Weather").n.sum().sort_values(ascending=False)
    wd["pct"] = wd.n / wd.Weather.map(tot) * 100
    a, b = st.columns(2)
    with a:
        show(px.bar(tot.reset_index(), x="n", y="Weather", orientation="h", title="Accidents by weather").update_yaxes(autorange="reversed"))
    with b:
        show(px.bar(wd, x="pct", y="Weather", color="Severity", orientation="h", color_discrete_map=SEV_COLORS,
                    category_orders={"Weather": tot.index.tolist()}, title="Severity mix by weather (%)").update_yaxes(autorange="reversed"))
    vis = F('SELECT Severity, "Visibility(mi)" AS vis FROM (SELECT Severity, "Visibility(mi)" FROM t WHERE {w} '
            'AND "Visibility(mi)" IS NOT NULL) USING SAMPLE 50000 ROWS')
    vis["Severity"] = vis.Severity.astype(str)
    show(px.box(vis, x="Severity", y="vis", color="Severity", color_discrete_map=SEV_COLORS, title="Visibility vs severity",
                labels={"vis": "Visibility (mi)"}))

# ---------------- Infrastructure ----------------
with tabs[4]:
    sel = ", ".join(f"sum({c}::INT)::BIGINT AS n_{c}, avg(Severity) FILTER (WHERE {c}) AS s_{c}" for c in INFRA)
    r = F(f"SELECT {sel} FROM t WHERE {{w}}").iloc[0]
    inf_df = pd.DataFrame({"Feature": INFRA, "n": [r[f"n_{c}"] for c in INFRA], "Avg severity": [r[f"s_{c}"] for c in INFRA]})
    a, b = st.columns(2)
    with a:
        show(px.bar(inf_df.sort_values("n", ascending=False), x="n", y="Feature", orientation="h",
                    title="Accidents near each feature").update_yaxes(autorange="reversed"))
    with b:
        show(px.bar(inf_df, x="Feature", y="Avg severity", title="Average severity by feature"))

# ---------------- Map ----------------
with tabs[5]:
    k = st.slider("Points on map", 1_000, 30_000, 8_000, step=1_000)
    mp = F(f"SELECT Start_Lat, Start_Lng, Severity, {PLACE} AS Place FROM (SELECT Start_Lat, Start_Lng, Severity, City, State "
           f"FROM t WHERE {{w}} AND Start_Lat IS NOT NULL) USING SAMPLE {int(k)} ROWS")
    mp["Severity"] = mp.Severity.astype(str)
    show(px.scatter_geo(mp, lat="Start_Lat", lon="Start_Lng", color="Severity", scope="usa", color_discrete_map=SEV_COLORS,
                        hover_name="Place", opacity=0.6, title="Accident locations (random sample)").update_layout(height=600))

# ---------------- Model ----------------
with tabs[6]:
    st.subheader("Severity prediction results (from Final_Model notebook)")
    res = pd.DataFrame({"Model": ["XGBoost", "LightGBM", "CatBoost", "Voting ensemble (test)"], "Accuracy": [0.7584, 0.7595, 0.7218, 0.7510]})
    show(px.bar(res, x="Model", y="Accuracy", text="Accuracy", range_y=[0.6, 0.8]))
    rep = pd.DataFrame({"Severity": ["1", "2", "3", "4"], "Precision": [0.23, 0.96, 0.47, 0.22], "Recall": [0.57, 0.73, 0.88, 0.37],
                        "F1": [0.33, 0.83, 0.62, 0.27], "Support": [4814, 350090, 81203, 6933]})
    table(rep.set_index("Severity"))
    st.caption("Classes 1 and 4 are rare, so their precision stays low even after SMOTE + undersampling.")
