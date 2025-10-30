import os
import json
import re
from collections import Counter
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import networkx.algorithms.community as nx_comm

import dash                     # <-- add this line
from dash import dcc, html, dash_table
import plotly.graph_objects as go

# ---------- Paths (portable for Render) ----------
BASE_DIR = os.path.dirname(__file__)
GRAPH_PATH = os.path.join(BASE_DIR, "MC3_graph.json")

print("Loading graph from:", GRAPH_PATH)


with open(GRAPH_PATH) as f:
    data = json.load(f)
G = json_graph.node_link_graph(data, edges="edges")
print(G)

# ---------- App constants ----------
TOP_ENTITIES = 40
TOP_ALIASES  = 40

def is_entity(n): return G.nodes[n].get("type") == "Entity"
def is_event(n):  return G.nodes[n].get("type") == "Event"
def is_rel(n):    return G.nodes[n].get("type") == "Relationship"

def is_comm_event(n):
    d = G.nodes[n]
    return is_event(n) and (
        "communication" in str(d.get("subtype","")).lower()
        or "communication" in str(n).lower()
    )

def get_ts(d):
    for k in ("time","timestamp","datetime","date_time","date"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            try: return pd.to_datetime(v)
            except: pass
    return None

# ----- temporal rows (entity, time, comm) -----
rows = []
comm_nodes = [n for n in G if is_comm_event(n)]
for c in comm_nodes:
    ts = get_ts(G.nodes[c])
    if ts is None: 
        continue
    ents = [n for n in (set(G.predecessors(c))|set(G.successors(c))) if is_entity(n)]
    for e in ents:
        rows.append({"entity": e, "time": ts, "comm": c})

if not rows:
    raise SystemExit("No timestamped Communication events detected. Adjust get_ts() keys as needed.")

df = pd.DataFrame(rows).sort_values("time")
min_date = df["time"].dt.normalize().min()
df["day"]  = (df["time"].dt.normalize() - min_date).dt.days
df["hour"] = df["time"].dt.hour

# ====================== Entity→Entity projection ======================
G_proj = nx.Graph()
for n,d in G.nodes(data=True):
    if d.get("type") == "Entity":
        G_proj.add_node(n, **d)

def add_pair(u,v, via):
    if u==v: return
    if not G_proj.has_edge(u,v):
        G_proj.add_edge(u,v, event_count=0, rel_count=0)
    if via=="Event":          G_proj[u][v]["event_count"] += 1
    elif via=="Relationship": G_proj[u][v]["rel_count"] += 1

for n,d in G.nodes(data=True):
    if d.get("type") in ("Event","Relationship"):
        ents = [x for x in set(G.predecessors(n))|set(G.successors(n)) if is_entity(x)]
        via = d.get("type")
        for i in range(len(ents)):
            for j in range(i+1,len(ents)):
                add_pair(ents[i], ents[j], via)

# Communities on projection
comms = nx_comm.louvain_communities(G_proj, seed=42)
comm_label = {}
for i,cset in enumerate(comms):
    label = f"Community {i+1}"
    for n in cset:
        comm_label[n] = label

# ====================== Alias Extraction (ROBUST) ======================
# Whitelist of known codenames (add variants as needed)
KNOWN_ALIASES = {
    "Boss", "The Lookout", "Lookout", "The Middleman", "Middleman",
    "Small Fry", "The Accountant", "Accountant"
}

# Normal English words to ignore even if capitalized
COMMON_CAPS_LOWER = {
    "i","we","you","he","she","it","they","the","a","an","this","that","these","those",
    "please","thank","thanks","can","could","will","would","may","might","your","our","my",
    "his","her","its","need","want","should","one","all","any","some","is","are","was","were",
    "just","have","has","had","do","does","did","and","but","or","if","as","at","by","for","from",
    "on","of","to","with","in","out","over","under","into","about","after","while","very","got",
    "still","then","there","more","even","when","before","again","ok","okay","yes","no"
}

# Command/ack verbs that sometimes sneak in
VERB_ACK_STOP = {
    "yes","roger","received","confirming","confirmed","ack","acknowledged",
    "abort","aborting","send","sent","let"
}

# Context triggers (helps keep genuine handles)
TRIGGERS = [
    r'\bcall (?:me|him|her|them)\s+',
    r'\bthis is\s+',
    r"\bit's\s+",
    r'\bthey call (?:me|him|her|them)\s+',
    r'\baka\s+',
    r'\bcodename\s+'
]

# TitleCase surface (1–2 words, optional “The ”)
ALIAS_RE = re.compile(r'\b(?:The\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b')

# Build vocabulary of entity names/tokens to exclude as aliases
entity_names = {n for n in G.nodes if is_entity(n)}
entity_tokens = set()
for name in entity_names:
    for t in re.findall(r"[A-Za-z]+", name):
        entity_tokens.add(t.lower())

def get_all_text(d: dict) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return " ".join(parts)

def is_entity_name_or_token(a: str) -> bool:
    if a in entity_names:
        return True
    for t in a.split():
        if t.lower() in entity_tokens:
            return True
    return False

def looks_like_alias(a: str) -> bool:
    if a in KNOWN_ALIASES:   # keep knowns through shape checks
        return True
    if len(a) < 3 or len(a.split()) > 3:
        return False
    al = a.lower()
    if al in COMMON_CAPS_LOWER or al in VERB_ACK_STOP:
        return False
    if is_entity_name_or_token(a):
        return False
    return True

def has_alias_context(txt: str, alias: str, window: int = 18) -> bool:
    if re.search(r'[\"“”]' + re.escape(alias) + r'[\"“”]', txt):
        return True
    for trig in TRIGGERS:
        if re.search(trig + re.escape(alias), txt, flags=re.IGNORECASE):
            return True
        m = re.search(trig, txt, flags=re.IGNORECASE)
        if m and re.search(re.escape(alias), txt[m.end(): m.end()+window]):
            return True
    return False

def shape_is_codename(a: str) -> bool:
    parts = a.split()
    if a.startswith("The ") and len(parts)==2:
        return True
    if len(parts)==2 and all(p and p[0].isupper() for p in parts):
        return True
    return False

alias_rows = []
alias_global_freq = Counter()
known_hits = Counter()
total_comm_texts = 0

for c in comm_nodes:
    dnode = G.nodes[c]
    blob = get_all_text(dnode)
    if not blob:
        continue
    total_comm_texts += 1

    # A) KNOWN_ALIASES case-insensitive
    for ka in KNOWN_ALIASES:
        if re.search(rf"\b{re.escape(ka)}\b", blob, flags=re.IGNORECASE):
            known_hits[ka] += 1

    # B) TitleCase candidates + filters
    candidates = set(ALIAS_RE.findall(blob))
    candidates = {a for a in candidates if looks_like_alias(a)}

    strong = set()

    # include known aliases that actually appear in this message
    for ka in KNOWN_ALIASES:
        if re.search(rf"\b{re.escape(ka)}\b", blob, flags=re.IGNORECASE):
            strong.add(ka)

    # include other candidates if they look like real handles
    for a in candidates:
        if has_alias_context(blob, a) or shape_is_codename(a):
            strong.add(a)

    if not strong:
        continue

    ents = [n for n in (set(G.predecessors(c)) | set(G.successors(c))) if is_entity(n)]
    if not ents:
        continue

    for a in strong:
        alias_global_freq[a] += 1
        for e in ents:
            alias_rows.append({"alias": a, "entity": e})

# Frequency gating (lenient)
MIN_OCCURS = 1      # allow singletons if they passed structure/context
MAX_SHARE = 0.80    # only drop truly ubiquitous surfaces

keep_aliases = set()
for a, f in alias_global_freq.items():
    if (a in KNOWN_ALIASES) or (f >= MIN_OCCURS and (f / max(1, total_comm_texts)) <= MAX_SHARE):
        keep_aliases.add(a)

alias_rows = [r for r in alias_rows if r["alias"] in keep_aliases]
df_alias = pd.DataFrame(alias_rows) if alias_rows else pd.DataFrame(columns=["alias","entity"])

print("Known alias hits (case-insensitive):", dict(known_hits))
print("Alias (robust) — unique kept:", 0 if df_alias.empty else df_alias["alias"].nunique())
if not df_alias.empty:
    preview = (df_alias.groupby("alias")["entity"]
               .count().sort_values(ascending=False).head(20))
    print("Top aliases (robust):", list(preview.items()))
else:
    print("No aliases kept. If known alias hits show >0 above, they are in the text; check filters.")

# ====================== Dash app ======================
app = dash.Dash(__name__, title="Oceanus Explorer")
server = app.server

# Controls
entity_options = ["All"] + list(df["entity"].value_counts().head(TOP_ENTITIES).index)
community_options = ["All"] + sorted({comm_label.get(e,"Unassigned") for e in G_proj.nodes()})
edge_mode_options = [
    {"label":"Both", "value":"both"},
    {"label":"Events only", "value":"event"},
    {"label":"Relationships only", "value":"rel"}
]

app.layout = html.Div([
    html.H2("Oceanus Interactive Explorer"),

    dcc.Tabs([
        dcc.Tab(label="Temporal Explorer", children=[
            html.Div([
                html.Div([
                    html.Label("Entity"),
                    dcc.Dropdown(id="ent-dd", options=[{"label":e, "value":e} for e in entity_options],
                                 value="All", clearable=False),
                ], style={"width":"24%","display":"inline-block","verticalAlign":"top"}),
                html.Div([
                    html.Label("Community"),
                    dcc.Dropdown(id="comm-dd", options=[{"label":c, "value":c} for c in community_options],
                                 value="All", clearable=False),
                ], style={"width":"24%","display":"inline-block","marginLeft":"10px","verticalAlign":"top"}),
                html.Div([
                    html.Label("Hour range"),
                    dcc.RangeSlider(id="hour-slider", min=0, max=23, step=1, value=[0,23],
                                    marks={0:"0",6:"6",12:"12",18:"18",23:"23"}),
                ], style={"width":"48%","display":"inline-block","marginLeft":"10px","verticalAlign":"top"}),
            ], style={"marginBottom":"8px"}),

            html.Div([
                html.Div([
                    html.Label("Day range"),
                    dcc.RangeSlider(id="day-slider",
                                    min=int(df["day"].min()), max=int(df["day"].max()),
                                    value=[int(df["day"].min()), int(df["day"].max())],
                                    marks=None, tooltip={"placement":"bottom","always_visible":True}),
                ], style={"width":"48%","display":"inline-block"}),
                html.Div([
                    dcc.Checklist(id="normalize-ck", options=[{"label":" Normalize by day", "value":"norm"}], value=[]),
                    dcc.Checklist(id="pseudo-ck", options=[{"label":" Show pseudonym overlay (Boss/Lookout/etc.)", "value":"pseudo"}], value=[]),
                ], style={"width":"48%","display":"inline-block","marginLeft":"10px"}),
            ], style={"marginBottom":"8px"}),

            dcc.Graph(id="temporal-heat", style={"height":"520px"}),
            dcc.Graph(id="temporal-hour-bar", style={"height":"280px"}),
        ]),

        dcc.Tab(label="Network Projection", children=[
            html.Div([
                html.Div([
                    html.Label("Edge mode"),
                    dcc.RadioItems(id="edge-mode", options=edge_mode_options, value="both", inline=True),
                    html.Br(),
                    html.Label("Min edge weight"),
                    dcc.Slider(id="min-weight", min=0, max=10, step=1, value=0, marks={0:"0",2:"2",5:"5",10:"10"}),
                ], style={"width":"49%","display":"inline-block","verticalAlign":"top"}),

                html.Div([
                    html.Label("Min degree"),
                    dcc.Slider(id="min-degree", min=0, max=10, step=1, value=0, marks={0:"0",3:"3",5:"5",10:"10"}),
                    html.Br(),
                    html.Label("Highlight entity (optional)"),
                    dcc.Dropdown(id="hi-entity", options=[{"label":e, "value":e} for e in entity_options if e!="All"], value=None),
                ], style={"width":"49%","display":"inline-block","verticalAlign":"top","marginLeft":"10px"}),
            ], style={"marginBottom":"8px"}),

            # ===== Network Cohesion Panel =====
            html.Div([
                html.Div(id="cohesion-panel",
                         style={"padding":"8px 12px","background":"#f7f7f9","border":"1px solid #e2e2e2",
                                "borderRadius":"10px","marginBottom":"8px","fontSize":"13px"}),
            ]),

            dcc.Graph(id="proj-graph", style={"height":"640px"}),

            html.Div([
                html.Div([
                    html.H4("Top Betweenness (current filter)"),
                    dash_table.DataTable(id="bet-table",
                                         columns=[{"name":"Entity","id":"entity"},{"name":"Betweenness","id":"bet"}],
                                         style_table={"height":"260px","overflowY":"auto"},
                                         style_cell={"padding":"4px","fontSize":"12px"}, page_size=10)
                ], style={"width":"49%","display":"inline-block"}),
                html.Div([
                    html.H4("Top Degree (current filter)"),
                    dash_table.DataTable(id="deg-table",
                                         columns=[{"name":"Entity","id":"entity"},{"name":"Degree","id":"deg"}],
                                         style_table={"height":"260px","overflowY":"auto"},
                                         style_cell={"padding":"4px","fontSize":"12px"}, page_size=10)
                ], style={"width":"49%","display":"inline-block","marginLeft":"10px"}),
            ]),
        ]),

        dcc.Tab(label="Aliases", children=[
            html.Div([
                html.Div([
                    html.Label("Top aliases"),
                    dcc.Slider(id="top-aliases", min=10, max=80, step=5, value=TOP_ALIASES, marks={10:"10",40:"40",80:"80"}),
                ], style={"width":"49%","display":"inline-block"}),
                html.Div([
                    html.Label("Min co-occurrence"),
                    dcc.Slider(id="min-co", min=1, max=5, step=1, value=2, marks={1:"1",2:"2",3:"3",5:"5"}),
                ], style={"width":"49%","display":"inline-block","marginLeft":"10px"}),
            ], style={"marginBottom":"8px"}),

            dcc.Graph(id="alias-bipartite", style={"height":"640px"}),

            html.Div(id="alias-stats", style={"fontSize":"12px","marginTop":"8px"})
        ])
    ])
])

# ====================== Helpers ======================
def filter_entities_by_community(entities, comm_name):
    if comm_name=="All":
        return entities
    return [e for e in entities if comm_label.get(e,"Unassigned")==comm_name]

def mat_from_df(sub, day_min, day_max, hour_min, hour_max, normalize=False):
    all_days = list(range(day_min, day_max+1))
    p = sub.pivot_table(index="day", columns="hour", values="time", aggfunc="count").fillna(0)
    p = p.reindex(index=all_days, columns=list(range(24)), fill_value=0)
    p = p.loc[all_days, list(range(hour_min, hour_max+1))]
    if normalize and len(p.index)>0:
        row_sums = p.sum(axis=1).replace(0,1)
        p = (p.T / row_sums).T
    return p

def build_projection(edge_mode, min_w, min_deg):
    H = nx.Graph()
    H.add_nodes_from(G_proj.nodes(data=True))
    for u,v,d in G_proj.edges(data=True):
        w = 0
        if edge_mode in ("both","event"): w += d.get("event_count",0)
        if edge_mode in ("both","rel"):   w += d.get("rel_count",0)
        if w >= min_w:
            H.add_edge(u,v, weight=w, event_count=d.get("event_count",0), rel_count=d.get("rel_count",0))
    if min_deg>0:
        keep = [n for n in H if H.degree(n)>=min_deg]
        H = H.subgraph(keep).copy()
    return H

def compute_cohesion_and_bridges(H):
    """Return (within, cross, pct_cross, top_bridges_list) for current filtered graph H."""
    within = cross = 0
    for u, v in H.edges():
        cu = comm_label.get(u, None)
        cv = comm_label.get(v, None)
        if cu is not None and cv is not None and cu == cv:
            within += 1
        else:
            cross += 1
    total = within + cross
    pct_cross = (100.0 * cross / total) if total else 0.0
    bet = nx.betweenness_centrality(H) if H.number_of_edges() > 0 else {}
    top = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:5]
    return within, cross, pct_cross, top

# ====================== Temporal callbacks ======================
@app.callback(
    [Output("temporal-heat","figure"),
     Output("temporal-hour-bar","figure")],
    [Input("ent-dd","value"),
     Input("comm-dd","value"),
     Input("hour-slider","value"),
     Input("day-slider","value"),
     Input("normalize-ck","value"),
     Input("pseudo-ck","value")]
)
def update_temporal(entity, community, hour_range, day_range, norm_vals, pseudo_vals):
    hour_min, hour_max = hour_range
    day_min, day_max   = day_range
    normalize = "norm" in (norm_vals or [])
    show_pseudo = "pseudo" in (pseudo_vals or [])

    sub = df.copy()
    if entity!="All":
        sub = sub[sub["entity"]==entity]
    if community!="All":
        valid = set(filter_entities_by_community(df["entity"].unique(), community))
        sub = sub[sub["entity"].isin(valid)]
    sub = sub[(sub["day"]>=day_min)&(sub["day"]<=day_max)&(sub["hour"]>=hour_min)&(sub["hour"]<=hour_max)]

    mat = mat_from_df(sub, day_min, day_max, hour_min, hour_max, normalize=normalize)
    zmax = max(1.0, float(mat.values.max())) if not normalize else 1.0

    fig1 = go.Figure(data=[go.Heatmap(
        z=mat.values, x=mat.columns, y=mat.index, colorscale="YlGnBu",
        zmin=0, zmax=zmax, colorbar=dict(title="Message Count" if not normalize else "Rel. Intensity")
    )])
    fig1.update_layout(title=f"Daily Heatmap — {entity} / {community}",
                       margin=dict(l=40,r=20,t=50,b=40),
                       xaxis_title="Hour", yaxis_title="Day (from first observed)")

    if show_pseudo:
        pseudo_points = []
        for c in comm_nodes:
            dnode = G.nodes[c]
            t = get_ts(dnode)
            if t is None: continue
            dy = int((t.normalize()-min_date).days); hr = t.hour
            if dy<day_min or dy>day_max or hr<hour_min or hr>hour_max: continue
            txt = (dnode.get("content") or dnode.get("message") or dnode.get("text") or "").lower()
            if any(a in txt for a in ["boss","lookout","middleman","accountant","small fry"]):
                pseudo_points.append((hr,dy))
        if pseudo_points:
            xs = [x for x,_ in pseudo_points]; ys = [y for _,y in pseudo_points]
            fig1.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                                      marker=dict(color="red", size=6, symbol="x", opacity=0.6),
                                      name="Pseudonym msgs"))

    hour_totals = mat.sum(axis=0)
    fig2 = go.Figure(data=[go.Bar(x=list(hour_totals.index), y=list(hour_totals.values))])
    fig2.update_layout(title="Hourly totals (current selection)",
                       xaxis_title="Hour", yaxis_title="Messages",
                       margin=dict(l=40,r=20,t=50,b=40))
    return fig1, fig2

# ====================== Network projection callbacks ======================
@app.callback(
    [Output("proj-graph","figure"),
     Output("bet-table","data"),
     Output("deg-table","data"),
     Output("cohesion-panel","children")],
    [Input("edge-mode","value"),
     Input("min-weight","value"),
     Input("min-degree","value"),
     Input("hi-entity","value")]
)
def update_projection(edge_mode, min_w, min_degree, hi_ent):
    H = build_projection(edge_mode, min_w, min_degree)
    if H.number_of_nodes()==0:
        panel = html.Div([html.B("Network Cohesion"), html.Br(),
                          html.Span("No nodes under current filters.")])
        return go.Figure(), [], [], panel

    # Cohesion panel stats
    within, cross, pct_cross, top = compute_cohesion_and_bridges(H)
    pct_within = 100.0 - pct_cross
    bridges_list = ", ".join([f"{n} ({v:.3f})" for n, v in top]) if top else "—"
    panel = html.Div([
        html.B("Network Cohesion"), html.Br(),
        html.Span(f"Within-community edges: {within} ({pct_within:.1f}%)  •  "
                  f"Cross-community edges: {cross} ({pct_cross:.1f}%)"), html.Br(),
        html.Span("Top bridge entities (betweenness): "),
        html.Span(bridges_list, style={"fontFamily":"monospace"})
    ])

    # Layout & metrics
    pos = nx.spring_layout(H, seed=42)
    deg = dict(H.degree())
    bet = nx.betweenness_centrality(H)

    palette = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ab']
    colors, sizes = [], []
    for n in H.nodes():
        c = comm_label.get(n, "Unassigned")
        idx = int(c.split()[-1]) if c.startswith("Community") else 0
        colors.append(palette[(idx-1)%len(palette)] if idx else "#888888")
        sizes.append(6 + 3*deg.get(n,0))

    edge_x, edge_y = [], []
    for u,v in H.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode="lines",
                             line=dict(width=0.5,color="#B0B0B0"),
                             hoverinfo="skip", name="edges"))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in H], y=[pos[n][1] for n in H],
        mode="markers+text",
        text=[n if (hi_ent and n==hi_ent) else "" for n in H],
        textposition="top center",
        marker=dict(size=sizes, color=colors, line=dict(width=0.5,color="#333"), opacity=0.95),
        hovertext=[f"{n}<br>deg:{deg.get(n,0)} bet:{bet.get(n,0):.3f}" for n in H],
        hoverinfo="text",
        name="entities"
    ))
    if hi_ent and hi_ent in H:
        fig.add_trace(go.Scatter(
            x=[pos[hi_ent][0]], y=[pos[hi_ent][1]], mode="markers+text",
            text=[hi_ent], textposition="top center",
            marker=dict(size=18, color="red", symbol="star", line=dict(width=1,color="#222")),
            name="highlight"
        ))

    fig.update_layout(
        title=f"Entity–Entity Projection (mode={edge_mode}, min_w={min_w}, min_deg={min_degree})",
        showlegend=False, margin=dict(l=20,r=20,t=50,b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest"
    )

    bet_rows = sorted(bet.items(), key=lambda x:x[1], reverse=True)[:10]
    deg_rows = sorted(deg.items(), key=lambda x:x[1], reverse=True)[:10]
    bet_tbl = [{"entity":k, "bet":f"{v:.3f}"} for k,v in bet_rows]
    deg_tbl = [{"entity":k, "deg":int(v)} for k,v in deg_rows]
    return fig, bet_tbl, deg_tbl, panel

# ====================== Alias bipartite callbacks ======================
@app.callback(
    [Output("alias-bipartite","figure"),
     Output("alias-stats","children")],
    [Input("top-aliases","value"),
     Input("min-co","value")]
)
def update_alias(top_aliases, min_co):
    if df_alias.empty:
        return go.Figure(), "No alias evidence found."

    counts = df_alias.groupby("alias")["entity"].count().sort_values(ascending=False)
    keep = set(counts.head(top_aliases).index)

    pairs = df_alias[df_alias["alias"].isin(keep)].groupby(["alias","entity"]).size().reset_index(name="w")
    pairs = pairs[pairs["w"]>=min_co]

    if pairs.empty:
        return go.Figure(), "No alias–entity pairs above threshold."

    B = nx.Graph()
    for _,r in pairs.iterrows():
        B.add_node(("alias", r["alias"]), bipartite=0)
        B.add_node(("entity", r["entity"]), bipartite=1)
        B.add_edge(("alias", r["alias"]), ("entity", r["entity"]), weight=int(r["w"]))
    pos = nx.spring_layout(B, seed=42)

    edge_x, edge_y = [], []
    for u,v,d in B.edges(data=True):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode="lines",
                             line=dict(width=1,color="#CCC"), hoverinfo="skip"))
    nodes = list(B.nodes())
    xs = [pos[n][0] for n in nodes]; ys = [pos[n][1] for n in nodes]
    texts = [n[1] for n in nodes]
    cols = ["#F28E2B" if n[0]=="alias" else "#4E79A7" for n in nodes]
    sizes = [12 + 2*len(n[1]) for n in nodes]
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="markers+text",text=texts,textposition="middle right",
                             marker=dict(size=sizes,color=cols,line=dict(width=0.5,color="#333"),opacity=0.95)))
    fig.update_layout(title=f"Alias ↔ Entity bipartite (top {top_aliases}, min co={min_co})",
                      showlegend=False, margin=dict(l=20,r=20,t=50,b=20),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    shared_handle = pairs.groupby("alias")["entity"].nunique()
    multi_alias  = pairs.groupby("entity")["alias"].nunique()
    msg = [
        html.Div(f"Handles used by multiple entities: {(shared_handle>1).sum()}"),
        html.Div(f"Entities using multiple handles: {(multi_alias>1).sum()}"),
        html.Br(),
        html.Div("Top shared aliases:"),
        html.Ul([html.Li(f"{a}: {k} entities") for a,k in shared_handle.sort_values(ascending=False).head(10).items()])
    ]
    return fig, msg

# ====================== Run ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Open http://127.0.0.1:{port} in your browser")
    app.run(debug=False, port=port)
