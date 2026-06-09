import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ──
st.set_page_config(
    page_title="OSS Health Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Light Theme ──
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #7c3aed;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #7c3aed;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 4px;
    }
    .header-box {
        background: linear-gradient(135deg, #7c3aed, #0d9488);
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──
@st.cache_data
def load_data():
    rows = []
    with open("ecosystem_data_small.json", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            payload = event.get("payload", {})
            rows.append({
                "event_type"   : event.get("type", ""),
                "actor"        : event.get("actor", {}).get("login", ""),
                "repo_name"    : event.get("repo",  {}).get("name",  ""),
                "created_at"   : event.get("created_at", ""),
                "hour"         : event.get("created_at", "")[:13],
                "is_org"       : "org" in event,
                "commit_count" : len(payload.get("commits", [])),
            })
    return pd.DataFrame(rows)

@st.cache_data
def load_scores():
    return pd.read_csv("ecosystem_health_scores.csv", index_col=0)

df      = load_data()
repo_df = load_scores()

# ── Sidebar ──
st.sidebar.markdown("## 🔬 OSS Health Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.info("GitHub Archive\nJan 1, 2025\n3PM – 5PM UTC")
st.sidebar.markdown("**Libraries Used**")
st.sidebar.markdown("- Polars\n- Pandas\n- NumPy\n- Matplotlib\n- Streamlit")
st.sidebar.markdown("---")
top_n = st.sidebar.slider("Top N Repositories", 5, 20, 10)

# ── Header ──
st.markdown("""
<div class="header-box">
    <h1 style='margin:0; font-size:2rem;'>🔬 Open Source Ecosystem Health Dashboard</h1>
    <p style='margin:8px 0 0 0; opacity:0.85;'>Analyzing real GitHub activity — January 1, 2025</p>
</div>
""", unsafe_allow_html=True)

# ── Metric Cards ──
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Events</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{df['repo_name'].nunique():,}</div>
        <div class="metric-label">Unique Repositories</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{df['actor'].nunique():,}</div>
        <div class="metric-label">Unique Contributors</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{int(df['commit_count'].sum()):,}</div>
        <div class="metric-label">Total Commits</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Activity Trends",
    "🏥 Health Rankings",
    "🥧 Event Analysis",
    "🔍 Search & Export"
])

# ────────────────────────────────
# TAB 1 — Activity Trends
# ────────────────────────────────
with tab1:
    st.markdown("### 📈 Hourly Activity Trends")

    hourly = (
        df.groupby("hour")
          .agg(
              total_events        = ("event_type",    "count"),
              unique_contributors = ("actor",         "nunique"),
              active_repos        = ("repo_name",     "nunique"),
              total_commits       = ("commit_count",  "sum")
          )
          .reset_index()
          .sort_values("hour")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="#f8f9fa")

    # Left chart
    axes[0].set_facecolor("white")
    axes[0].plot(hourly["hour"], hourly["total_events"],
                 marker="o", color="#7c3aed", linewidth=2.5, label="Total Events")
    axes[0].plot(hourly["hour"], hourly["unique_contributors"],
                 marker="s", color="#0d9488", linewidth=2.5, label="Contributors")
    axes[0].fill_between(hourly["hour"], hourly["total_events"], alpha=0.1, color="#7c3aed")
    axes[0].set_title("Events & Contributors per Hour", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].spines[['top','right']].set_visible(False)

    # Right chart
    axes[1].set_facecolor("white")
    axes[1].plot(hourly["hour"], hourly["total_commits"],
                 marker="^", color="#f59e0b", linewidth=2.5, label="Total Commits")
    axes[1].plot(hourly["hour"], hourly["active_repos"],
                 marker="D", color="#ef4444", linewidth=2.5, label="Active Repos")
    axes[1].fill_between(hourly["hour"], hourly["total_commits"], alpha=0.1, color="#f59e0b")
    axes[1].set_title("Commits & Active Repos per Hour", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    # Org vs Individual
    st.markdown("### 🏢 Organisation vs Individual")
    org_data = (
        df.groupby("is_org")
          .agg(events=("event_type","count"), contributors=("actor","nunique"))
          .reset_index()
    )
    org_data["label"] = org_data["is_org"].map({True: "Organisation", False: "Individual"})

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4), facecolor="#f8f9fa")
    colors = ["#7c3aed", "#0d9488"]

    axes2[0].set_facecolor("white")
    axes2[0].bar(org_data["label"], org_data["events"], color=colors, width=0.4)
    axes2[0].set_title("Events by Type", fontsize=12, fontweight="bold")
    axes2[0].set_ylabel("Total Events")
    axes2[0].spines[['top','right']].set_visible(False)

    axes2[1].set_facecolor("white")
    axes2[1].bar(org_data["label"], org_data["contributors"], color=colors[::-1], width=0.4)
    axes2[1].set_title("Contributors by Type", fontsize=12, fontweight="bold")
    axes2[1].set_ylabel("Unique Contributors")
    axes2[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig2)

# ────────────────────────────────
# TAB 2 — Health Rankings
# ────────────────────────────────
with tab2:
    st.markdown(f"### 🏥 Top {top_n} Healthiest Repositories")

    top_n_df = repo_df.head(top_n)

    fig3, ax = plt.subplots(figsize=(12, top_n * 0.6 + 2), facecolor="#f8f9fa")
    ax.set_facecolor("white")

    colors_bar = plt.cm.RdYlGn(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(top_n_df["repo_name"][::-1],
                   top_n_df["health_score"][::-1],
                   color=colors_bar, edgecolor="white", height=0.6)

    for bar, score in zip(bars, top_n_df["health_score"][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{score:.1f}", va="center", fontsize=9, color="#374151")

    ax.set_xlabel("Health Score (0–100)", fontsize=11)
    ax.set_title(f"Top {top_n} Healthiest Open Source Repositories", fontsize=13, fontweight="bold")
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("#### 📋 Detailed Health Metrics")
    st.dataframe(
        top_n_df[["repo_name","health_score","total_commits",
                  "pull_requests","unique_contributors","stars","forks"]],
        use_container_width=True,
        hide_index=False
    )

    # Stars vs Forks
    st.markdown("### ⭐ Stars vs Forks")
    scatter_df = repo_df[(repo_df["stars"] > 0) | (repo_df["forks"] > 0)]

    fig4, ax4 = plt.subplots(figsize=(10, 5), facecolor="#f8f9fa")
    ax4.set_facecolor("white")
    ax4.scatter(scatter_df["stars"], scatter_df["forks"],
                alpha=0.7, color="#7c3aed", s=80, edgecolors="white", linewidths=0.5)
    ax4.set_xlabel("⭐ Stars (WatchEvents)", fontsize=11)
    ax4.set_ylabel("🍴 Forks", fontsize=11)
    ax4.set_title("Popularity vs Adoption", fontsize=13, fontweight="bold")
    ax4.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig4)

# ────────────────────────────────
# TAB 3 — Event Analysis
# ────────────────────────────────
with tab3:
    st.markdown("### 🥧 Event Type Distribution")

    event_counts = df["event_type"].value_counts().reset_index()
    event_counts.columns = ["event_type", "count"]

    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(event_counts)))

    fig5, ax5 = plt.subplots(figsize=(10, 7), facecolor="#f8f9fa")
    wedges, texts, autotexts = ax5.pie(
        event_counts["count"],
        labels=event_counts["event_type"],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        labeldistance=1.12,
        colors=colors_pie,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    for text in autotexts:
        text.set_fontsize(8)
    ax5.set_title("GitHub Event Types Breakdown", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig5)

    # Event counts table
    st.markdown("#### 📊 Event Count Table")
    st.dataframe(event_counts, use_container_width=True, hide_index=True)

# ────────────────────────────────
# TAB 4 — Search & Export
# ────────────────────────────────
with tab4:
    st.markdown("### 🔍 Search Repository")

    search = st.text_input("Enter repository name to search", placeholder="e.g. torvalds/linux")
    if search:
        result = repo_df[repo_df["repo_name"].str.contains(search, case=False, na=False)]
        if len(result) > 0:
            st.success(f"✅ Found {len(result)} matching repositories")
            st.dataframe(
                result[["repo_name","health_score","total_commits",
                        "pull_requests","unique_contributors","stars","forks"]],
                use_container_width=True
            )
        else:
            st.warning("❌ No repositories found. Try a different name.")

    st.markdown("---")
    st.markdown("### ⬇️ Download Full Results")
    csv = repo_df.to_csv(index=True)
    st.download_button(
        label="📥 Download ecosystem_health_scores.csv",
        data=csv,
        file_name="ecosystem_health_scores.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### 📋 Health Score Formula")
    st.code("""
Score = 0.25 × Total Commits
      + 0.25 × Pull Requests
      + 0.20 × Unique Contributors
      + 0.15 × Issue Engagement
      + 0.10 × Stars
      + 0.05 × Forks
    """)

st.markdown("---")
st.caption("Data Source: GitHub Archive (gharchive.org) | January 1, 2025 | Built with Streamlit")
