import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Page Config ──
st.set_page_config(
    page_title="Open Source Ecosystem Analytics",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Open Source Ecosystem Analytics Platform")
st.markdown("**Dataset:** GitHub Archive — January 1, 2025 (3PM–5PM UTC)")

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

# ── Metric Cards ──
st.markdown("---")
st.subheader("📊 Ecosystem Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events",        f"{len(df):,}")
col2.metric("Unique Repositories", f"{df['repo_name'].nunique():,}")
col3.metric("Unique Contributors", f"{df['actor'].nunique():,}")
col4.metric("Total Commits",       f"{int(df['commit_count'].sum()):,}")

# ── Hourly Trend ──
st.markdown("---")
st.subheader("📈 Hourly Activity Trend")

hourly = (
    df.groupby("hour")
      .agg(
          total_events       = ("event_type", "count"),
          unique_contributors= ("actor",      "nunique"),
          active_repos       = ("repo_name",  "nunique"),
          total_commits      = ("commit_count","sum")
      )
      .reset_index()
      .sort_values("hour")
)

fig1, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(hourly["hour"], hourly["total_events"],
             marker="o", color="steelblue", linewidth=2, label="Total Events")
axes[0].plot(hourly["hour"], hourly["unique_contributors"],
             marker="s", color="coral", linewidth=2, label="Contributors")
axes[0].set_title("Events & Contributors per Hour")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].tick_params(axis='x', rotation=15)

axes[1].plot(hourly["hour"], hourly["total_commits"],
             marker="^", color="mediumseagreen", linewidth=2, label="Total Commits")
axes[1].plot(hourly["hour"], hourly["active_repos"],
             marker="D", color="mediumpurple", linewidth=2, label="Active Repos")
axes[1].set_title("Commits & Active Repos per Hour")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Count")
axes[1].legend()
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
st.pyplot(fig1)

# ── Top 10 Healthiest Repos ──
st.markdown("---")
st.subheader("🏥 Top 10 Healthiest Repositories")

top10 = repo_df.head(10)

fig2, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top10["repo_name"][::-1], top10["health_score"][::-1],
               color="steelblue", edgecolor="white")
for bar, score in zip(bars, top10["health_score"][::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{score}", va="center", fontsize=9)
ax.set_xlabel("Ecosystem Health Score (0–100)")
ax.set_title("Top 10 Healthiest Open Source Repositories")
plt.tight_layout()
st.pyplot(fig2)

# Show table below chart
st.dataframe(
    top10[["repo_name","health_score","total_commits",
           "pull_requests","unique_contributors","stars","forks"]],
    use_container_width=True
)

# ── Event Type Distribution ──
st.markdown("---")
st.subheader("🥧 Activity Breakdown by Event Type")

event_counts = df["event_type"].value_counts().reset_index()
event_counts.columns = ["event_type", "count"]

fig3, ax = plt.subplots(figsize=(10, 7))
ax.pie(
    event_counts["count"],
    labels=event_counts["event_type"],
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.75,
    labeldistance=1.15
)
ax.set_title("Open Source Activity Types Distribution")
plt.tight_layout()
st.pyplot(fig3)

# ── Org vs Individual ──
st.markdown("---")
st.subheader("🏢 Organisation vs Individual Contributions")

org_data = (
    df.groupby("is_org")
      .agg(events=("event_type","count"),
           contributors=("actor","nunique"))
      .reset_index()
)
org_data["label"] = org_data["is_org"].map({True: "Organisation", False: "Individual"})

fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(org_data["label"], org_data["events"],
            color=["steelblue","coral"], edgecolor="white")
axes[0].set_title("Events: Org vs Individual")
axes[0].set_ylabel("Total Events")

axes[1].bar(org_data["label"], org_data["contributors"],
            color=["mediumseagreen","mediumpurple"], edgecolor="white")
axes[1].set_title("Contributors: Org vs Individual")
axes[1].set_ylabel("Unique Contributors")

plt.suptitle("Organisation vs Individual Contribution Patterns", fontsize=13)
plt.tight_layout()
st.pyplot(fig4)

# ── Stars vs Forks Scatter ──
st.markdown("---")
st.subheader("⭐ Popularity vs Adoption — Stars vs Forks")

scatter_df = repo_df[(repo_df["stars"] > 0) | (repo_df["forks"] > 0)]

fig5, ax = plt.subplots(figsize=(10, 6))
ax.scatter(scatter_df["stars"], scatter_df["forks"],
           alpha=0.6, color="coral", s=60)
ax.set_xlabel("⭐ Stars (WatchEvents)")
ax.set_ylabel("🍴 Forks")
ax.set_title("Popularity vs Adoption")
plt.tight_layout()
st.pyplot(fig5)

# ── Search Box ──
st.markdown("---")
st.subheader("🔍 Search Any Repository")

search = st.text_input("Enter repo name (e.g. torvalds/linux)")
if search:
    result = repo_df[repo_df["repo_name"].str.contains(search, case=False, na=False)]
    if len(result) > 0:
        st.success(f"Found {len(result)} matching repositories")
        st.dataframe(
            result[["repo_name","health_score","total_commits",
                    "pull_requests","unique_contributors","stars","forks"]],
            use_container_width=True
        )
    else:
        st.warning("No repositories found. Try a different name.")

# ── Download CSV ──
st.markdown("---")
st.subheader("⬇️ Download Full Results")

csv = repo_df.to_csv(index=True)
st.download_button(
    label="Download ecosystem_health_scores.csv",
    data=csv,
    file_name="ecosystem_health_scores.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Data Source: GitHub Archive (gharchive.org) | January 1, 2025 | Built with Streamlit")
