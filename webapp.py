import streamlit as st
st.write("✅ App loaded")
import re
import io
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from openai import OpenAI

def generate_ai_insights(kpis, mom_text, top_channel, top_product):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = f"""
You are a senior business analyst.

Analyze the sales data below and write clear business insights.

KPIs:
- Total Sales: {kpis.get("total_sales")}
- Total Orders: {kpis.get("total_orders")}
- Total Profit: {kpis.get("total_profit")}

Month on Month Summary:
{mom_text}

Top Channel Driver:
{top_channel}

Top Product Driver:
{top_product}

Tasks:
1. Explain what happened this month.
2. Explain likely reasons (hypotheses only).
3. Give 3 actionable recommendations.

Tone: professional, simple, client-ready.
Format: bullet points.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content



# ---------------------------
# Helpers: column cleaning
# ---------------------------
def normalize_colname(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^\w\s]", "", c)
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"_+", "_", c)
    return c.strip("_")


def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[,\s]", "", regex=True)
    s = s.str.replace(r"^(rs|pkr|usd)\.?", "", regex=True, case=False)
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def try_parse_datetime(series: pd.Series) -> pd.Series:
    s = series.copy()
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
    if dt.notna().mean() < 0.6:
        dt2 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=False)
        if dt2.notna().mean() > dt.notna().mean():
            dt = dt2
    return dt


def unique_ratio(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    return s.nunique() / len(s)


# ---------------------------
# Column detection
# ---------------------------
def detect_columns(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()
    colmap = {
        "age": None,
        "id": None,
        "date": None,
        "gender": None,
        "sales": None,
        "profit": None,
        "qty": None,
        "product": None,
        "channel": None,
        "order_id": None,
    }

    def name_has(col, patterns):
        return any(p in col for p in patterns)

    for c in cols:
        cl = str(c).lower()

        if colmap["age"] is None and name_has(cl, ["age"]):
            colmap["age"] = c
        if colmap["gender"] is None and name_has(cl, ["gender", "sex"]):
            colmap["gender"] = c
        if colmap["date"] is None and name_has(cl, ["date", "datetime", "order_date", "invoice_date"]):
            colmap["date"] = c
        if colmap["sales"] is None and name_has(cl, ["sales", "revenue", "amount", "net_sales", "total_sales", "sale"]):
            colmap["sales"] = c
        if colmap["profit"] is None and name_has(cl, ["profit", "margin"]):
            colmap["profit"] = c
        if colmap["qty"] is None and name_has(cl, ["qty", "quantity", "units", "items_sold"]):
            colmap["qty"] = c
        if colmap["product"] is None and name_has(cl, ["product", "sku", "item", "product_name"]):
            colmap["product"] = c
        if colmap["channel"] is None and name_has(cl, ["channel", "source", "platform", "store", "medium"]):
            colmap["channel"] = c

        if colmap["order_id"] is None and name_has(cl, ["order_id", "invoice_id", "receipt_id", "transaction_id"]):
            colmap["order_id"] = c

        if colmap["id"] is None and (cl.endswith("_id") or cl == "id" or "client_id" in cl or "customer_id" in cl):
            colmap["id"] = c

    # Age fallback (0-120 numeric)
    if colmap["age"] is None:
        for c in cols:
            s = coerce_numeric(df[c])
            if s.notna().mean() > 0.8:
                valid = s.dropna()
                if len(valid) > 0 and (valid.between(0, 120).mean() > 0.9):
                    colmap["age"] = c
                    break

    # Date fallback
    if colmap["date"] is None:
        best = (None, 0.0)
        for c in cols:
            dt = try_parse_datetime(df[c])
            score = dt.notna().mean()
            if score > best[1] and score > 0.7:
                best = (c, score)
        colmap["date"] = best[0]

    # Sales fallback: numeric-ish, larger magnitude
    if colmap["sales"] is None:
        best = (None, 0.0)
        for c in cols:
            s = coerce_numeric(df[c])
            score = s.notna().mean()
            if score > 0.8:
                median = s.dropna().median() if s.notna().any() else 0
                mag = float(median) if pd.notna(median) else 0
                rank = score * (1 + min(abs(mag), 1e6) / 1e6)
                if rank > best[1]:
                    best = (c, rank)
        colmap["sales"] = best[0]

    # Order id fallback: high uniqueness
    if colmap["order_id"] is None:
        best = (None, 0.0)
        for c in cols:
            ur = unique_ratio(df[c])
            if ur > best[1] and ur > 0.85:
                best = (c, ur)
        colmap["order_id"] = best[0]

    # Generic id fallback: high uniqueness
    if colmap["id"] is None:
        best = (None, 0.0)
        for c in cols:
            ur = unique_ratio(df[c])
            if ur > best[1] and ur > 0.85:
                best = (c, ur)
        colmap["id"] = best[0]

    return colmap


# ---------------------------
# Age grouping
# ---------------------------
def make_age_groups(age_series: pd.Series, method="default", n_quantiles=4) -> pd.Series:
    s = coerce_numeric(age_series)
    s = s.where(s.between(0, 120), np.nan)

    if method == "quantile":
        valid = s.dropna()
        if len(valid) >= 10:
            try:
                return pd.qcut(s, q=n_quantiles, duplicates="drop").astype(str)
            except Exception:
                pass

    bins = [-math.inf, 17, 24, 34, 44, 54, 64, math.inf]
    labels = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(s, bins=bins, labels=labels)


# ---------------------------
# Cleaning pipeline
# ---------------------------
def clean_dataset(df: pd.DataFrame, colmap: dict) -> tuple[pd.DataFrame, dict, dict]:
    report = {"rows_before": len(df), "cols_before": df.shape[1], "actions": []}

    original_cols = df.columns.tolist()
    new_cols = [normalize_colname(c) for c in original_cols]
    rename_map = dict(zip(original_cols, new_cols))
    df = df.rename(columns=rename_map)

    def norm_col(c):
        return rename_map.get(c, c) if c else None

    colmap = {k: norm_col(v) for k, v in colmap.items()}

    report["actions"].append("Standardized column names (lowercase + underscores).")

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        report["actions"].append(f"Dropped {len(empty_cols)} fully-empty columns.")

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"": np.nan, "nan": np.nan, "null": np.nan, "none": np.nan, "n/a": np.nan, "na": np.nan})

    for key in ["sales", "profit", "qty", "age"]:
        c = colmap.get(key)
        if c and c in df.columns:
            df[c] = coerce_numeric(df[c])

    if colmap.get("date") and colmap["date"] in df.columns:
        df[colmap["date"]] = try_parse_datetime(df[colmap["date"]])

    # Duplicates
    dup_removed = 0
    if colmap.get("order_id") and colmap["order_id"] in df.columns:
        before = len(df)
        if colmap.get("date") and colmap["date"] in df.columns:
            df = df.sort_values(colmap["date"]).drop_duplicates(subset=[colmap["order_id"]], keep="last")
        else:
            df = df.drop_duplicates(subset=[colmap["order_id"]], keep="last")
        dup_removed = before - len(df)
        if dup_removed > 0:
            report["actions"].append(f"Removed {dup_removed} duplicates based on '{colmap['order_id']}'.")
    else:
        before = len(df)
        df = df.drop_duplicates(keep="first")
        dup_removed = before - len(df)
        if dup_removed > 0:
            report["actions"].append(f"Removed {dup_removed} exact duplicate rows.")

    # Derive month
    if colmap.get("date") and colmap["date"] in df.columns:
        df["month"] = df[colmap["date"]].dt.to_period("M").astype(str)
        report["actions"].append("Created 'month' from date column.")
    else:
        df["month"] = np.nan

    # Derive age_group
    if colmap.get("age") and colmap["age"] in df.columns:
        df["age_group"] = make_age_groups(df[colmap["age"]], method="default")
        report["actions"].append("Created 'age_group' from age column.")
    else:
        df["age_group"] = np.nan

    report["rows_after"] = len(df)
    report["cols_after"] = df.shape[1]
    return df, report, colmap


# ---------------------------
# KPIs + Insights
# ---------------------------
def compute_kpis(df: pd.DataFrame, colmap: dict) -> dict:
    sales_col = colmap.get("sales")
    profit_col = colmap.get("profit")
    qty_col = colmap.get("qty")
    product_col = colmap.get("product")
    order_col = colmap.get("order_id") or colmap.get("id")

    kpis = {}
    kpis["total_sales"] = float(df[sales_col].sum()) if sales_col and sales_col in df.columns else None
    kpis["total_orders"] = int(df[order_col].nunique(dropna=True)) if order_col and order_col in df.columns else int(len(df))
    kpis["total_profit"] = float(df[profit_col].sum()) if profit_col and profit_col in df.columns else None
    kpis["total_items_sold"] = float(df[qty_col].sum()) if qty_col and qty_col in df.columns else None
    kpis["total_products"] = int(df[product_col].nunique(dropna=True)) if product_col and product_col in df.columns else None
    return kpis


def build_monthly_table(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    sales_col = colmap.get("sales")
    order_col = colmap.get("order_id") or colmap.get("id")
    if not sales_col or sales_col not in df.columns:
        return pd.DataFrame()

    if order_col and order_col in df.columns:
        g = df.groupby("month", dropna=False).agg(
            sales=(sales_col, "sum"),
            orders=(order_col, "nunique"),
        ).reset_index()
    else:
        g = df.groupby("month", dropna=False).agg(
            sales=(sales_col, "sum"),
            orders=(sales_col, "size"),
        ).reset_index()

    g = g[g["month"].notna()].sort_values("month")
    return g


def generate_insights(df: pd.DataFrame, colmap: dict) -> str:
    sales_col = colmap.get("sales")
    channel_col = colmap.get("channel")
    product_col = colmap.get("product")

    monthly = build_monthly_table(df, colmap)
    if monthly.empty or len(monthly) < 2:
        return "Not enough monthly data to generate MoM insights (need at least 2 months with sales)."

    last = monthly.iloc[-1]
    prev = monthly.iloc[-2]

    sales_change = last["sales"] - prev["sales"]
    sales_pct = (sales_change / prev["sales"] * 100) if prev["sales"] != 0 else np.nan

    orders_change = last["orders"] - prev["orders"]
    orders_pct = (orders_change / prev["orders"] * 100) if prev["orders"] != 0 else np.nan

    msg = []
    msg.append(f"Latest month: {last['month']} vs previous: {prev['month']}.")
    msg.append(f"Sales changed by {sales_change:,.2f} ({sales_pct:,.1f}%).")
    msg.append(f"Orders changed by {orders_change:,.0f} ({orders_pct:,.1f}%).")

    # AOV reasoning
    if last["orders"] != 0 and prev["orders"] != 0:
        aov_last = last["sales"] / last["orders"]
        aov_prev = prev["sales"] / prev["orders"]
        aov_change = aov_last - aov_prev
        aov_pct = (aov_change / aov_prev * 100) if aov_prev != 0 else np.nan

        if sales_change > 0 and orders_change > 0:
            msg.append("Reason hint: Sales and orders both up — demand/transactions increased.")
        elif sales_change > 0 and orders_change <= 0:
            msg.append("Reason hint: Sales up but orders not up — AOV increased (premium mix / bigger baskets).")
        elif sales_change < 0 and orders_change > 0:
            msg.append("Reason hint: Orders up but sales down — AOV decreased (discounting / cheaper mix).")
        else:
            msg.append("Reason hint: Both sales and orders fell — demand slowdown or supply/availability issues.")

        msg.append(f"AOV changed by {aov_change:,.2f} ({aov_pct:,.1f}%).")

    # Channel driver
    if channel_col and channel_col in df.columns and sales_col and sales_col in df.columns:
        cur_month = last["month"]
        prev_month = prev["month"]
        cur = df[df["month"] == cur_month].groupby(channel_col)[sales_col].sum()
        prv = df[df["month"] == prev_month].groupby(channel_col)[sales_col].sum()
        delta = (cur - prv).sort_values(ascending=False)
        if len(delta) > 0:
            top_driver = delta.index[0]
            msg.append(f"Top channel driver (MoM): '{top_driver}' contributed {delta.iloc[0]:,.2f} sales change.")

    # Product driver
    if product_col and product_col in df.columns and sales_col and sales_col in df.columns:
        cur_month = last["month"]
        prev_month = prev["month"]
        cur = df[df["month"] == cur_month].groupby(product_col)[sales_col].sum()
        prv = df[df["month"] == prev_month].groupby(product_col)[sales_col].sum()
        delta = (cur - prv).sort_values(ascending=False)
        if len(delta) > 0:
            top_prod = delta.index[0]
            msg.append(f"Top product driver (MoM): '{top_prod}' contributed {delta.iloc[0]:,.2f} sales change.")

    return "\n".join(msg)


# ---------------------------
# Charts with DATA LABELS
# ---------------------------
def apply_common_label_styling(fig):
    # Prevent label cut-off
    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=40),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )
    return fig


def make_charts(df: pd.DataFrame, colmap: dict):
    charts = {}

    sales_col = colmap.get("sales")
    gender_col = colmap.get("gender")
    channel_col = colmap.get("channel")
    product_col = colmap.get("product")

    monthly = build_monthly_table(df, colmap)

    # 1) Sales vs Orders (Monthly combo) - labels on bars + line points
    if not monthly.empty:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=monthly["month"],
            y=monthly["orders"],
            name="Orders",
            text=monthly["orders"],
            textposition="outside",
            opacity=0.75
        ))

        fig.add_trace(go.Scatter(
            x=monthly["month"],
            y=monthly["sales"],
            name="Sales",
            mode="lines+markers+text",
            text=monthly["sales"].round(0),
            textposition="top center",
            yaxis="y2"
        ))

        fig.update_layout(
            title="Sales vs Orders (Monthly)",
            xaxis_title="Month",
            yaxis=dict(title="Orders"),
            yaxis2=dict(title="Sales", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            height=420
        )

        fig.update_traces(cliponaxis=False)
        charts["sales_vs_orders"] = apply_common_label_styling(fig)

        # 2) Sales vs Month (line) - labels on points
        fig2 = px.line(
            monthly, x="month", y="sales",
            title="Sales vs Month",
            markers=True
        )
        fig2.update_traces(
            mode="lines+markers+text",
            text=monthly["sales"].round(0),
            textposition="top center",
            texttemplate="%{text:,.0f}"
        )
        charts["sales_vs_month"] = apply_common_label_styling(fig2)

    # 3) Sales vs Gender (bar) - labels
    if sales_col and sales_col in df.columns and gender_col and gender_col in df.columns:
        g = df.groupby(gender_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False)
        fig = px.bar(g, x=gender_col, y=sales_col, title="Sales vs Gender", text_auto=True)
        fig.update_traces(textposition="outside", texttemplate="%{y:,.0f}")
        charts["sales_vs_gender"] = apply_common_label_styling(fig)

    # 4) Sales vs Age Group (bar) - labels
    if sales_col and sales_col in df.columns and "age_group" in df.columns:
        g = df.dropna(subset=["age_group"]).groupby("age_group")[sales_col].sum().reset_index()
        order = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        g["age_group"] = pd.Categorical(g["age_group"].astype(str), categories=order, ordered=True)
        g = g.sort_values("age_group")

        fig = px.bar(g, x="age_group", y=sales_col, title="Sales vs Age Group", text_auto=True)
        fig.update_traces(textposition="outside", texttemplate="%{y:,.0f}")
        charts["sales_vs_age_group"] = apply_common_label_styling(fig)

    # 5) Top 5 Products by Sales (horizontal bar) - labels
    if sales_col and sales_col in df.columns and product_col and product_col in df.columns:
        g = df.groupby(product_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False).head(5)
        fig = px.bar(
            g.sort_values(sales_col),
            x=sales_col,
            y=product_col,
            orientation="h",
            title="Top 5 Products by Sales",
            text_auto=True
        )
        fig.update_traces(textposition="outside", texttemplate="%{x:,.0f}")
        charts["top5_products"] = apply_common_label_styling(fig)

    # 6) Sales by Channel (bar) - labels
    if sales_col and sales_col in df.columns and channel_col and channel_col in df.columns:
        g = df.groupby(channel_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False)
        fig = px.bar(g, x=channel_col, y=sales_col, title="Sales by Channel", text_auto=True)
        fig.update_traces(textposition="outside", texttemplate="%{y:,.0f}")
        charts["sales_by_channel"] = apply_common_label_styling(fig)

    return charts


# ---------------------------
# PDF Export
# ---------------------------


def fig_to_png_bytes(fig) -> bytes:
    # Force kaleido renderer (no browser dependency)
    return pio.to_image(fig, format="png", scale=2, engine="kaleido")



def export_pdf(kpis: dict, insights: str, charts: dict) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    def draw_title(text, y):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, text)

    def draw_text_block(text, x, y, max_width=520, leading=14):
        c.setFont("Helvetica", 10)
        lines = []
        for para in str(text).split("\n"):
            words = para.split()
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                if c.stringWidth(test, "Helvetica", 10) <= max_width:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            lines.append("")
        yy = y
        for line in lines:
            c.drawString(x, yy, line)
            yy -= leading
            if yy < 60:
                c.showPage()
                yy = height - 60
                c.setFont("Helvetica", 10)
        return yy

    # Page 1: KPIs + Insights
    draw_title("Sales Dashboard Report", height - 50)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 90, "Key KPIs")

    c.setFont("Helvetica", 11)
    y = height - 115
    kpi_lines = [
        ("Total Sales (PKR)", kpis.get("total_sales")),
        ("Total Orders", kpis.get("total_orders")),
        ("Total Profit (PKR)", kpis.get("total_profit")),
        ("Total Items Sold", kpis.get("total_items_sold")),
        ("Total Products", kpis.get("total_products")),
    ]
    for name, val in kpi_lines:
        val_str = "N/A" if val is None else (f"{val:,.2f}" if isinstance(val, float) else f"{val:,}")
        c.drawString(50, y, f"{name}: {val_str}")
        y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 10, "Auto Insights Summary")
    y = y - 35
    y = draw_text_block(insights, 40, y)

    # Charts pages
    for key, fig in charts.items():
        c.showPage()
        title = fig.layout.title.text if fig.layout.title.text else key
        draw_title(str(title), height - 50)

        try:
            img_bytes = fig_to_png_bytes(fig)
            img = ImageReader(io.BytesIO(img_bytes))
            margin = 40
            img_w = width - 2 * margin
            img_h = height - 140
            c.drawImage(img, margin, 80, width=img_w, height=img_h, preserveAspectRatio=True, anchor="c")
        except Exception as e:
            c.setFont("Helvetica", 10)
            c.drawString(40, height - 90, f"Could not render chart '{key}'. Error: {str(e)}")

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Auto Sales Data Cleaner + Dashboard", layout="wide")
st.title("Auto Sales Data Cleaner + Dashboard")
st.caption("Upload → Auto clean → KPIs & charts (with data labels) → Download PDF report")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Upload a dataset to start.")
    st.stop()

# Load
try:
    if uploaded.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("1) Preview (Raw Data)")
st.dataframe(df_raw.head(30), use_container_width=True)

# Detect + Clean
colmap_detected = detect_columns(df_raw)

with st.expander("Detected Columns (Auto)", expanded=True):
    st.write(colmap_detected)

st.subheader("2) Cleaning Options")
age_method = st.selectbox("Age grouping method", ["default", "quantile"], index=0)
quantiles = st.slider("If quantile: number of groups", 3, 8, 4)

df_clean, clean_report, colmap = clean_dataset(df_raw, colmap_detected)

# Apply age grouping choice (override)
if colmap.get("age") and colmap["age"] in df_clean.columns:
    df_clean["age_group"] = make_age_groups(df_clean[colmap["age"]], method=age_method, n_quantiles=quantiles)

with st.expander("Cleaning Report", expanded=True):
    st.write(f"Rows: {clean_report['rows_before']} → {clean_report['rows_after']}")
    st.write(f"Columns: {clean_report['cols_before']} → {clean_report['cols_after']}")
    for a in clean_report["actions"]:
        st.markdown(f"- {a}")

st.subheader("3) Cleaned Data Preview")
st.dataframe(df_clean.head(30), use_container_width=True)

# KPIs
kpis = compute_kpis(df_clean, colmap)
st.subheader("4) KPIs")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Total Sales (PKR)",
    "N/A" if kpis["total_sales"] is None else f"{kpis['total_sales']:,.0f}"
)

k2.metric(
    "Total Orders",
    f"{kpis['total_orders']:,}"
)

k3.metric(
    "Total Profit (PKR)",
    "N/A" if kpis["total_profit"] is None else f"{kpis['total_profit']:,.0f}"
)

k4.metric(
    "Total Items Sold",
    "N/A" if kpis["total_items_sold"] is None else f"{kpis['total_items_sold']:,.0f}"
)

k5.metric(
    "Total Products",
    "N/A" if kpis["total_products"] is None else f"{kpis['total_products']:,}"
)


# Charts
st.subheader("5) Charts (with Data Labels)")
charts = make_charts(df_clean, colmap)

if not charts:
    st.warning("Could not create charts. Ensure dataset has a Sales column (and ideally Date/Gender/Product/Channel).")
else:
    if "sales_vs_orders" in charts:
        st.plotly_chart(charts["sales_vs_orders"], use_container_width=True)

    c1, c2 = st.columns(2)
    if "sales_vs_month" in charts:
        c1.plotly_chart(charts["sales_vs_month"], use_container_width=True)
    if "sales_vs_gender" in charts:
        c2.plotly_chart(charts["sales_vs_gender"], use_container_width=True)

    c3, c4 = st.columns(2)
    if "sales_vs_age_group" in charts:
        c3.plotly_chart(charts["sales_vs_age_group"], use_container_width=True)
    if "sales_by_channel" in charts:
        c4.plotly_chart(charts["sales_by_channel"], use_container_width=True)

    if "top5_products" in charts:
        st.plotly_chart(charts["top5_products"], use_container_width=True)

# Insights summary
st.subheader("6) Auto Insights Summary (Rule-based)")
insights_text = generate_insights(df_clean, colmap)
st.text(insights_text)
st.subheader("7) 🤖 AI Insights (ChatGPT)")

# Simple drivers text (safe defaults)
top_channel_driver = "N/A"
top_product_driver = "N/A"

try:
    # reuse your existing rule-based insights text as MoM summary input
    mom_summary_text = insights_text

    # Optional: if you already compute drivers somewhere, plug them here
    # Otherwise keep N/A (AI will still generate useful narrative)
except Exception:
    mom_summary_text = insights_text

# Add a button so AI call happens only when user wants (saves cost)
if st.button("Generate AI Insights"):
    with st.spinner("Analyzing with AI..."):
        try:
            ai_text = generate_ai_insights(
                kpis=kpis,
                mom_text=mom_summary_text,
                top_channel=top_channel_driver,
                top_product=top_product_driver
            )
            st.markdown(ai_text)
        except Exception as e:
            st.error(f"AI Insights failed: {e}")
else:
    st.info("Click 'Generate AI Insights' to get a written analysis & recommendations.")


# Downloads
st.subheader("7) Downloads")
out_csv = df_clean.to_csv(index=False).encode("utf-8")
st.download_button("Download Cleaned CSV", data=out_csv, file_name="cleaned_dataset.csv", mime="text/csv")

# PDF report
st.caption("PDF export needs kaleido installed: pip install kaleido")
try:
    pdf_bytes = export_pdf(kpis=kpis, insights=insights_text, charts=charts)
    st.download_button(
        "Download PDF Report (KPIs + Charts + Insights)",
        data=pdf_bytes,
        file_name="sales_dashboard_report.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.error(f"PDF export failed: {e}")
    st.info("Fix: pip install kaleido reportlab")



