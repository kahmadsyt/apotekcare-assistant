
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for p in [str(ROOT_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from chatbot_engine import chatbot_response, load_metadata
except Exception as exc:
    chatbot_response = None
    load_metadata = None
    CHATBOT_IMPORT_ERROR = exc
else:
    CHATBOT_IMPORT_ERROR = None


st.set_page_config(
    page_title="ApotekCare Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Compatibility helpers
# =========================================================
def ui_image(path: Path, width: int | str = "stretch", caption: str | None = None):
    try:
        st.image(str(path), width=width, caption=caption)
    except TypeError:
        if width == "stretch":
            st.image(str(path), use_container_width=True, caption=caption)
        elif width == "content":
            st.image(str(path), use_container_width=False, caption=caption)
        else:
            st.image(str(path), width=width, caption=caption)


def ui_dataframe(df: pd.DataFrame, height: int | str = "content"):
    try:
        st.dataframe(df, width="stretch", height=height)
    except TypeError:
        if height == "content":
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True, height=height)


def ui_plotly(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
<style>
#MainMenu, footer {visibility:hidden;}

header[data-testid="stHeader"] {
    background: rgba(247,255,252,.96);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(15,118,110,.08);
}

.stApp {
    background: linear-gradient(180deg,#f7fffc 0%,#edf9f6 55%,#f8fffd 100%);
}

.block-container {
    max-width: 1080px;
    padding-top: 3.25rem;
    padding-bottom: 2.4rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#e6fffb 0%,#f0fdfa 100%);
    border-right: 1px solid rgba(15,118,110,.16);
}

[data-testid="stSidebar"] * {
    color:#0f172a !important;
}

.sidebar-title {
    color:#0f172a !important;
    font-size:1.12rem;
    font-weight:950;
    line-height:1.15;
    margin-top:.45rem;
    margin-bottom:.85rem;
}

/* Universal text readability */
h1,h2,h3,h4,h5,h6,p,li,span,small,label {
    color:#0f172a !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color:#0f172a !important;
}

/* Cards */
.hero-card, .panel-card, .metric-card {
    background: rgba(255,255,255,.985);
    border: 1px solid rgba(15,118,110,.16);
    box-shadow: 0 10px 24px rgba(15,23,42,.055);
}

.hero-card {
    border-radius: 26px;
    padding: 1.05rem 1.15rem;
    margin-bottom: 1rem;
}

.panel-card {
    border-radius: 20px;
    padding: 1.05rem 1.15rem;
    margin-bottom: 1rem;
}

.metric-card {
    border-radius: 18px;
    border-left: 6px solid #14b8a6;
    padding: .95rem 1rem;
    min-height: 104px;
}

.hero-title {
    color:#0f172a !important;
    font-size:2.25rem;
    font-weight:950;
    letter-spacing:-.045em;
    line-height:1.08;
}

.hero-subtitle {
    color:#334155 !important;
    font-size:1rem;
    line-height:1.48;
    margin-top:.35rem;
}

.page-title {
    color:#0f172a !important;
    font-size:1.55rem;
    font-weight:950;
    margin: .15rem 0 .25rem 0;
    line-height:1.2;
}

.page-caption {
    color:#475569 !important;
    font-size:.98rem;
    margin-bottom:1rem;
    line-height:1.55;
}

.card-title {
    color:#0f172a !important;
    font-size:1.25rem;
    font-weight:900;
    margin-bottom:.25rem;
}

.card-subtitle {
    color:#475569 !important;
    font-size:.94rem;
    margin-bottom:.85rem;
}

/* Alert boxes */
.medical-box {
    background:#fff7ed;
    border:1px solid #fed7aa;
    border-left:6px solid #fb923c;
    border-radius:18px;
    padding:.95rem 1rem;
    margin:.9rem 0 1rem 0;
    color:#7c2d12 !important;
    line-height:1.55;
}

.medical-box * {
    color:#7c2d12 !important;
}

.info-box {
    background:#eff6ff;
    border:1px solid #bfdbfe;
    border-left:6px solid #38bdf8;
    border-radius:18px;
    padding:.95rem 1rem;
    margin:.9rem 0 1rem 0;
    line-height:1.55;
}

.info-box * {
    color:#1e3a8a !important;
}

/* Metrics */
.metric-label {
    color:#64748b !important;
    font-size:.78rem;
    font-weight:850;
    text-transform:uppercase;
    letter-spacing:.04em;
}

.metric-value {
    color:#0f172a !important;
    font-size:1.55rem;
    font-weight:950;
    line-height:1.15;
    margin-top:.25rem;
}

.metric-note {
    color:#64748b !important;
    font-size:.78rem;
    margin-top:.25rem;
}

/* Badge */
.badge {
    display:inline-block;
    background:#ccfbf1;
    border:1px solid #99f6e4;
    color:#115e59 !important;
    border-radius:999px;
    padding:.28rem .68rem;
    margin:.14rem .18rem .14rem 0;
    font-size:.82rem;
    font-weight:800;
}

/* Footer */
.footer-custom {
    color:#475569 !important;
    font-size:.86rem;
    text-align:center;
    padding-top:1.2rem;
    padding-bottom:.3rem;
}

/* Chat message */
div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.97);
    border: 1px solid rgba(15,118,110,.12);
    border-radius: 18px;
}

div[data-testid="stChatMessage"] * {
    color:#0f172a !important;
}

/* Input */
.stTextInput input {
    background:#fff !important;
    color:#0f172a !important;
    border:1px solid rgba(15,118,110,.28) !important;
    border-radius:12px !important;
}

.stTextInput input::placeholder {
    color:#64748b !important;
}

/* Buttons: visible before hover */
.stButton > button {
    background:#0f766e !important;
    color:#ffffff !important;
    border:1px solid #0f766e !important;
    border-radius:12px !important;
    font-weight:850 !important;
    box-shadow:0 6px 14px rgba(15,118,110,.18);
}

.stButton > button p,
.stButton > button span,
.stButton > button div {
    color:#ffffff !important;
}

.stButton > button:hover {
    background:#0d5f59 !important;
    color:#ffffff !important;
    border-color:#0d5f59 !important;
}

/* Tabs and expanders */
button[data-baseweb="tab"] p {
    color:#0f172a !important;
    font-weight:800 !important;
}

button[data-baseweb="tab"][aria-selected="true"] p {
    color:#0f766e !important;
}

div[data-testid="stExpander"] {
    background: rgba(255,255,255,.98) !important;
    border: 1px solid rgba(15,118,110,.16) !important;
    border-radius: 14px !important;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary * {
    color:#0f172a !important;
    font-weight:800 !important;
}

div[data-testid="stExpander"] div,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] span {
    color:#0f172a !important;
}

/* Dataframe area */
[data-testid="stDataFrame"] {
    background:#ffffff !important;
    border-radius:14px !important;
}

hr {
    margin-top:.8rem;
    margin-bottom:.8rem;
}

@media (max-width:900px) {
    .block-container {padding-left:1rem; padding-right:1rem; padding-top:2.8rem;}
    .hero-title {font-size:1.85rem;}
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Data helpers
# =========================================================
def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df.empty:
        return None
    lower = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        col_l = str(col).lower().strip()
        for cand in candidates:
            if cand.lower() in col_l:
                return col
    return None


def csv_files() -> list[Path]:
    return sorted(DATA_DIR.rglob("*.csv")) if DATA_DIR.exists() else []


@st.cache_data(show_spinner=False)
def load_intents() -> pd.DataFrame:
    for path in [
        DATA_DIR / "processed" / "intent_dataset_clean.csv",
        DATA_DIR / "processed" / "intent_dataset.csv",
        DATA_DIR / "raw" / "intent_dataset.csv",
        DATA_DIR / "intent_dataset.csv",
    ]:
        df = read_csv(path)
        if not df.empty:
            return df
    for path in csv_files():
        df = read_csv(path)
        if get_col(df, ["question", "pertanyaan", "text"]) and get_col(df, ["intent", "label", "category", "kategori"]):
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    for path in [
        DATA_DIR / "raw" / "product_catalog.csv",
        DATA_DIR / "processed" / "product_catalog.csv",
        DATA_DIR / "product_catalog.csv",
    ]:
        df = read_csv(path)
        if not df.empty:
            return df
    for path in csv_files():
        df = read_csv(path)
        if get_col(df, ["product_name", "nama_produk", "name", "produk"]) or get_col(df, ["category", "kategori"]):
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return read_csv(REPORTS_DIR / "model_comparison_metrics.csv")


@st.cache_data(show_spinner=False)
def load_error() -> pd.DataFrame:
    for path in [
        REPORTS_DIR / "error_summary_logistic_regression.csv",
        REPORTS_DIR / "error_summary_multinomial_nb.csv",
        REPORTS_DIR / "error_analysis_logistic_regression.csv",
        REPORTS_DIR / "error_analysis_multinomial_nb.csv",
    ]:
        df = read_csv(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def metadata() -> dict[str, Any]:
    if load_metadata is None:
        return {"confidence_threshold": 0.30, "best_model": "-"}
    try:
        return load_metadata()
    except Exception:
        return {"confidence_threshold": 0.30, "best_model": "-"}


def asset(name: str) -> Path:
    return ASSETS_DIR / name


def show_asset(name: str, width: int | str = "stretch") -> bool:
    path = asset(name)
    if path.exists():
        ui_image(path, width=width)
        return True
    return False


def fmt_num(v: Any) -> str:
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return str(v)


# =========================================================
# UI parts
# =========================================================
def sidebar() -> str:
    with st.sidebar:
        if not show_asset("logo_apotekcare.png", width=76):
            st.markdown("## 💊")

        st.markdown('<div class="sidebar-title">ApotekCare Assistant</div>', unsafe_allow_html=True)
        st.markdown("---")

        menu = st.radio(
            "Menu",
            ["Chatbot", "Dashboard Visualisasi", "Cara Penggunaan", "Tentang Aplikasi"],
            label_visibility="collapsed",
        )
    return menu


def render_header():
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.9, 4.8, 1.2], vertical_alignment="center")
    with c1:
        if not show_asset("logo_apotekcare.png", width=82):
            st.markdown("<div style='font-size:3.4rem;text-align:center;'>💊</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="hero-title">ApotekCare Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-subtitle">Asisten layanan apotek yang membantu menjawab pertanyaan pelanggan secara ringkas, ramah, dan terarah.</div>',
            unsafe_allow_html=True,
        )
    with c3:
        if not show_asset("anime_pharmacist.png", width="stretch"):
            st.markdown("<div style='font-size:3.6rem;text-align:center;'>👩‍⚕️</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def metric(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown('<div class="footer-custom">Handmade with ❤️ @ 2026 - by Achmad Kamil</div>', unsafe_allow_html=True)


def call_chatbot(question: str, context_products: list[dict[str, Any]]) -> dict[str, Any]:
    if chatbot_response is None:
        return {
            "answer": f"Chatbot engine gagal dimuat: {CHATBOT_IMPORT_ERROR}",
            "intent": "import_error",
            "confidence": 0.0,
            "status": "error",
            "products": [],
        }
    try:
        sig = inspect.signature(chatbot_response)
        kwargs = {}
        if "threshold" in sig.parameters:
            kwargs["threshold"] = float(metadata().get("confidence_threshold", 0.30))
        if "context_products" in sig.parameters:
            kwargs["context_products"] = context_products
        return chatbot_response(question, **kwargs)
    except Exception as exc:
        return {
            "answer": f"Maaf, terjadi kendala saat memproses pertanyaan: {exc}",
            "intent": "runtime_error",
            "confidence": 0.0,
            "status": "error",
            "products": [],
        }


# =========================================================
# Pages
# =========================================================
def page_chatbot():
    render_header()

    st.markdown(
        """
        <div class="medical-box">
            <b>Batasan medis:</b> chatbot ini tidak memberikan diagnosis, resep dokter, atau dosis personal.
            Untuk kondisi berat, bayi, ibu hamil/menyusui, antibiotik, atau penyakit kronis,
            konsultasikan langsung dengan dokter atau apoteker.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Halo, saya ApotekCare Assistant 👋\n\n"
                    "Silakan ajukan pertanyaan seputar layanan apotek, stok produk, harga produk, "
                    "atau rekomendasi produk non-resep."
                ),
            }
        ]
    if "last_products" not in st.session_state:
        st.session_state.last_products = []

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Percakapan</div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Tulis Pertanyaan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Contoh: obat untuk sakit kepala, apakah stok obat batuk tersedia, atau berapa harga vitamin C.</div>',
        unsafe_allow_html=True,
    )
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Pertanyaan",
            placeholder="Tulis pertanyaan Anda di sini...",
            label_visibility="collapsed",
        )
        submit = st.form_submit_button("Kirim Pertanyaan")
    st.markdown("</div>", unsafe_allow_html=True)

    if submit and question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        result = call_chatbot(question, st.session_state.get("last_products", []))
        if result.get("products"):
            st.session_state.last_products = result["products"]
        st.session_state.messages.append(
            {"role": "assistant", "content": result.get("answer", "Maaf, saya belum dapat memberikan jawaban.")}
        )
        st.rerun()

    if st.button("Reset Percakapan"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Percakapan telah direset. Silakan ajukan pertanyaan baru seputar ApotekCare.",
            }
        ]
        st.session_state.last_products = []
        st.rerun()


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        st.info("Data visualisasi belum tersedia.")
        return
    if px:
        fig = px.bar(df, x=x, y=y, color=y, title=title, color_continuous_scale="Tealgrn")
        fig.update_layout(
            height=430,
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="rgba(255,255,255,0)",
            xaxis_tickangle=-25,
        )
        ui_plotly(fig)
    else:
        st.bar_chart(df.set_index(x)[y])


def plot_pie(df: pd.DataFrame, names: str, values: str, title: str):
    if df.empty:
        st.info("Data visualisasi belum tersedia.")
        return
    if px:
        fig = px.pie(
            df,
            names=names,
            values=values,
            hole=.48,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(height=430, plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)")
        ui_plotly(fig)
    else:
        ui_dataframe(df)


def page_dashboard():
    st.markdown('<div class="page-title">Dashboard Visualisasi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-caption">Ringkasan dataset, katalog produk, evaluasi model, confusion matrix, dan error analysis.</div>',
        unsafe_allow_html=True,
    )

    intents = load_intents()
    products = load_products()
    metrics_df = load_metrics()
    errors = load_error()

    intent_col = get_col(intents, ["intent", "label", "category", "kategori", "kelas", "tag"])
    product_col = get_col(products, ["category", "kategori", "product_category", "jenis_produk", "jenis"])

    best_f1 = "-"
    if not metrics_df.empty and "f1_macro" in metrics_df.columns:
        best_f1 = f"{float(metrics_df['f1_macro'].max()):.4f}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Total Pertanyaan", fmt_num(len(intents)), "dataset intent")
    with c2:
        metric("Total Intent", fmt_num(intents[intent_col].nunique() if not intents.empty and intent_col else 0), "kelas klasifikasi")
    with c3:
        metric("Total Produk", fmt_num(len(products)), "katalog produk")
    with c4:
        metric("F1 Macro Terbaik", best_f1, "evaluasi model")

    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Evaluasi Model", "Error Analysis", "Visualisasi Report"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribusi Intent")
            if not intents.empty and intent_col:
                df = intents[intent_col].value_counts().reset_index()
                df.columns = ["intent", "jumlah"]
                plot_bar(df, "intent", "jumlah", "Distribusi Intent Dataset")
            else:
                st.info("Dataset intent belum tersedia.")
        with col2:
            st.markdown("#### Distribusi Kategori Produk")
            if not products.empty and product_col:
                df = products[product_col].value_counts().reset_index()
                df.columns = ["kategori", "jumlah"]
                plot_pie(df, "kategori", "jumlah", "Distribusi Kategori Produk")
            else:
                st.info("Katalog produk belum tersedia.")

        with st.expander("Preview Dataset Intent"):
            ui_dataframe(intents.head(30))
        with st.expander("Preview Product Catalog"):
            ui_dataframe(products.head(30))

    with tab2:
        st.markdown("#### Perbandingan Metrik Model")
        if not metrics_df.empty and "model" in metrics_df.columns:
            metric_cols = [c for c in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"] if c in metrics_df.columns]
            if px and metric_cols:
                plot_df = metrics_df.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="score")
                fig = px.bar(
                    plot_df,
                    x="metric",
                    y="score",
                    color="model",
                    barmode="group",
                    title="Perbandingan Metrik Model",
                    color_discrete_sequence=["#0f766e", "#38bdf8", "#f59e0b"],
                )
                fig.update_layout(
                    height=430,
                    yaxis_range=[0, 1.05],
                    plot_bgcolor="rgba(255,255,255,0)",
                    paper_bgcolor="rgba(255,255,255,0)",
                )
                ui_plotly(fig)
            ui_dataframe(metrics_df)
        else:
            st.info("File model_comparison_metrics.csv belum tersedia.")

    with tab3:
        st.markdown(
            """
            <div class="info-box">
                Error analysis digunakan untuk melihat intent yang sering tertukar dan menjadi dasar
                perbaikan dataset atau template respons chatbot.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not errors.empty:
            ui_dataframe(errors.head(50))
        else:
            st.info("File error analysis belum tersedia.")

    with tab4:
        st.markdown("#### Confusion Matrix")
        col1, col2 = st.columns(2)
        with col1:
            path = FIGURES_DIR / "confusion_matrix_logistic_regression.png"
            if path.exists():
                ui_image(path, width="stretch", caption="Confusion Matrix - Logistic Regression")
            else:
                st.info("Confusion matrix Logistic Regression belum tersedia.")
        with col2:
            path = FIGURES_DIR / "confusion_matrix_multinomial_nb.png"
            if path.exists():
                ui_image(path, width="stretch", caption="Confusion Matrix - Multinomial Naive Bayes")
            else:
                st.info("Confusion matrix Multinomial Naive Bayes belum tersedia.")

        st.markdown("#### Visualisasi Tambahan")
        cols = st.columns(3)
        for i, (cap, path) in enumerate([
            ("Distribusi Intent", FIGURES_DIR / "intent_distribution.png"),
            ("Perbandingan Model", FIGURES_DIR / "model_metrics_comparison.png"),
            ("Distribusi Produk", FIGURES_DIR / "product_category_distribution.png"),
        ]):
            with cols[i]:
                if path.exists():
                    ui_image(path, width="stretch", caption=cap)
                else:
                    st.caption(f"{path.name} belum tersedia.")


def page_usage():
    st.markdown('<div class="page-title">Cara Penggunaan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-caption">Panduan penggunaan ApotekCare Assistant untuk pengguna dan penguji aplikasi.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="panel-card">
                <h3>Langkah Penggunaan</h3>
                <ol>
                    <li>Buka menu <b>Chatbot</b>.</li>
                    <li>Baca greeting awal dari ApotekCare Assistant.</li>
                    <li>Tulis pertanyaan pada form percakapan.</li>
                    <li>Gunakan pertanyaan yang spesifik agar jawaban lebih akurat.</li>
                    <li>Gunakan tombol <b>Reset Percakapan</b> jika ingin memulai ulang.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="panel-card">
                <h3>Contoh Pertanyaan</h3>
                <ul>
                    <li>obat untuk sakit kepala</li>
                    <li>apakah stok obat batuk tersedia?</li>
                    <li>berapa harga vitamin C?</li>
                    <li>harganya berapa?</li>
                    <li>jam operasional apotek sampai jam berapa?</li>
                    <li>vitamin untuk daya tahan tubuh</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="medical-box">
            <b>Batasan:</b> sistem ini tidak memberikan diagnosis, resep dokter, atau dosis personal.
            Harga dan stok pada katalog bersifat simulasi/prototipe dan tetap perlu dikonfirmasi ke admin/apoteker.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel-card">
            <h3>Tips Penggunaan</h3>
            <ul>
                <li>Lebih baik menulis <b>berapa harga obat batuk?</b> daripada hanya <b>berapa harganya?</b>.</li>
                <li>Lebih baik menulis <b>obat untuk sakit kepala</b> daripada hanya <b>sakit</b>.</li>
                <li>Untuk pertanyaan medis berat, chatbot akan mengarahkan pengguna ke dokter atau apoteker.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_about():
    st.markdown('<div class="page-title">Tentang Aplikasi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-caption">Informasi teknis dan konteks akademik ApotekCare Assistant.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel-card">
            <h3>ApotekCare Assistant</h3>
            <p>
                Chatbot layanan apotek berbasis <b>Natural Language Processing</b> dan <b>Text Mining</b>
                untuk membantu informasi layanan, stok, harga, serta rekomendasi produk non-resep.
            </p>
            <span class="badge">NLP</span>
            <span class="badge">Text Mining</span>
            <span class="badge">Intent Classification</span>
            <span class="badge">ApotekCare</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="panel-card">
                <h3>Metode yang Digunakan</h3>
                <ul>
                    <li>Text preprocessing</li>
                    <li>TF-IDF Vectorization</li>
                    <li>Logistic Regression</li>
                    <li>Multinomial Naive Bayes</li>
                    <li>Confidence threshold</li>
                    <li>Rule-based handling untuk kasus khusus</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="panel-card">
                <h3>Evaluasi Model</h3>
                <ul>
                    <li>Accuracy</li>
                    <li>Precision</li>
                    <li>Recall</li>
                    <li>F1-score</li>
                    <li>Classification report</li>
                    <li>Confusion matrix</li>
                    <li>Error analysis</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    meta = metadata()
    with st.expander("Detail Konfigurasi Teknis"):
        st.write(f"Model terbaik berdasarkan evaluasi: `{meta.get('best_model', '-')}`")
        st.write(f"Confidence threshold default: `{meta.get('confidence_threshold', '-')}`")
        st.write("Detail ini ditempatkan di halaman Tentang Aplikasi agar halaman chatbot tetap bersih.")

    st.markdown(
        """
        <div class="info-box">
            Project ini dikembangkan untuk memenuhi UAS Mata Kuliah Data Mining.
            Fokus utama project adalah implementasi text mining, klasifikasi intent, evaluasi model,
            error analysis, dan penyajian hasil melalui dashboard interaktif.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    menu = sidebar()
    if menu == "Chatbot":
        page_chatbot()
    elif menu == "Dashboard Visualisasi":
        page_dashboard()
    elif menu == "Cara Penggunaan":
        page_usage()
    elif menu == "Tentang Aplikasi":
        page_about()
    footer()


if __name__ == "__main__":
    main()
