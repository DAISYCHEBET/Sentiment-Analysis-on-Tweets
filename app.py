# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import altair as alt

# try to import the preprocessing module: accept either preprocessing.py or preprocess.py
try:
    import preprocessing as preproc
except Exception:
    import preprocess as preproc

st.set_page_config(page_title="Tweet Sentiment (LSTM)", layout="wide")

# --------- Config (match training) ----------
MAX_SEQUENCE_LENGTH = 50   # must match your training MAX_SEQUENCE_LENGTH
CLASS_MAP = {0: "negative", 1: "neutral", 2: "positive"}
MODEL_PATH = "sentiment_lstm.keras"
TOKENIZER_PATH = "tokenizer.pkl"
# --------------------------------------------

# session state flags for stopping long jobs & holding example text
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "example_text" not in st.session_state:
    st.session_state.example_text = ""

@st.cache_resource(show_spinner=False)
def load_resources():
    model = load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    return model, tokenizer

model, tokenizer = load_resources()

st.title("Twitter Sentiment Analysis")
st.markdown("Single tweet or batch file prediction — uses your trained LSTM model.")

menu = st.sidebar.selectbox("Mode", ["Home", "Single Prediction", "Batch Prediction", "About"])

# --------- helper: get probabilities & label ----------
def predict_from_texts_vectorized(texts):
    cleaned = [preproc.preprocess_new_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    pad = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    probs = model.predict(pad, verbose=0)
    if probs.ndim == 1:
        probs = np.vstack([1-probs, probs]).T
    dfp = pd.DataFrame(probs, columns=[f"prob_{CLASS_MAP[i]}" for i in range(probs.shape[1])])
    preds = np.argmax(probs, axis=1)
    dfp["pred_label"] = [CLASS_MAP[int(p)] for p in preds]
    dfp["pred_label_id"] = preds
    return dfp, cleaned

def predict_single_text(text):
    cleaned = preproc.preprocess_new_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    probs = model.predict(pad, verbose=0)[0]
    if probs.ndim == 0:
        probs = np.array([1-probs, probs])
    pred = int(np.argmax(probs))
    probs_dict = {f"prob_{CLASS_MAP[i]}": float(probs[i]) for i in range(len(probs))}
    return pred, probs_dict, cleaned

# ---------- Home ----------
if menu == "Home":
    st.write("This app uses a Bidirectional LSTM for tweet sentiment classification.")
    st.write("- Preprocessing: your `preprocessing.py` pipeline (contractions, slang, lemmatization).")
    st.write("- Model: `sentiment_lstm.keras`")
    st.write("- Tokenizer: `tokenizer.pkl`")
    st.info("Choose 'Single Prediction' to input one tweet or 'Batch Prediction' to upload CSV/XLSX.")

# ---------------- Single Prediction (with examples + colored horizontal bars) ---------------
if menu == "Single Prediction":
    st.header("Single Tweet Prediction")
    # Example demo tweets
    st.subheader("Quick examples")
    col_a, col_b, col_c = st.columns(3)
    examples = [
        "I love this new phone! Best purchase ever ❤️",
        "This is okay, nothing special. Could be better.",
        "Worst customer service. Very disappointed and frustrated."
    ]
    with col_a:
        if st.button("Example 1"):
            st.session_state.example_text = examples[0]
    with col_b:
        if st.button("Example 2"):
            st.session_state.example_text = examples[1]
    with col_c:
        if st.button("Example 3"):
            st.session_state.example_text = examples[2]

    # Text area (prefilled with selected example if any)
    tweet = st.text_area("Enter tweet text", value=st.session_state.example_text, height=140, placeholder="Type or paste a tweet here...")
    st.write("")  # spacing

    if st.button("Predict"):
        if not tweet or tweet.strip() == "":
            st.warning("Please enter tweet text first.")
        else:
            with st.spinner("Preprocessing & predicting..."):
                dfp, cleaned_list = predict_from_texts_vectorized([tweet])
                probs = dfp[[c for c in dfp.columns if c.startswith("prob_")]].iloc[0]
                pred_label = dfp.loc[0, "pred_label"]
                cleaned_text = cleaned_list[0]

            # Prominent predicted label with emoji + color
            emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😞"}
            color_map = {"positive": "#2ca02c", "neutral": "#d6a600", "negative": "#d62728"}
            pred_emoji = emoji_map.get(pred_label, "")
            pred_color = color_map.get(pred_label, "#000000")

            st.markdown(f"### Predicted sentiment: <span style='color:{pred_color}'>{pred_emoji}  {pred_label.upper()}</span>", unsafe_allow_html=True)
            st.write("**Cleaned input used for prediction:**")
            st.code(cleaned_text)

            # Build a DataFrame for altair
            prob_items = [{"class": c.replace("prob_", ""), "prob": float(p)} for c, p in probs.items()]
            prob_df = pd.DataFrame(prob_items)
            prob_df["class"] = prob_df["class"].map(lambda x: x.title())
            # Color mapping: negative->red, neutral->yellow, positive->green
            alt_colors = {"Negative": color_map["negative"], "Neutral": color_map["neutral"], "Positive": color_map["positive"]}

            # Chart: horizontal bars using Altair
            chart = alt.Chart(prob_df).mark_bar().encode(
                x=alt.X('prob:Q', axis=alt.Axis(format='.0%')),
                y=alt.Y('class:N', sort='-x'),
                color=alt.Color('class:N', scale=alt.Scale(domain=list(alt_colors.keys()), range=list(alt_colors.values())), legend=None)
            ).properties(width=600, height=120)
            # text labels on bars
            text = chart.mark_text(
                align='left',
                dx=3
            ).encode(text=alt.Text('prob:Q', format='.2f'))
            st.altair_chart(chart + text, use_container_width=False)

# ---------------- Batch Prediction (live streaming, cancel, XLSX + CSV download, filtering, uncertainty) ---------------
if menu == "Batch Prediction":
    st.header("Batch Prediction (CSV / XLSX)")
    uploaded = st.file_uploader("Upload a CSV or Excel file with at least one text column", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df = None

        if df is not None:
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # let user choose which column contains text
            text_cols = [c for c in df.columns if df[c].dtype == object]
            if not text_cols:
                st.error("No text-like columns found. Ensure there is at least one string column with tweet text.")
            else:
                col_selected = st.selectbox("Select text column", text_cols)

                # Uncertainty / threshold slider
                threshold = st.slider("Uncertainty threshold (max class probability below this is flagged 'uncertain')", min_value=0.5, max_value=0.95, value=0.6, step=0.05)

                # Run controls
                col_run, col_opts = st.columns([1,2])
                with col_opts:
                    use_vectorized = st.checkbox("Use fast vectorized prediction (no progress ETA) — recommended for large files", value=False)
                    show_stream = st.checkbox("Stream results as they come (live update)", value=True)
                with col_run:
                    run_btn = st.button("Run predictions on file")
                    stop_btn = st.button("STOP processing")  # sets a flag to stop

                if stop_btn:
                    st.session_state.stop_requested = True

                if run_btn:
                    # reset stop flag
                    st.session_state.stop_requested = False

                    texts = df[col_selected].astype(str).tolist()
                    n = len(texts)
                    if n == 0:
                        st.warning("No texts found in selected column.")
                    else:
                        # Prepare streaming area and progress
                        placeholder = st.empty()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results_rows = []
                        cleaned_texts = []

                        # Option A: vectorized (fast) - but we still stream rows if show_stream True by iterating results for display
                        if use_vectorized:
                            with st.spinner("Preprocessing & vectorized predicting..."):
                                preds_df, cleaned_texts = predict_from_texts_vectorized(texts)
                                # add max prob and uncertainty
                                preds_df["max_prob"] = preds_df[[c for c in preds_df.columns if c.startswith("prob_")]].max(axis=1)
                                preds_df["uncertain"] = preds_df["max_prob"] < threshold
                                out = df.copy().reset_index(drop=True)
                                out["cleaned_text_for_model"] = cleaned_texts
                                out = pd.concat([out, preds_df.reset_index(drop=True)], axis=1)

                                # optionally stream rows progressively (simulate)
                                if show_stream:
                                    # stream by chunks for UX
                                    chunk_size = max(1, n // 20)
                                    for start in range(0, n, chunk_size):
                                        end = min(n, start + chunk_size)
                                        progress = int((end / n) * 100)
                                        progress_bar.progress(progress)
                                        status_text.markdown(f"Processed **{end}/{n}**")
                                        placeholder.dataframe(out.iloc[:end].head(50))
                                        time.sleep(0.1)
                                        if st.session_state.stop_requested:
                                            status_text.markdown("**Stopped by user.**")
                                            break
                                else:
                                    progress_bar.progress(100)
                                    placeholder.dataframe(out.head(50))

                        else:
                            # Per-item prediction (gives accurate ETA) with streaming
                            start_time = time.time()
                            for i, t in enumerate(texts):
                                if st.session_state.stop_requested:
                                    status_text.markdown("**Stopped by user.**")
                                    break
                                pred_id, probs_dict, cleaned = predict_single_text(t)
                                row = {"pred_label_id": pred_id, "pred_label": CLASS_MAP[pred_id]}
                                # merge probs
                                row.update(probs_dict)
                                results_rows.append(row)
                                cleaned_texts.append(cleaned)

                                # Update progress
                                elapsed = time.time() - start_time
                                completed = i + 1
                                avg_time = elapsed / completed
                                remaining = n - completed
                                eta = avg_time * remaining
                                pct = int((completed / n) * 100)
                                progress_bar.progress(pct)
                                status_text.markdown(f"Processed **{completed}/{n}** — elapsed: **{elapsed:.1f}s**, ETA: **{eta:.1f}s** (avg {avg_time:.3f}s/row)")

                                # Stream current partial results
                                if show_stream:
                                    partial = pd.DataFrame(results_rows)
                                    # compute max_prob & uncertain so display includes them
                                    prob_cols = [c for c in partial.columns if c.startswith("prob_")]
                                    if prob_cols:
                                        partial["max_prob"] = partial[prob_cols].max(axis=1)
                                        partial["uncertain"] = partial["max_prob"] < threshold
                                    display_df = df.copy().reset_index(drop=True).loc[:len(partial)-1].copy()
                                    display_df["cleaned_text_for_model"] = cleaned_texts
                                    display_df = pd.concat([display_df.reset_index(drop=True), partial.reset_index(drop=True)], axis=1)
                                    placeholder.dataframe(display_df.head(50))

                            # assemble final out
                            if results_rows:
                                probs_df = pd.DataFrame(results_rows)
                                probs_df = probs_df.reset_index(drop=True)
                                # compute max_prob & uncertain
                                prob_cols = [c for c in probs_df.columns if c.startswith("prob_")]
                                if prob_cols:
                                    probs_df["max_prob"] = probs_df[prob_cols].max(axis=1)
                                    probs_df["uncertain"] = probs_df["max_prob"] < threshold

                                out = df.copy().reset_index(drop=True)
                                out["cleaned_text_for_model"] = cleaned_texts
                                out = pd.concat([out, probs_df.reset_index(drop=True)], axis=1)
                            else:
                                out = df.copy().reset_index(drop=True)
                                out["cleaned_text_for_model"] = cleaned_texts
                                st.warning("No predictions were produced (possibly stopped).")

                        # If processing wasn't stopped, show final results and controls
                        if not st.session_state.stop_requested:
                            progress_bar.progress(100)
                            status_text.markdown("Processing complete.")
                        # Final output visible
                        if 'out' in locals():
                            st.success("Predictions ready.")
                            st.write("Results preview (first 50 rows):")
                            st.dataframe(out.head(50))

                            # Add filtering and sorting controls
                            st.sidebar.subheader("Result filters & sorting")
                            filter_label = st.sidebar.selectbox("Filter by predicted label (All / value)", ["All"] + list(CLASS_MAP.values()))
                            sort_by = st.sidebar.selectbox("Sort by", ["None", "max_prob"] + [f"prob_{v}" for v in CLASS_MAP.values()])
                            asc = st.sidebar.checkbox("Ascending sort", value=False)

                            display_out = out.copy()
                            # compute max_prob if not present
                            prob_cols = [c for c in display_out.columns if c.startswith("prob_")]
                            if prob_cols and "max_prob" not in display_out.columns:
                                display_out["max_prob"] = display_out[prob_cols].max(axis=1)
                            # apply filter
                            if filter_label != "All":
                                display_out = display_out[display_out["pred_label"] == filter_label]
                            # apply sort
                            if sort_by != "None" and sort_by in display_out.columns:
                                display_out = display_out.sort_values(by=sort_by, ascending=asc)

                            st.subheader("Filtered/Sorted results (first 200 rows)")
                            st.dataframe(display_out.head(200))

                            # Distribution
                            try:
                                dist = out["pred_label"].value_counts().reindex(list(CLASS_MAP.values())).fillna(0)
                            except Exception:
                                dist = out["pred_label"].value_counts()
                            st.subheader("Predicted class distribution")
                            st.bar_chart(dist)

                            # Downloads: CSV + XLSX
                            csv = out.to_csv(index=False).encode('utf-8')
                            st.download_button(label="Download results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

                            # XLSX
                            towrite = io.BytesIO()
                            with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                                display_out.to_excel(writer, index=False, sheet_name='predictions')
                            towrite.seek(0)
                            st.download_button(label="Download results as Excel (XLSX)", data=towrite, file_name="predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- About ----------
if menu == "About":
    st.header("About this app")
    st.write("Built with your LSTM model and preprocessing pipeline.")
    st.write("Ensure these files are in the same folder: `preprocessing.py` (or `preprocess.py`), `tokenizer.pkl`, and `sentiment_lstm.keras`.")
    st.write("Run the app with: `streamlit run app.py`")
