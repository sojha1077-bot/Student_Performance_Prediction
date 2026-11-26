"""
app.py  -  Student Performance Prediction Using Linear Regression
-------------------------------------------------------------------
This is the Streamlit dashboard. Run it with:

    streamlit run app.py

The dashboard has 6 pages (see the menu on the left):
  Home -> 1. Upload Data -> 2. Clean Data -> 3. Visualize -> 4. Train & Evaluate -> 5. Predict Score
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import visuals
from src.data_utils import (
    clean_data,
    dataset_summary,
    guess_target_column,
    is_id_like,
    load_csv,
    numeric_like_columns,
)
from src.model import (
    MODEL_PATH,
    bundle_to_bytes,
    explain_prediction,
    load_model_bundle,
    make_bundle,
    make_equation,
    predict_scores,
    prepare_new_data,
    save_model_bundle,
    train_and_evaluate,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(APP_DIR, "data", "sample_students.csv")

PAGES = [
    "🏠 Home",
    "1️⃣ Upload Data",
    "2️⃣ Clean Data",
    "3️⃣ Visualize",
    "4️⃣ Train & Evaluate",
    "5️⃣ Predict Score",
]

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)


# --------------------------------------------------------------------------
# Memory of the app (session state keeps values while you click around)
# --------------------------------------------------------------------------
def init_state():
    defaults = {
        "raw_df": None,          # the CSV exactly as uploaded
        "source_name": None,     # file name
        "source_key": None,      # used to notice when a new file is uploaded
        "clean_df": None,        # cleaned table
        "target": None,          # the score column
        "cleaning_report": [],
        "cleaning_info": {"yes_no_columns": [], "dropped_columns": []},
        "results": None,         # training results
        "bundle": None,          # trained model + everything needed to predict
        "page": PAGES[0],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_after_new_data():
    """A new file means old cleaning and old models no longer apply."""
    st.session_state.clean_df = None
    st.session_state.target = None
    st.session_state.cleaning_report = []
    st.session_state.cleaning_info = {"yes_no_columns": [], "dropped_columns": []}
    st.session_state.results = None
    st.session_state.bundle = None


def show_chart(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def go_to(page_name):
    st.session_state.page = page_name


def need(raw=False, clean=False):
    """Stop the page politely if an earlier step has not been done yet."""
    if raw and st.session_state.raw_df is None:
        st.warning("No data yet. Please upload your CSV file first.")
        st.button("Go to Upload Data", on_click=go_to, args=(PAGES[1],))
        st.stop()
    if clean and st.session_state.clean_df is None:
        st.warning("Please clean the data first.")
        st.button("Go to Clean Data", on_click=go_to, args=(PAGES[2],))
        st.stop()


init_state()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Student Score Predictor")
    st.caption("BCA Project · Linear Regression")
    st.radio("Go to", PAGES, key="page")
    st.divider()
    st.subheader("Progress")
    st.write(("✅" if st.session_state.raw_df is not None else "⬜") + " Data uploaded")
    st.write(("✅" if st.session_state.clean_df is not None else "⬜") + " Data cleaned")
    st.write(("✅" if st.session_state.results is not None else "⬜") + " Model trained")
    if st.session_state.source_name:
        st.caption(f"File: {st.session_state.source_name}")

page = st.session_state.page

# --------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------
if page == PAGES[0]:
    st.title("Student Performance Prediction Using Linear Regression")
    st.write(
        "This dashboard predicts a student's score from things like study hours, previous "
        "marks and sleep. It uses **Linear Regression**, which finds the best straight-line "
        "equation that connects the inputs to the score."
    )
    st.subheader("How to use it (5 steps)")
    st.markdown(
        """
1. **Upload Data** – upload your Kaggle CSV file (or try the sample file).
2. **Clean Data** – choose the score column and let the app fix empty cells, duplicates and Yes/No text.
3. **Visualize** – look at charts to understand the data.
4. **Train & Evaluate** – train the Linear Regression model and see how accurate it is.
5. **Predict Score** – type in a student's details and get a predicted score.
        """
    )
    st.info("Tip: start with the menu on the left. A ✅ appears in 'Progress' after each step.")
    st.subheader("Tools used")
    st.write("Python · Streamlit · Pandas · Scikit-learn · Matplotlib · Joblib")
    st.button("Start ➜ Upload Data", type="primary", on_click=go_to, args=(PAGES[1],))

# --------------------------------------------------------------------------
# 1. UPLOAD
# --------------------------------------------------------------------------
elif page == PAGES[1]:
    st.title("Step 1 · Upload your data")
    st.write("Upload the student CSV file you downloaded from Kaggle. Each row should be one student.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    use_sample = st.button("Or use the sample dataset (data/sample_students.csv)")

    if uploaded is not None:
        key = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.source_key != key:
            try:
                df = load_csv(uploaded)
            except ValueError as err:
                st.error(str(err))
                st.stop()
            reset_after_new_data()
            st.session_state.raw_df = df
            st.session_state.source_name = uploaded.name
            st.session_state.source_key = key
            st.success(f"Loaded '{uploaded.name}' successfully.")
    elif use_sample:
        if not os.path.exists(SAMPLE_PATH):
            st.error("The sample file data/sample_students.csv was not found.")
            st.stop()
        reset_after_new_data()
        st.session_state.raw_df = load_csv(SAMPLE_PATH)
        st.session_state.source_name = "sample_students.csv"
        st.session_state.source_key = "sample"
        st.success("Loaded the sample dataset.")

    df = st.session_state.raw_df
    if df is None:
        st.info("Waiting for a file…")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (students)", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Empty cells", int(df.isna().sum().sum()))
    c4.metric("Duplicate rows", int(df.duplicated().sum()))

    st.subheader("First 10 rows")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("About each column")
    st.dataframe(dataset_summary(df), width="stretch", hide_index=True)

    with st.expander("Basic statistics of the number columns"):
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            st.write("No number columns yet (they may be stored as text – cleaning will fix this).")
        else:
            st.dataframe(numeric.describe().T.round(2), width="stretch")

    st.button("Next ➜ Clean Data", type="primary", on_click=go_to, args=(PAGES[2],))

# --------------------------------------------------------------------------
# 2. CLEAN
# --------------------------------------------------------------------------
elif page == PAGES[2]:
    need(raw=True)
    st.title("Step 2 · Clean the data")
    raw = st.session_state.raw_df

    candidates = [c for c in numeric_like_columns(raw) if not is_id_like(c)]
    if not candidates:
        st.error("This file has no number column that could be a score. Please upload a different file.")
        st.stop()
    guess = st.session_state.target if st.session_state.target in candidates else guess_target_column(raw)
    target = st.selectbox(
        "Which column is the score you want to predict?",
        candidates,
        index=candidates.index(guess) if guess in candidates else 0,
        help="This is the 'target'. The model learns to predict this column.",
    )

    st.write("**Cleaning options**")
    remove_dupes = st.checkbox("Remove duplicate rows", value=True)
    remove_outliers = st.checkbox(
        "Remove extreme values (outliers)", value=False,
        help="Removes rows with values far away from the rest (IQR rule). Leave this off if unsure.",
    )

    if st.button("🧹 Clean my data", type="primary"):
        try:
            cleaned, report, info = clean_data(raw, target, remove_dupes, remove_outliers)
        except ValueError as err:
            st.error(str(err))
            st.stop()
        st.session_state.clean_df = cleaned
        st.session_state.target = str(target).strip()
        st.session_state.cleaning_report = report
        st.session_state.cleaning_info = info
        st.session_state.results = None
        st.session_state.bundle = None

    if st.session_state.clean_df is None:
        st.info("Choose the score column, then click **Clean my data**.")
        st.stop()

    cleaned = st.session_state.clean_df
    st.success("Data cleaned! Here is everything that was done:")
    for line in st.session_state.cleaning_report:
        st.write("✔️ " + line)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows before", f"{len(raw):,}")
    c2.metric("Rows after", f"{len(cleaned):,}", delta=len(cleaned) - len(raw))
    c3.metric("Empty cells left", int(cleaned.isna().sum().sum()))

    st.subheader("Clean data (first 10 rows)")
    st.dataframe(cleaned.head(10), width="stretch")
    st.download_button(
        "⬇️ Download clean data as CSV",
        cleaned.to_csv(index=False).encode("utf-8"),
        file_name="clean_student_data.csv",
        mime="text/csv",
    )
    st.button("Next ➜ Visualize", type="primary", on_click=go_to, args=(PAGES[3],))

# --------------------------------------------------------------------------
# 3. VISUALIZE
# --------------------------------------------------------------------------
elif page == PAGES[3]:
    need(raw=True, clean=True)
    st.title("Step 3 · Visualize the data")
    df = st.session_state.clean_df
    target = st.session_state.target
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target and not is_id_like(c)]
    text_cols = [c for c in df.columns if c not in df.select_dtypes(include="number").columns and not is_id_like(c)]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Score distribution", "Correlation", "Input vs score", "Groups", "Empty cells"]
    )

    with tab1:
        show_chart(visuals.score_histogram(df, target))
        s = df[target]
        st.caption(
            f"Most students scored around {s.median():.1f}. Lowest: {s.min():g}, highest: {s.max():g}. "
            "This chart shows how many students fall into each score range."
        )

    with tab2:
        if df.select_dtypes(include="number").shape[1] < 2:
            st.info("At least two number columns are needed for a correlation chart.")
        else:
            show_chart(visuals.correlation_heatmap(df))
            corr = df.select_dtypes(include="number").corr()[target].drop(target).sort_values(key=abs, ascending=False)
            st.caption(
                "Correlation goes from -1 to +1. Close to +1 = when this goes up, the score goes up. "
                "Close to -1 = when this goes up, the score goes down. Close to 0 = no straight-line link."
            )
            st.write("**Link of each column with the score (strongest first):**")
            st.dataframe(corr.round(3).rename("Correlation with score"), width="stretch")

    with tab3:
        if not numeric_cols:
            st.info("No number input columns to plot.")
        else:
            feature = st.selectbox("Choose an input column", numeric_cols)
            show_chart(visuals.feature_vs_target(df, feature, target))
            st.caption("Each dot is a student. The orange line shows the general trend.")

    with tab4:
        if not text_cols:
            st.info("This dataset has no text (category) columns such as gender. Nothing to compare here.")
        else:
            category = st.selectbox("Choose a group column", text_cols)
            show_chart(visuals.box_by_category(df, category, target))
            st.caption("Each box shows the middle 50% of scores for that group. The line inside is the median.")

    with tab5:
        fig = visuals.missing_values_chart(st.session_state.raw_df)
        if fig is None:
            st.success("Your original file had no empty cells.")
        else:
            show_chart(fig)
            st.caption("These empty cells were filled or removed in Step 2.")

    st.button("Next ➜ Train & Evaluate", type="primary", on_click=go_to, args=(PAGES[4],))

# --------------------------------------------------------------------------
# 4. TRAIN & EVALUATE
# --------------------------------------------------------------------------
elif page == PAGES[4]:
    need(raw=True, clean=True)
    st.title("Step 4 · Train & evaluate the Linear Regression model")
    df = st.session_state.clean_df
    target = st.session_state.target

    options = [c for c in df.columns if c != target]
    default_features = [c for c in options if not is_id_like(c)]
    features = st.multiselect(
        "Input columns (what the model uses to predict the score)",
        options,
        default=default_features,
        help="ID or name columns are left out by default because they do not affect marks.",
    )
    c1, c2 = st.columns(2)
    test_percent = c1.slider("Part of data kept for testing (%)", 10, 40, 20, step=5,
                             help="The model never sees these students while learning. They are used to check it.")
    seed = c2.number_input("Random seed (keeps results the same every run)", 0, 9999, 42, step=1)

    if st.button("🚀 Train Linear Regression Model", type="primary"):
        try:
            with st.spinner("Training…"):
                results = train_and_evaluate(df, target, features, test_percent / 100, int(seed))
        except ValueError as err:
            st.error(str(err))
            st.stop()
        bundle = make_bundle(results, st.session_state.cleaning_info["yes_no_columns"],
                             st.session_state.source_name)
        st.session_state.results = results
        st.session_state.bundle = bundle
        try:
            save_model_bundle(bundle)
            st.toast("Model saved to models/student_model.joblib")
        except OSError:
            st.toast("Model trained (could not save the file, but it still works in this session).")

    results = st.session_state.results
    if results is None:
        st.info("Choose the input columns and click **Train Linear Regression Model**.")
        st.stop()

    m = results["metrics"]
    st.success(f"Model trained on {m['train_rows']} students and tested on {m['test_rows']} new students.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² score (test)", f"{m['r2_test']:.3f}")
    c2.metric("MAE", f"{m['mae']:.2f}")
    c3.metric("RMSE", f"{m['rmse']:.2f}")
    c4.metric("MSE", f"{m['mse']:.2f}")

    r2 = m["r2_test"]
    quality = ("excellent" if r2 >= 0.9 else "good" if r2 >= 0.7 else "fair" if r2 >= 0.5 else "weak")
    st.markdown(
        f"""
**What these numbers mean (in simple words):**
- **R² = {r2:.3f}** → the model explains about **{max(r2, 0) * 100:.0f}%** of the differences in students' scores. That is **{quality}**. (1.0 would be perfect.)
- **MAE = {m['mae']:.2f}** → on average, a prediction is about **{m['mae']:.2f} marks** away from the real score.
- **RMSE = {m['rmse']:.2f}** → like MAE, but big mistakes count more.
- **MSE = {m['mse']:.2f}** → RMSE squared (used inside the maths).
- Training R² = {m['r2_train']:.3f}. If this is much higher than the test R², the model has memorised the training data (overfitting). Here the gap is {abs(m['r2_train'] - r2):.3f}.
        """
    )
    if "cv_r2_mean" in m:
        st.caption(
            f"5-fold cross-validation R²: {m['cv_r2_mean']:.3f} ± {m['cv_r2_std']:.3f} "
            "(the same test repeated 5 times on different splits – a steady number means a reliable model)."
        )

    st.subheader("The learned equation")
    st.code(make_equation(results), language=None)
    st.caption(
        "Read it like this: start from the first number (the intercept), then for every input multiply its "
        "value by the number in front of it and add them up. That total is the predicted score."
    )

    left, right = st.columns(2)
    with left:
        show_chart(visuals.actual_vs_predicted(results["y_test"], results["y_pred_test"], target))
        st.caption("Dots close to the dashed line = accurate predictions.")
    with right:
        show_chart(visuals.residual_plot(results["y_test"], results["y_pred_test"], target))
        st.caption("Errors spread evenly above and below 0 = the model is not biased.")

    st.subheader("Which inputs matter most?")
    show_chart(visuals.coefficient_chart(results["coefficients"]))
    st.dataframe(results["coefficients"].round(4), width="stretch", hide_index=True)
    st.caption("Coefficient = how many marks the score changes when that input goes up by 1.")

    st.subheader("Predictions for the test students")
    st.dataframe(results["test_table"].head(50), width="stretch", hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download test predictions (CSV)",
        results["test_table"].to_csv(index=False).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv",
    )
    d2.download_button(
        "⬇️ Download trained model (.joblib)",
        bundle_to_bytes(st.session_state.bundle),
        file_name="student_model.joblib",
        mime="application/octet-stream",
    )
    st.button("Next ➜ Predict Score", type="primary", on_click=go_to, args=(PAGES[5],))

# --------------------------------------------------------------------------
# 5. PREDICT
# --------------------------------------------------------------------------
elif page == PAGES[5]:
    st.title("Step 5 · Predict a student's score")

    bundle = st.session_state.bundle
    if bundle is None:
        bundle = load_model_bundle(MODEL_PATH)
        if bundle is not None:
            st.session_state.bundle = bundle
            st.info(f"Using the saved model from models/student_model.joblib "
                    f"(trained on '{bundle.get('source_name', 'unknown file')}').")
    if bundle is None:
        st.warning("No trained model yet. Please train the model first.")
        st.button("Go to Train & Evaluate", on_click=go_to, args=(PAGES[4],))
        st.stop()

    target = bundle["target"]
    st.caption(f"Model accuracy on test data: R² = {bundle['metrics']['r2_test']:.3f}, "
               f"average error = {bundle['metrics']['mae']:.2f} marks.")

    single_tab, many_tab = st.tabs(["One student", "Many students (CSV)"])

    with single_tab:
        with st.form("predict_form"):
            st.write("Enter the student's details:")
            values = {}
            columns = st.columns(2)
            for i, feature in enumerate(bundle["features"]):
                info = bundle["feature_info"][feature]
                box = columns[i % 2]
                if info["kind"] == "yes_no":
                    choice = box.selectbox(feature, ["Yes", "No"],
                                           index=0 if info["median"] >= 0.5 else 1)
                    values[feature] = 1 if choice == "Yes" else 0
                elif info["kind"] == "category":
                    options = info["options"]
                    values[feature] = box.selectbox(
                        feature, options,
                        index=options.index(info["mode"]) if info["mode"] in options else 0,
                    )
                else:
                    help_text = f"In the training data this was between {info['min']:g} and {info['max']:g}."
                    if info["whole_numbers"]:
                        values[feature] = box.number_input(
                            feature, value=int(round(info["median"])), step=1, help=help_text
                        )
                    else:
                        values[feature] = box.number_input(
                            feature, value=float(round(info["median"], 2)), step=0.1,
                            format="%.2f", help=help_text,
                        )
            submitted = st.form_submit_button("🔮 Predict score", type="primary")

        if submitted:
            row = pd.DataFrame([values])
            predicted = float(predict_scores(bundle, row)[0])
            low, high = bundle["target_min"], bundle["target_max"]
            st.metric(f"Predicted {target}", f"{predicted:.2f}")
            if high > low:
                position = min(max((predicted - low) / (high - low), 0.0), 1.0)
                st.progress(position, text=f"Position between the lowest ({low:g}) and highest ({high:g}) score in the data")
            if predicted < low or predicted > high:
                st.warning(
                    "This prediction is outside the range of scores in the training data. "
                    "Check that the inputs are realistic."
                )
            outside = [
                f for f in bundle["numeric"]
                if bundle["feature_info"][f]["kind"] == "number"
                and not (bundle["feature_info"][f]["min"] <= values[f] <= bundle["feature_info"][f]["max"])
            ]
            if outside:
                st.info("These inputs are outside the training range, so the prediction is less reliable: "
                        + ", ".join(outside))

            st.write("**How each input affected this prediction:**")
            explanation = explain_prediction(bundle, row)
            st.dataframe(explanation.round(3), width="stretch", hide_index=True)
            intercept = float(bundle["pipeline"].named_steps["model"].intercept_)
            st.caption(
                f"Predicted score = intercept ({intercept:.2f}) + the sum of the 'Points added' column."
            )

    with many_tab:
        st.write(
            "Upload a CSV with the same input columns as your training data: "
            + ", ".join(f"`{f}`" for f in bundle["features"])
            + ". The score column is not needed."
        )
        template = pd.DataFrame(
            [{f: (bundle["feature_info"][f].get("mode") if bundle["feature_info"][f]["kind"] == "category"
                  else ("Yes" if bundle["feature_info"][f]["kind"] == "yes_no"
                        else bundle["feature_info"][f]["median"]))
              for f in bundle["features"]}]
        )
        st.download_button("⬇️ Download an empty template CSV",
                           template.to_csv(index=False).encode("utf-8"),
                           file_name="new_students_template.csv", mime="text/csv")
        batch_file = st.file_uploader("Upload new students CSV", type=["csv"], key="batch_upload")
        if batch_file is not None:
            try:
                new_df = load_csv(batch_file)
                prepared = prepare_new_data(new_df, bundle)
            except ValueError as err:
                st.error(str(err))
                st.stop()
            output = new_df.copy()
            output[f"Predicted {target}"] = predict_scores(bundle, prepared).round(2)
            st.success(f"Predicted scores for {len(output)} students.")
            st.dataframe(output, width="stretch", hide_index=True)
            st.download_button(
                "⬇️ Download predictions (CSV)",
                output.to_csv(index=False).encode("utf-8"),
                file_name="predicted_scores.csv",
                mime="text/csv",
            )
