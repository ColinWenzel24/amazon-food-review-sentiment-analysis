"""Tab 3: unified model comparison table, charts, confusion matrices, and DistilBERT training curves."""

import streamlit as st

from controllers import insights_controller as ic
from frontend import charts
from frontend import components as ui

PCT = "{:.1%}"


def render() -> None:
    ui.section("Model comparison",
               "All models are scored on the same held-out test set: 9,253 reviews (20%), stratified split, "
               "seed 48. Macro averages weight each class equally, so they show how well the rare "
               "Neutral and Negative classes are handled.",
               kicker="How well does each model do?")

    table = ic.comparison_table()
    for spec in ic.missing_metrics():
        ui.alert("info", "ℹ", f"No saved test metrics for {spec.name} yet.",
                 f"Expected <code>results/{spec.metrics_file}</code>. For classical models, run "
                 "<code>python scripts/export_classical_models.py</code>.")
    if table.empty:
        return

    best = ic.best_model(table)
    baseline = table[table["Model"] == ic.BASELINE_NAME]
    if best is not None:
        tiles = [(best["Model"], "best model (highest macro-F1)"),
                 (PCT.format(best["Accuracy"]), f"{best['Model']} accuracy"),
                 (f"{best['F1 (macro)']:.3f}", f"{best['Model']} macro-F1")]
        if not baseline.empty:
            tiles.append((PCT.format(baseline.iloc[0]["Accuracy"]), "majority-class baseline accuracy"))
        ui.stat_row(tiles)

    st.write("")
    st.dataframe(
        table.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision (macro)", "Recall (macro)",
                                                  "F1 (macro)", "F1 (weighted)"]}, na_rep="—")
        .highlight_max(subset=["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)"],
                       props="font-weight:700; background-color:#FEF3E2;"),
        hide_index=True, width="stretch",
    )
    st.caption("Highlighted = best value in each column. Training time for DistilBERT was measured on a GPU "
               "(AMD Radeon RX 9070 XT); Logistic Regression on a CPU.")

    st.write("")
    st.markdown("##### Headline metrics")
    st.altair_chart(charts.metric_bars(ic.metric_long_format(), ic.model_colors()), width="content")
    st.caption("The baseline gets 78% accuracy just by always saying “Positive”, which is why accuracy alone "
               "is misleading here. Look at macro recall and F1.")

    pc_df = ic.per_class_f1()
    if not pc_df.empty:
        st.markdown("##### F1-score by sentiment class")
        st.altair_chart(charts.per_class_f1(pc_df, ic.model_colors()), width="stretch")
        st.caption("Neutral (3★) reviews are the hardest class for every model: they are rare (7.7% of the data) "
                   "and often mix praise and complaints.")

    st.markdown("##### Confusion matrices (test set)")
    images = ic.confusion_images()
    cols = st.columns(len(images))
    for col, (spec, path) in zip(cols, images):
        with col:
            if path:
                st.image(str(path), caption=f"{spec.name}: rows = true label, columns = predicted",
                         width="stretch")
            else:
                st.info(f"No confusion matrix image for {spec.name} yet.")

    _render_distilbert_training()


def _render_distilbert_training() -> None:
    st.divider()
    ui.section("Inside the deep model: DistilBERT training",
               "Fine-tuned from distilbert-base-uncased (67M parameters, 6 transformer layers) for 2 epochs "
               "with class-weighted cross-entropy loss. 10% of the training set was held out for validation.",
               kicker="Deep / modern model")
    m = ic.distilbert_metrics()
    if m:
        s = m.get("settings", {})
        ui.stat_row([(str(s.get("max_length", "—")), "max tokens per review"),
                     (str(s.get("batch_size", "—")), "batch size"),
                     (f"{s.get('learning_rate', 0):.0e}", "learning rate (AdamW)"),
                     (f"{m['training_minutes']:.1f} min", "training time (GPU)")])
        st.write("")

    curves = ic.distilbert_loss_curves()
    f1 = ic.distilbert_val_f1()
    if curves is None:
        st.info("Training log not found (results/distilbert_training_log.csv).")
        return
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Training vs. validation loss**")
        st.altair_chart(charts.loss_curves(curves), width="stretch")
    with c2:
        st.markdown("**Validation macro-F1 during training**")
        st.altair_chart(charts.val_f1_curve(f1), width="stretch")
    best = f1.loc[f1["val_macro_f1"].idxmax()]
    low = curves[curves["Curve"] == "Validation loss"].sort_values("Loss").iloc[0]
    st.caption(f"Validation loss is lowest at epoch {low['epoch']:.2f} and then rises slightly while training "
               "loss keeps falling, an early sign of overfitting. The checkpoint with the best validation "
               f"macro-F1 ({best['val_macro_f1']:.3f} at step {int(best['step']):,}) was kept for testing and "
               "is the one served in this app.")
