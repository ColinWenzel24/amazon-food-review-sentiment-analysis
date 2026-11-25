"""Altair charts. Colors: models use a validated categorical pair (teal / dark orange) with the
baseline in neutral gray; sentiment uses a red-gray-green diverging scheme with gray (Neutral)
at the midpoint. Every chart has tooltips and text labels so identity is never color-only."""

import altair as alt
import pandas as pd

from backend.config import LABELS

SENTIMENT_COLORS = {"Negative": "#B12704", "Neutral": "#8D9096", "Positive": "#067D62"}
AXIS = dict(labelColor="#565959", titleColor="#565959", gridColor="#EAEDED", domainColor="#D5D9D9",
            tickColor="#D5D9D9", labelFont="Arial", titleFont="Arial")


def _style(chart: alt.Chart) -> alt.Chart:
    return (chart.configure_axis(**AXIS)
            .configure_view(stroke=None)
            .configure_legend(labelColor="#0F1111", titleColor="#565959", orient="top")
            .configure_title(color="#0F1111", font="Arial", fontSize=14, anchor="start"))


def _scale(mapping: dict[str, str], domain: list[str]) -> alt.Scale:
    return alt.Scale(domain=domain, range=[mapping[d] for d in domain])


def metric_bars(long_df: pd.DataFrame, colors: dict[str, str]) -> alt.Chart:
    models = list(dict.fromkeys(long_df["Model"]))
    base = alt.Chart(long_df).encode(
        x=alt.X("Model:N", sort=models, axis=None),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None, tickCount=5)),
        color=alt.Color("Model:N", scale=_scale(colors, models), legend=alt.Legend(title=None)),
        tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width={"band": 0.85})
    labels = base.mark_text(dy=-7, fontSize=10).encode(color=alt.value("#0F1111"), text=alt.Text("Score:Q", format=".0%"))
    chart = (bars + labels).properties(width=150, height=240).facet(
        column=alt.Column("Metric:N", sort=["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
                          header=alt.Header(title=None, labelColor="#0F1111", labelFontWeight="bold",
                                            labelFontSize=12))
    ).resolve_scale(x="independent")
    return _style(chart)


def per_class_f1(df: pd.DataFrame, colors: dict[str, str]) -> alt.Chart:
    models = list(dict.fromkeys(df["Model"]))
    base = alt.Chart(df).encode(
        y=alt.Y("Sentiment:N", sort=LABELS, title=None),
        yOffset=alt.YOffset("Model:N", sort=models),
        x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="F1-score")),
        color=alt.Color("Model:N", scale=_scale(colors, models), legend=alt.Legend(title=None)),
        tooltip=["Model", "Sentiment", alt.Tooltip("F1:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f"),
                 alt.Tooltip("Recall:Q", format=".3f"), "Support"],
    )
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height={"band": 0.9})
    labels = base.mark_text(align="left", dx=4, fontSize=11).encode(
        color=alt.value("#0F1111"), text=alt.Text("F1:Q", format=".2f"))
    return _style((bars + labels).properties(height=230))


def loss_curves(long_df: pd.DataFrame) -> alt.Chart:
    domain = ["Training loss", "Validation loss"]
    colors = {"Training loss": "#007EA8", "Validation loss": "#C45500"}
    hover = alt.selection_point(fields=["epoch"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long_df).encode(
        x=alt.X("epoch:Q", title="Epoch", scale=alt.Scale(domain=[0, 2.05])),
        y=alt.Y("Loss:Q", title="Cross-entropy loss", scale=alt.Scale(zero=False)),
        color=alt.Color("Curve:N", scale=_scale(colors, domain), legend=alt.Legend(title=None)),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(filled=True, size=60).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0.35)),
        tooltip=[alt.Tooltip("epoch:Q", title="Epoch"), "step", "Curve", alt.Tooltip("Loss:Q", format=".4f")],
    ).add_params(hover)
    rule = alt.Chart(long_df).mark_rule(color="#D5D9D9").encode(x="epoch:Q").transform_filter(hover)
    return _style((lines + points + rule).properties(height=300))


def val_f1_curve(df: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(df).encode(
        x=alt.X("epoch:Q", title="Epoch"),
        y=alt.Y("val_macro_f1:Q", title="Validation macro-F1", scale=alt.Scale(zero=False),
                axis=alt.Axis(format=".2f")),
        tooltip=[alt.Tooltip("epoch:Q", title="Epoch"), "step",
                 alt.Tooltip("val_macro_f1:Q", title="Val macro-F1", format=".4f")],
    )
    best = df.loc[df["val_macro_f1"].idxmax()]
    marker = alt.Chart(pd.DataFrame([best])).mark_point(size=140, shape="diamond", filled=True,
                                                        color="#C45500").encode(x="epoch:Q", y="val_macro_f1:Q")
    note = alt.Chart(pd.DataFrame([best])).mark_text(dy=-14, fontSize=11, color="#0F1111").encode(
        x="epoch:Q", y="val_macro_f1:Q", text=alt.value("best checkpoint (kept)"))
    chart = base.mark_line(strokeWidth=2, color="#007EA8") + base.mark_point(filled=True, size=50,
                                                                             color="#007EA8") + marker + note
    return _style(chart.properties(height=300))


def sentiment_distribution(counts: dict[str, int], title: str = "") -> alt.Chart:
    total = sum(counts.values())
    df = pd.DataFrame([{"Sentiment": k, "Reviews": v, "Share": v / total} for k, v in counts.items()])
    base = alt.Chart(df).encode(
        x=alt.X("Sentiment:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Reviews:Q", title="Reviews"),
        color=alt.Color("Sentiment:N", scale=_scale(SENTIMENT_COLORS, LABELS), legend=None),
        tooltip=["Sentiment", alt.Tooltip("Reviews:Q", format=","), alt.Tooltip("Share:Q", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width={"band": 0.6})
    labels = base.mark_text(dy=-8, fontSize=12).encode(
        color=alt.value("#0F1111"), text=alt.Text("label:N")).transform_calculate(
        label="format(datum.Reviews, ',') + '  (' + format(datum.Share, '.1%') + ')'")
    return _style((bars + labels).properties(height=280, title=title))


def star_distribution(score_counts: dict[str, int]) -> alt.Chart:
    df = pd.DataFrame([{"Stars": f"{k} ★", "n": int(k), "Reviews": v} for k, v in score_counts.items()])
    df["Sentiment"] = df["n"].map({1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"})
    base = alt.Chart(df).encode(
        y=alt.Y("Stars:N", sort="descending", title=None),
        x=alt.X("Reviews:Q", title="Reviews"),
        color=alt.Color("Sentiment:N", scale=_scale(SENTIMENT_COLORS, LABELS), legend=alt.Legend(title="Label")),
        tooltip=["Stars", "Sentiment", alt.Tooltip("Reviews:Q", format=",")],
    )
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height={"band": 0.7})
    labels = base.mark_text(align="left", dx=4, fontSize=11).encode(
        color=alt.value("#0F1111"), text=alt.Text("Reviews:Q", format=","))
    return _style((bars + labels).properties(height=230))


def length_histogram(hist: dict[str, int]) -> alt.Chart:
    order = list(hist.keys())
    df = pd.DataFrame([{"Words": k, "Reviews": v} for k, v in hist.items()])
    chart = alt.Chart(df).mark_bar(color="#007EA8", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Words:N", sort=order, title="Review length (words)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Reviews:Q", title="Reviews"),
        tooltip=[alt.Tooltip("Words:N", title="Words"), alt.Tooltip("Reviews:Q", format=",")],
    )
    return _style(chart.properties(height=260))


def top_words_chart(words: list[list], label: str) -> alt.Chart:
    df = pd.DataFrame(words[:10], columns=["Word", "Weight"])
    chart = alt.Chart(df).mark_bar(color=SENTIMENT_COLORS[label], cornerRadiusTopRight=4,
                                   cornerRadiusBottomRight=4).encode(
        y=alt.Y("Word:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("Weight:Q", title="Logistic Regression weight"),
        tooltip=["Word", alt.Tooltip("Weight:Q", format=".2f")],
    )
    labels = chart.mark_text(align="left", dx=4, fontSize=11, color="#0F1111").encode(
        text=alt.Text("Weight:Q", format=".2f"))
    return _style((chart + labels).properties(height=300, title=f"Top 10 {label.lower()} words"))


def batch_distribution(df: pd.DataFrame, colors: dict[str, str]) -> alt.Chart:
    models = list(dict.fromkeys(df["Model"]))
    base = alt.Chart(df).encode(
        x=alt.X("Sentiment:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Model:N", sort=models),
        y=alt.Y("Reviews:Q", title="Reviews"),
        color=alt.Color("Model:N", scale=_scale(colors, models), legend=alt.Legend(title=None)),
        tooltip=["Model", "Sentiment", "Reviews"],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    labels = base.mark_text(dy=-7, fontSize=11).encode(color=alt.value("#0F1111"), text="Reviews:Q")
    return _style((bars + labels).properties(height=280))
