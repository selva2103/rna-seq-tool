import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("RNA-Seq Report Generator")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Data Preview")
    st.write(df.head())

    if "log2FoldChange" in df.columns and "padj" in df.columns:

        df = df.dropna()

        df["Category"] = "Not Significant"
        df.loc[(df["padj"] < 0.05) & (df["log2FoldChange"] > 1), "Category"] = "Upregulated"
        df.loc[(df["padj"] < 0.05) & (df["log2FoldChange"] < -1), "Category"] = "Downregulated"

        up = (df["Category"] == "Upregulated").sum()
        down = (df["Category"] == "Downregulated").sum()

        st.write("Upregulated:", up)
        st.write("Downregulated:", down)

        df["-log10(padj)"] = -np.log10(df["padj"])

        plt.scatter(df["log2FoldChange"], df["-log10(padj)"])
        plt.xlabel("log2FoldChange")
        plt.ylabel("-log10(padj)")

        st.pyplot(plt)

    else:
        st.error("CSV must contain log2FoldChange and padj")
