import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet

# Page config
st.set_page_config(page_title="RNA-Seq Report Generator", layout="wide")

# Title
st.title("🧬 RNA-Seq Report Generator")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.write(df.head())

    # Check required columns
    if "log2FoldChange" in df.columns and "padj" in df.columns:

        df = df.dropna(subset=["log2FoldChange", "padj"])

        # Categorization
        df["Category"] = "Not Significant"
        df.loc[(df["padj"] < 0.05) & (df["log2FoldChange"] > 1), "Category"] = "Upregulated"
        df.loc[(df["padj"] < 0.05) & (df["log2FoldChange"] < -1), "Category"] = "Downregulated"

        # Summary counts
        up = (df["Category"] == "Upregulated").sum()
        down = (df["Category"] == "Downregulated").sum()

        st.subheader("📊 Summary")
        st.write(f"Upregulated: {up}")
        st.write(f"Downregulated: {down}")

        # Detect gene column
        gene_col = None
        possible_cols = ["gene", "Gene", "gene_id", "GeneID", "symbol", "Symbol"]

        for col in possible_cols:
            if col in df.columns:
                gene_col = col
                break

        # Fallback if no gene column
        if gene_col is None:
            df["gene_temp"] = df.index.astype(str)
            gene_col = "gene_temp"
            st.warning("No gene column found. Using row index as labels.")

        # Volcano plot
        st.subheader("🌋 Volcano Plot")

        df["-log10(padj)"] = -np.log10(df["padj"])

        plt.figure()

        # Plot categories
        for category, color in zip(
            ["Upregulated", "Downregulated", "Not Significant"],
            ["red", "blue", "grey"]
        ):
            subset = df[df["Category"] == category]
            plt.scatter(
                subset["log2FoldChange"],
                subset["-log10(padj)"],
                c=color,
                label=category
            )

        # Label top 10 genes
        top_genes = df[df["padj"] < 0.05].sort_values("padj").head(10)

        for _, row in top_genes.iterrows():
            plt.text(
                row["log2FoldChange"],
                row["-log10(padj)"],
                str(row[gene_col]),
                fontsize=8
            )

        # Threshold lines
        plt.axvline(x=1, linestyle="--")
        plt.axvline(x=-1, linestyle="--")
        plt.axhline(y=-np.log10(0.05), linestyle="--")

        plt.xlabel("log2FoldChange")
        plt.ylabel("-log10(padj)")
        plt.title("Volcano Plot")

        plt.legend()

        # Save plot
        plot_path = "volcano.png"
        plt.savefig(plot_path)

        st.pyplot(plt)

        # PDF Section
        st.subheader("📄 Download Report")

        if st.button("Generate PDF Report"):

            doc = SimpleDocTemplate("RNASeq_Report.pdf")
            styles = getSampleStyleSheet()

            elements = []

            elements.append(Paragraph("RNA-Seq Analysis Report", styles["Title"]))
            elements.append(Paragraph(f"Upregulated genes: {up}", styles["Normal"]))
            elements.append(Paragraph(f"Downregulated genes: {down}", styles["Normal"]))

            elements.append(Paragraph("Top Significant Genes:", styles["Heading2"]))

            for _, row in top_genes.iterrows():
                elements.append(Paragraph(str(row[gene_col]), styles["Normal"]))

            elements.append(Image(plot_path))

            doc.build(elements)

            with open("RNASeq_Report.pdf", "rb") as f:
                st.download_button("Download PDF", f, file_name="RNASeq_Report.pdf")

            st.success("PDF Generated Successfully!")

    else:
        st.error("CSV must contain 'log2FoldChange' and 'padj' columns")

else:
    st.info("Upload a CSV file to begin")
