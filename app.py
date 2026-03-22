import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Page setup
st.set_page_config(page_title="RNA-Seq Report Generator", layout="wide")

st.title("🧬 RNA-Seq Report Generator")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.write(df.head())

    if "log2FoldChange" in df.columns and "padj" in df.columns:

        df = df.dropna(subset=["log2FoldChange", "padj"])

        # Categorization
        df["Category"] = "Not Significant"
        df.loc[(df["padj"] < 0.05) & (df["log2FoldChange"] > 1), "Category"] = "Upregulated"
        df.loc[(df["padj"] < 0.05) & (df["log2FoldChange"] < -1), "Category"] = "Downregulated"

        # Summary
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

        if gene_col is None:
            df["gene_temp"] = df.index.astype(str)
            gene_col = "gene_temp"
            st.warning("No gene column found. Using row index as labels.")

        # Volcano plot
        st.subheader("🌋 Volcano Plot")

        df["-log10(padj)"] = -np.log10(df["padj"])

        plt.figure()

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

        # Label top genes
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

        # ================= PDF SECTION =================

        st.subheader("📄 Download Reports")

        # FREE PDF
        if st.button("🆓 Download Free PDF"):

            doc = SimpleDocTemplate("Free_Report.pdf")
            styles = getSampleStyleSheet()

            elements = []

            elements.append(Paragraph("RNA-Seq Basic Report", styles["Title"]))
            elements.append(Spacer(1, 10))

            elements.append(Paragraph(f"Upregulated genes: {up}", styles["Normal"]))
            elements.append(Paragraph(f"Downregulated genes: {down}", styles["Normal"]))
            elements.append(Spacer(1, 10))

            elements.append(Paragraph("Top Significant Genes:", styles["Heading2"]))

            for _, row in top_genes.iterrows():
                elements.append(Paragraph(str(row[gene_col]), styles["Normal"]))

            elements.append(Spacer(1, 10))
            elements.append(Image(plot_path))

            doc.build(elements)

            with open("Free_Report.pdf", "rb") as f:
                st.download_button("Download Free PDF", f, file_name="Free_Report.pdf")

            st.success("Free PDF Generated!")

        # PREMIUM SECTION
        st.subheader("💎 Premium Report (₹100)")

        access_code = st.text_input("Enter Access Code")

        if access_code == "BIO100":

            if st.button("💎 Download Premium PDF"):

                doc = SimpleDocTemplate("Premium_Report.pdf")
                styles = getSampleStyleSheet()

                elements = []

                elements.append(Paragraph("RNA-Seq Premium Analysis Report", styles["Title"]))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph(f"Upregulated genes: {up}", styles["Normal"]))
                elements.append(Paragraph(f"Downregulated genes: {down}", styles["Normal"]))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("Biological Interpretation:", styles["Heading2"]))

                elements.append(Paragraph(
                    f"This dataset reveals {up} upregulated and {down} downregulated genes. "
                    "Upregulated genes suggest activation of biological pathways under treatment conditions, "
                    "while downregulated genes indicate suppressed pathways. "
                    "Key genes such as TP53, CDK1, and BRCA1 may be involved in regulatory or disease-related pathways.",
                    styles["Normal"]
                ))

                elements.append(Spacer(1, 12))

                elements.append(Paragraph("Top Significant Genes:", styles["Heading2"]))

                for _, row in top_genes.iterrows():
                    elements.append(Paragraph(str(row[gene_col]), styles["Normal"]))

                elements.append(Spacer(1, 12))
                elements.append(Image(plot_path))

                doc.build(elements)

                with open("Premium_Report.pdf", "rb") as f:
                    st.download_button("Download Premium PDF", f, file_name="Premium_Report.pdf")

                st.success("Premium PDF Generated!")

        else:
            st.info("Enter valid access code (after payment)")

    else:
        st.error("CSV must contain 'log2FoldChange' and 'padj' columns")

else:
    st.info("Upload a CSV file to begin")
