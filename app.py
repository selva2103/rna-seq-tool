import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Page setup
st.set_page_config(page_title="RNA-Seq Report Generator", layout="wide")

st.title("🧬 RNA-Seq Report Generator")
st.write("Updated Version Running ✅")

uploaded_file = st.file_uploader("Upload CSV", type=["csv", "txt"])

if uploaded_file:

    # ================= FILE READING (ROBUST) =================
    try:
        df = pd.read_csv(uploaded_file)
    except:
        try:
            df = pd.read_csv(uploaded_file, encoding="latin1")
        except:
            df = pd.read_csv(uploaded_file, encoding="latin1", comment='!')

    st.subheader("📄 Data Preview")
    st.write(df.head())

    # ================= CHECK REQUIRED COLUMNS =================
    if "log2FoldChange" not in df.columns or "padj" not in df.columns:
        st.error("❌ This file is not a DESeq2 result file.")

        st.info("""
        👉 Your file looks like a GEO expression matrix.

        ✔ This tool requires:
        - log2FoldChange
        - padj

        👉 Please upload processed RNA-Seq results (DESeq2 output).
        """)

    else:

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

        # ================= VOLCANO PLOT =================
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

        top_genes = df[df["padj"] < 0.05].sort_values("padj").head(10)

        for _, row in top_genes.iterrows():
            plt.text(
                row["log2FoldChange"],
                row["-log10(padj)"],
                str(row[gene_col]),
                fontsize=8
            )

        plt.axvline(x=1, linestyle="--")
        plt.axvline(x=-1, linestyle="--")
        plt.axhline(y=-np.log10(0.05), linestyle="--")

        plt.xlabel("log2FoldChange")
        plt.ylabel("-log10(padj)")
        plt.title("Volcano Plot")

        plt.legend()

        plot_path = "volcano.png"
        plt.savefig(plot_path)

        st.pyplot(plt)

        # ================= PDF SECTION =================

        st.subheader("📄 Download Reports")

        # -------- FREE PDF --------
        if st.button("🆓 Download Free PDF"):

            free_filename = f"Free_Report_{np.random.randint(1000)}.pdf"

            doc = SimpleDocTemplate(free_filename)
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

            with open(free_filename, "rb") as f:
                st.download_button("Download Free PDF", f, file_name=free_filename)

            st.success("Free PDF Generated!")

        # -------- PREMIUM PDF --------
        st.subheader("💎 Premium Report (₹100)")

        access_code = st.text_input("Enter Access Code")

        if access_code == "BIO100":

            if st.button("💎 Download Premium PDF"):

                premium_filename = f"Premium_Report_{np.random.randint(1000)}.pdf"

                doc = SimpleDocTemplate(premium_filename)
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
                    "Upregulated genes indicate activation of biological pathways, while downregulated genes suggest suppression. "
                    "Key genes such as TP53, CDK1, and BRCA1 may be biologically important.",
                    styles["Normal"]
                ))

                elements.append(Spacer(1, 12))
                elements.append(Paragraph("Top Significant Genes:", styles["Heading2"]))

                for _, row in top_genes.iterrows():
                    elements.append(Paragraph(str(row[gene_col]), styles["Normal"]))

                elements.append(Spacer(1, 12))
                elements.append(Image(plot_path))

                doc.build(elements)

                with open(premium_filename, "rb") as f:
                    st.download_button("Download Premium PDF", f, file_name=premium_filename)

                st.success("Premium PDF Generated!")

        else:
            st.info("Enter valid access code (after payment)")

else:
    st.info("Upload a CSV file to begin")
