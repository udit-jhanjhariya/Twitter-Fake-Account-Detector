from fpdf import FPDF
from datetime import datetime
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")

TITLE = "Twitter Fake Profile Detection"
SUBTITLE = "Model Testing & Evaluation Report"
OUTPUT_PDF = "Twitter_Fake_Profile_Testing_Report.pdf"

class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 10, TITLE, align="C")
        self.ln(10)
        self.set_font("Helvetica", "", 14)
        self.cell(0, 10, SUBTITLE, align="C")
        self.ln(10)
        self.line(10, 30, 200, 30)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 10)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%d %b %Y %H:%M')}", 0, 0, "C")

if not os.path.exists("testing_results.txt"):
    raise FileNotFoundError("❌ Run test_model.py first to generate results!")

with open("testing_results.txt", "r") as f:
    results_text = f.read()

pdf = PDFReport()
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "Model Testing Summary", ln=True)
pdf.set_font("Helvetica", "", 12)
pdf.multi_cell(0, 8, results_text)
pdf.ln(10)

if os.path.exists("confusion_matrix.png"):
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Confusion Matrix Visualization", ln=True)
    pdf.ln(5)
    image = Image.open("confusion_matrix.png")
    width, height = image.size
    aspect = height / width
    pdf.image("confusion_matrix.png", x=25, w=160, h=160 * aspect)
else:
    pdf.set_font("Helvetica", "I", 12)
    pdf.cell(0, 10, "⚠️ Confusion matrix image not found.", ln=True)

pdf.output(OUTPUT_PDF)
print(f"✅ Report successfully generated: {OUTPUT_PDF}")
