import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.ensemble import IsolationForest
import re
import pdfplumber
import docx
import io

try:
    from thefuzz import fuzz
    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False

st.set_page_config(
    page_title="Invoice Sentinel | Fast Mode",
    layout="wide",
    page_icon="⚡"
)

st.markdown("""
<style>
    .stApp { background-color: #f4f6f9; }
    [data-testid="stSidebar"] { background-color: #2c3e50; }
    [data-testid="stSidebar"] * { color: #ecf0f1 !important; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="stMetricValue"] { color: #2980b9; font-weight: bold; }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

Base = declarative_base()
engine = create_engine('sqlite:///invoice_sentinel_fast.db')
Session = sessionmaker(bind=engine)
session = Session()

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    type = Column(String)
    ref_number = Column(String)
    vendor_name = Column(String)
    total_amount = Column(Float)
    date = Column(String)
    status = Column(String)

Base.metadata.create_all(engine)

def standardize_columns(df, target_name, possible_matches):
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    if target_name in df.columns:
        return df
    search_list = [m.lower() for m in possible_matches]
    search_list.append(target_name.lower())
    for col in df.columns:
        if col.lower() in search_list:
            return df.rename(columns={col: target_name})
    return df

def clean_dataframe(df, type_name):
    if type_name == 'invoice':
        df = standardize_columns(df, 'Invoice_ID', ['Invoice ID', 'Inv No', 'Invoice Number', 'id', 'invoice_id', 'Invoice #'])
        df = standardize_columns(df, 'PO_Number', ['PO Number', 'PO #', 'PO No', 'Reference PO', 'po_number', 'PO Reference', 'Reference No'])
        df = standardize_columns(df, 'Vendor', ['Vendor Name', 'Supplier', 'Biller', 'vendor'])
        df = standardize_columns(df, 'Amount', ['Line Total', 'Total Amount', 'Value', 'Amount', 'Total', 'Unit Price'])
    elif type_name == 'po':
        df = standardize_columns(df, 'PO_Number', ['PO Number', 'PO #', 'PO No', 'Purchase Order', 'po_number'])
        df = standardize_columns(df, 'Vendor', ['Vendor Name', 'Supplier', 'vendor'])
        df = standardize_columns(df, 'Amount', ['Line Total', 'Total Amount', 'Value', 'Amount', 'Total'])
    return df

def extract_text_from_file(uploaded_file):
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'pdf':
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        elif file_type in ['docx', 'doc']:
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_type in ['png', 'jpg', 'jpeg']:
            return "IMAGE_SKIPPED"
    except Exception as e:
        return f"Error: {e}"
    return text

def smart_parse_text(text, doc_type):
    data = {}
    text_lower = text.lower()
    
    if doc_type == 'invoice':
        pattern = r'(?:invoice\s*(?:no|number|#|id)|inv\.|inv)\s*[:.]?\s*([a-zA-Z0-9\-_]+)'
        match = re.search(pattern, text_lower)
        data['Invoice_ID'] = match.group(1).upper() if match else "UNKNOWN-INV"
        
        po_pattern = r'(?:po|purchase\s*order|reference|ref\.?)\s*(?:no\.?|number|#|id|ref\.?|reference)?\s*[:.-]?\s*([a-zA-Z0-9\-_]+)'
        po_match = re.search(po_pattern, text_lower)
        data['PO_Number'] = po_match.group(1).upper() if po_match else "MISSING-PO"

    elif doc_type == 'po':
        pattern = r'(?:po|purchase\s*order|order)\s*(?:no\.?|number|#|id)?\s*[:.-]?\s*([a-zA-Z0-9\-_]+)'
        match = re.search(pattern, text_lower)
        data['PO_Number'] = match.group(1).upper() if match else "UNKNOWN-PO"

    amounts = re.findall(r'[₹$€]?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
    if amounts:
        clean_amounts = []
        for a in amounts:
            try:
                clean_amounts.append(float(a.replace(',', '')))
            except: pass
        data['Amount'] = max(clean_amounts) if clean_amounts else 0.0
    else:
        data['Amount'] = 0.0

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    vendor_pattern = r'(?:vendor|supplier|from|bill\s*to)\s*[:.]?\s*(.*)'
    vendor_match = re.search(vendor_pattern, text_lower)
    if vendor_match:
        data['Vendor'] = vendor_match.group(1).title().strip()
    else:
        data['Vendor'] = lines[0] if lines else "Unknown Vendor"

    return data

def match_logic_only(df_invoices, df_pos):
    results = []
    
    if 'PO_Number' not in df_invoices.columns:
        st.error("❌ Error: 'PO_Number' column missing in Invoice Data.")
        return pd.DataFrame()
    if 'PO_Number' not in df_pos.columns:
        st.error("❌ Error: 'PO_Number' column missing in PO Data.")
        return pd.DataFrame()

    for _, inv in df_invoices.iterrows():
        match_status = "Unmatched"
        notes = []
        inv_po = str(inv.get('PO_Number', '')).strip().upper()
        po_match = df_pos[df_pos['PO_Number'].astype(str).str.strip().str.upper() == inv_po]
        
        if po_match.empty:
            match_status = "Missing PO"
            notes.append("PO ID not found.")
        else:
            po = po_match.iloc[0]
            try:
                inv_amt = float(inv['Amount'])
                po_amt = float(po['Amount'])
                diff = inv_amt - po_amt
            except:
                inv_amt, po_amt, diff = 0, 0, 0
            
            if diff > 0.01:
                match_status = "Discrepancy"
                notes.append(f"Exceeds PO by ₹{diff:,.2f}")
            elif diff < -0.01:
                match_status = "Underbilled"
            else:
                match_status = "Matched"
            
            inv_vnd = str(inv.get('Vendor', '')).lower()
            po_vnd = str(po.get('Vendor', '')).lower()
            
            is_match = False
            if HAS_FUZZ:
                if fuzz.partial_ratio(inv_vnd, po_vnd) > 70: is_match = True
            elif inv_vnd in po_vnd or po_vnd in inv_vnd:
                is_match = True
                
            if not is_match:
                if match_status == "Matched": match_status = "Vendor Mismatch"

        results.append({
            "Invoice_ID": inv.get('Invoice_ID', 'N/A'),
            "PO_Number": inv.get('PO_Number', 'N/A'),
            "Vendor": inv.get('Vendor', 'Unknown'),
            "Amount": inv.get('Amount', 0),
            "Match_Status": match_status,
            "Notes": "; ".join(notes)
        })
    return pd.DataFrame(results)

def anomaly_logic_only(df_matched):
    if df_matched.empty or len(df_matched) < 5:
        df_matched['Anomaly_Status'] = "Insufficient Data"
        return df_matched
    model = IsolationForest(contamination=0.1, random_state=42)
    X = df_matched['Amount'].values.reshape(-1, 1)
    df_matched['Anomaly_Score'] = model.fit_predict(X)
    df_matched['Anomaly_Status'] = df_matched['Anomaly_Score'].apply(lambda x: '⚠️ Potential Fraud' if x == -1 else '✅ Normal')
    return df_matched

def sidebar_nav():
    st.sidebar.title("⚡ Invoice Sentinel Lite")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Document Upload (PDF/DOCX)", "Reconciliation"])
    st.sidebar.info("Mode: Fast Cloud Deploy\nNo Image OCR")
    return page

def page_dashboard():
    st.title("Executive Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Processed Documents", "1,240", "+12%")
    col2.metric("Discrepancies", "45", "-5%")
    col3.metric("Fraud Alerts", "8", "High Risk", delta_color="inverse")
    col4.metric("Value at Risk", "₹ 12,45,000", "Alert")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Invoice Matching Status")
        labels = ['Matched', 'Discrepancy', 'Missing PO']
        values = [850, 120, 50]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Risk Heatmap")
        df_viz = pd.DataFrame({'Amount': np.random.exponential(50000, 100), 'Risk': np.random.uniform(0, 100, 100)})
        fig2 = px.scatter(df_viz, x="Amount", y="Risk", color="Risk", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig2, use_container_width=True)

def page_entry():
    st.title("Document Ingestion (PDF/Word/CSV)")
    
    tab1, tab2 = st.tabs(["📄 Document Parser", "📂 CSV Batch Upload"])
    
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Smart Document Parser (No Images)")
        st.info("Upload PDF or Word files only.")
        
        c1, c2 = st.columns(2)
        inv_files = c1.file_uploader("Upload Invoices (PDF/DOCX)", type=['pdf', 'docx', 'doc'], accept_multiple_files=True)
        po_files = c2.file_uploader("Upload POs (PDF/DOCX)", type=['pdf', 'docx', 'doc'], accept_multiple_files=True)
            
        if st.button("🚀 Extract & Analyze"):
            if not inv_files or not po_files:
                st.error("Please upload at least one Invoice and one PO.")
            else:
                extracted_inv = []
                for f in inv_files:
                    raw_text = extract_text_from_file(f)
                    if raw_text == "IMAGE_SKIPPED":
                        st.warning(f"Skipped {f.name} (Image/OCR disabled in Lite Mode)")
                        continue
                    data = smart_parse_text(raw_text, 'invoice')
                    data['Source_File'] = f.name
                    extracted_inv.append(data)
                
                extracted_po = []
                for f in po_files:
                    raw_text = extract_text_from_file(f)
                    data = smart_parse_text(raw_text, 'po')
                    data['Source_File'] = f.name
                    extracted_po.append(data)
                
                if extracted_inv and extracted_po:
                    df_inv = pd.DataFrame(extracted_inv)
                    df_po = pd.DataFrame(extracted_po)
                    st.session_state['df_inv'] = df_inv
                    st.session_state['df_po'] = df_po
                    st.success("Extraction Complete!")
                    st.write("Invoices:", df_inv.head())
                    st.write("POs:", df_po.head())
                else:
                    st.error("No valid text found in documents.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Legacy CSV Upload")
        c1, c2 = st.columns(2)
        inv_files = c1.file_uploader("Upload Invoice CSVs", type=['csv'], accept_multiple_files=True, key='csv_inv')
        po_files = c2.file_uploader("Upload PO CSVs", type=['csv'], accept_multiple_files=True, key='csv_po')
        
        if inv_files and po_files:
            try:
                df_inv = pd.concat((pd.read_csv(f, encoding='utf-8-sig') for f in inv_files), ignore_index=True)
                df_po = pd.concat((pd.read_csv(f, encoding='utf-8-sig') for f in po_files), ignore_index=True)
                df_inv = clean_dataframe(df_inv, 'invoice')
                df_po = clean_dataframe(df_po, 'po')
                
                if 'Invoice_ID' in df_inv.columns and df_inv['Invoice_ID'].duplicated().any():
                     if 'Amount' in df_inv.columns:
                        try:
                            df_inv['Amount'] = df_inv['Amount'].astype(str).str.replace(',', '').astype(float)
                            cols = ['Invoice_ID', 'PO_Number', 'Vendor']
                            available_cols = [c for c in cols if c in df_inv.columns]
                            df_inv = df_inv.groupby(available_cols, as_index=False)['Amount'].sum()
                        except: pass

                st.session_state['df_inv'] = df_inv
                st.session_state['df_po'] = df_po
                st.success(f"CSVs Loaded & Cleaned.")
            except Exception as e:
                st.error(f"CSV Error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

def page_analysis():
    st.title("Reconciliation & Analysis")
    if 'df_inv' not in st.session_state or 'df_po' not in st.session_state:
        st.warning("No data found. Please upload documents first.")
        return
    df_inv = st.session_state['df_inv']
    df_po = st.session_state['df_po']
    st.markdown("### Step 1: Matching Results")
    if st.button("Run Matching Engine"):
        matched = match_logic_only(df_inv, df_po)
        if not matched.empty:
            st.session_state['matched'] = matched
            def color_status(val):
                color = 'green' if val == 'Matched' else 'red'
                return f'color: {color}; font-weight: bold'
            st.dataframe(matched.style.applymap(color_status, subset=['Match_Status']))
        else:
            st.warning("No matches found.")

    st.markdown("---")
    st.markdown("### Step 2: AI Fraud Detection")
    if 'matched' in st.session_state:
        if st.button("Run Anomaly Detection"):
            analyzed = anomaly_logic_only(st.session_state['matched'])
            st.write(analyzed)

def main():
    page = sidebar_nav()
    if page == "Dashboard": page_dashboard()
    elif page == "Document Upload (PDF/DOCX)": page_entry()
    elif page == "Reconciliation": page_analysis()

if __name__ == "__main__":
    main()
