import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.ensemble import IsolationForest

try:
    from thefuzz import fuzz
    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False

st.set_page_config(
    page_title="Invoice Sentinel | Enterprise Edition",
    layout="wide",
    page_icon="🛡️"
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
engine = create_engine('sqlite:///invoice_sentinel_simple.db')
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

def process_uploaded_data(df, id_col, file_type):
    if id_col not in df.columns:
        st.error(f"❌ Missing Column: The {file_type} file is missing '{id_col}'. Found: {list(df.columns)}")
        return pd.DataFrame()

    try:
        df['Amount'] = df['Amount'].astype(str).str.replace(',', '').str.replace('₹', '').astype(float)
    except:
        df['Amount'] = 0.0

    if df[id_col].duplicated().any():
        st.info(f"ℹ️ {file_type}: Aggregating multiple line items automatically.")
        group_cols = [id_col]
        if 'Vendor' in df.columns: group_cols.append('Vendor')
        if 'PO_Number' in df.columns and id_col != 'PO_Number': group_cols.append('PO_Number')
        df = df.groupby(group_cols, as_index=False)['Amount'].sum()
        
    return df

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
            notes.append("PO ID not found in database.")
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
                notes.append(f"Invoice exceeds PO by ₹{diff:,.2f}")
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
                notes.append(f"Vendor mismatch detected.")
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
    st.sidebar.title("🛡️ Invoice Sentinel")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Data Ingestion", "Reconciliation"])
    st.sidebar.markdown("---")
    st.sidebar.info("Mode: Fast (CSV Only)")
    return page

def page_dashboard():
    st.title("Executive Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Processed Transactions", "1,240", "+12%")
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
    st.title("Data Ingestion")
    
    tab1, tab2 = st.tabs(["📂 Batch CSV Upload", "✍️ Manual Entry"])
    
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Bulk Processing (CSV)")
        st.caption("Upload CSVs. System auto-cleans headers and sums line items.")
        
        c1, c2 = st.columns(2)
        inv_files = c1.file_uploader("Upload Invoice CSVs", type=['csv'], accept_multiple_files=True, key="inv_up")
        po_files = c2.file_uploader("Upload PO CSVs", type=['csv'], accept_multiple_files=True, key="po_up")
        
        if inv_files and po_files:
            try:
                df_inv = pd.concat((pd.read_csv(f, encoding='utf-8-sig') for f in inv_files), ignore_index=True)
                df_po = pd.concat((pd.read_csv(f, encoding='utf-8-sig') for f in po_files), ignore_index=True)
                
                df_inv = clean_dataframe(df_inv, 'invoice')
                df_po = clean_dataframe(df_po, 'po')
                
                df_inv = process_uploaded_data(df_inv, 'Invoice_ID', 'Invoice')
                df_po = process_uploaded_data(df_po, 'PO_Number', 'PO')
                
                if not df_inv.empty and not df_po.empty:
                    st.success(f"✅ Loaded: {len(df_inv)} Invoices and {len(df_po)} POs.")
                    if st.button("Save & Proceed to Analysis", key="btn_upload"):
                        st.session_state['df_inv'] = df_inv
                        st.session_state['df_po'] = df_po
                        st.success("Data Saved!")
            except Exception as e:
                st.error(f"Error processing file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Supplier & Vendor Relationship Entry")
        col_inv, col_po = st.columns(2)
        
        with col_inv:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📄 Vendor Invoice (Supplier)")
            inv_num = st.text_input("Invoice Number", value="INV-2024-001")
            inv_ref_po = st.text_input("Reference PO #", value="PO-998877")
            inv_vendor = st.text_input("Vendor Name", value="Acme Corp")
            inv_date = st.date_input("Invoice Date", key="date_inv")
            
            st.markdown("**Line Items**")
            inv_data = {'Product Name': ['Industrial Motor'], 'Quantity': [2], 'Unit Price (₹)': [25000.0]}
            inv_df_input = pd.DataFrame(inv_data)
            edited_inv = st.data_editor(inv_df_input, num_rows="dynamic", key="editor_inv")
            
            if not edited_inv.empty:
                edited_inv['Line Total'] = edited_inv['Quantity'] * edited_inv['Unit Price (₹)']
                inv_total = edited_inv['Line Total'].sum()
            else:
                inv_total = 0.0
            st.metric("Total Invoice Value", f"₹ {inv_total:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_po:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🏢 Internal Purchase Order (Buyer)")
            po_num = st.text_input("PO Number", value="PO-998877")
            po_vendor = st.text_input("Supplier Name", value="Acme Corp")
            po_date = st.date_input("PO Date", key="date_po")
            
            st.markdown("**Line Items**")
            po_data = {'Product Name': ['Industrial Motor'], 'Quantity': [2], 'Unit Price (₹)': [25000.0]}
            po_df_input = pd.DataFrame(po_data)
            edited_po = st.data_editor(po_df_input, num_rows="dynamic", key="editor_po")
            
            if not edited_po.empty:
                edited_po['Line Total'] = edited_po['Quantity'] * edited_po['Unit Price (₹)']
                po_total = edited_po['Line Total'].sum()
            else:
                po_total = 0.0
            st.metric("Total PO Value", f"₹ {po_total:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Submit & Run Reconciliation", type="primary"):
            try:
                txn_inv = Transaction(type='INVOICE', ref_number=inv_num, vendor_name=inv_vendor, total_amount=float(inv_total), date=str(inv_date), status='Pending')
                session.add(txn_inv)
                txn_po = Transaction(type='PO', ref_number=po_num, vendor_name=po_vendor, total_amount=float(po_total), date=str(po_date), status='Created')
                session.add(txn_po)
                session.commit()
                
                df_manual_inv = pd.DataFrame([{'Invoice_ID': inv_num, 'PO_Number': inv_ref_po, 'Vendor': inv_vendor, 'Amount': inv_total}])
                df_manual_po = pd.DataFrame([{'PO_Number': po_num, 'Vendor': po_vendor, 'Amount': po_total}])
                
                st.session_state['df_inv'] = df_manual_inv
                st.session_state['df_po'] = df_manual_po
                st.success("Transaction Saved! Data loaded for Analysis.")
            except Exception as e:
                st.error(f"Database Error: {e}")

def page_analysis():
    st.title("Reconciliation & Analysis")
    if 'df_inv' not in st.session_state or 'df_po' not in st.session_state:
        st.warning("No data found. Please upload files or enter data manually.")
        return

    df_inv = st.session_state['df_inv']
    df_po = st.session_state['df_po']

    st.markdown("### Step 1: Rule-Based Matching")
    if st.button("Run Matching Engine"):
        matched_results = match_logic_only(df_inv, df_po)
        st.session_state['matched_results'] = matched_results
        
        st.write(f"Processed {len(matched_results)} transactions.")
        def color_status(val):
            color = 'green' if val == 'Matched' else 'red'
            return f'color: {color}; font-weight: bold'
        st.dataframe(matched_results.style.applymap(color_status, subset=['Match_Status']))

    st.markdown("---")
    st.markdown("### Step 2: AI Fraud Detection")
    if 'matched_results' in st.session_state:
        if st.button("Run Anomaly Detection"):
            analyzed_data = anomaly_logic_only(st.session_state['matched_results'])
            st.write(analyzed_data)

def main():
    page = sidebar_nav()
    if page == "Dashboard": page_dashboard()
    elif page == "Data Ingestion": page_entry()
    elif page == "Reconciliation": page_analysis()

if __name__ == "__main__":
    main()
