pip install matplotlib.pyplot
pip install seaborn
pip install plotly

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Private Portfolio Analytics", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced default columns with more fields
DEFAULT_COLUMNS = [
    "Company Name", "ISIN", "Market Value", "Investment Date", 
    "Sector", "Stage", "Ownership %", "Initial Investment", "Notes"
]

# Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > div > div > div > div {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def create_empty_df():
    return pd.DataFrame(columns=DEFAULT_COLUMNS)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def download_link(df: pd.DataFrame, filename: str = "portfolio.csv"):
    csv = df_to_csv_bytes(df)
    b64 = base64.b64encode(csv).decode()
    href = f"data:file/csv;base64,{b64}"
    return href

def validate_company_data(name, isin, market_value):
    """Validate input data for new companies"""
    errors = []
    if not name.strip():
        errors.append("Company name is required")
    if not isin.strip():
        errors.append("ISIN is required")
    if market_value <= 0:
        errors.append("Market value must be greater than 0")
    return errors

def calculate_portfolio_metrics(df):
    """Calculate key portfolio metrics"""
    if df.empty:
        return {}
    
    total_value = df['Market Value'].sum()
    total_initial = df['Initial Investment'].sum() if 'Initial Investment' in df.columns else 0
    
    metrics = {
        'total_value': total_value,
        'total_initial': total_initial,
        'total_gain_loss': total_value - total_initial if total_initial > 0 else 0,
        'total_return_pct': ((total_value - total_initial) / total_initial * 100) if total_initial > 0 else 0,
        'num_companies': len(df),
        'avg_holding': total_value / len(df),
        'largest_holding': df['Market Value'].max(),
        'smallest_holding': df['Market Value'].min(),
    }
    
    return metrics

def create_sector_analysis(df):
    """Create sector-wise analysis"""
    if 'Sector' not in df.columns or df.empty:
        return None
    
    sector_stats = df.groupby('Sector').agg({
        'Market Value': ['sum', 'count', 'mean'],
        'Ownership %': 'mean'
    }).round(2)
    
    sector_stats.columns = ['Total Value', 'Count', 'Avg Value', 'Avg Ownership %']
    sector_stats['Percentage'] = (sector_stats['Total Value'] / df['Market Value'].sum() * 100).round(2)
    
    return sector_stats.sort_values('Total Value', ascending=False)

def create_stage_analysis(df):
    """Create investment stage analysis"""
    if 'Stage' not in df.columns or df.empty:
        return None
    
    stage_stats = df.groupby('Stage').agg({
        'Market Value': ['sum', 'count'],
        'Ownership %': 'mean'
    }).round(2)
    
    stage_stats.columns = ['Total Value', 'Count', 'Avg Ownership %']
    stage_stats['Percentage'] = (stage_stats['Total Value'] / df['Market Value'].sum() * 100).round(2)
    
    return stage_stats.sort_values('Total Value', ascending=False)

# Initialize session state
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = create_empty_df()

# Sidebar
with st.sidebar:
    st.title("🚀 Portfolio Analytics")
    st.markdown("---")
    
    # Quick stats in sidebar
    if not st.session_state.portfolio_df.empty:
        metrics = calculate_portfolio_metrics(st.session_state.portfolio_df)
        st.metric("Total Value", f"€{metrics['total_value']:,.0f}")
        st.metric("Companies", f"{metrics['num_companies']}")
        if metrics['total_initial'] > 0:
            st.metric("Total Return", f"{metrics['total_return_pct']:.1f}%")
    
    st.markdown("---")
    page = st.radio(
        "Navigation", 
        ["📊 Dashboard", "💼 Portfolio", "📈 Analytics", "📁 Import/Export", "⚙️ Settings"],
        index=0
    )

# Main content area
if page == "📊 Dashboard":
    st.title("Portfolio Dashboard")
    
    if st.session_state.portfolio_df.empty:
        st.info("🏃‍♂️ Get started by adding companies in the Portfolio section or importing a CSV!")
        
        # Quick start section
        st.subheader("Quick Start")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Add First Company", type="primary"):
                st.switch_page = "💼 Portfolio"
        with col2:
            if st.button("📄 Download Template"):
                template_df = pd.DataFrame([{
                    "Company Name": "TechStart Inc",
                    "ISIN": "TECH001",
                    "Market Value": 500000,
                    "Investment Date": "2023-01-15",
                    "Sector": "Technology",
                    "Stage": "Series A",
                    "Ownership %": 15.5,
                    "Initial Investment": 250000,
                    "Notes": "AI-powered fintech startup"
                }])
                href = download_link(template_df, "portfolio_template.csv")
                st.markdown(f"[Download Template]({href})", unsafe_allow_html=True)
        with col3:
            st.markdown("📖 [View Documentation](#)")
    else:
        df = st.session_state.portfolio_df
        metrics = calculate_portfolio_metrics(df)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Portfolio Value", f"€{metrics['total_value']:,.0f}")
        with col2:
            st.metric("Number of Companies", f"{metrics['num_companies']}")
        with col3:
            if metrics['total_initial'] > 0:
                delta_color = "normal" if metrics['total_return_pct'] >= 0 else "inverse"
                st.metric("Total Return", f"{metrics['total_return_pct']:.1f}%", 
                         f"€{metrics['total_gain_loss']:,.0f}")
        with col4:
            st.metric("Average Holding", f"€{metrics['avg_holding']:,.0f}")
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Composition")
            if len(df) > 0:
                # Create Plotly pie chart
                fig = px.pie(
                    df, 
                    values='Market Value', 
                    names='Company Name',
                    title="Holdings by Market Value"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sector Distribution")
            if 'Sector' in df.columns and not df['Sector'].isna().all():
                sector_data = df.groupby('Sector')['Market Value'].sum().reset_index()
                fig = px.bar(
                    sector_data,
                    x='Sector',
                    y='Market Value',
                    title="Value by Sector"
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add sector information to see sector distribution")
        
        # Top holdings table
        st.subheader("Top Holdings")
        top_holdings = df.nlargest(10, 'Market Value')[['Company Name', 'Market Value', 'Sector', 'Stage']]
        top_holdings['Market Value'] = top_holdings['Market Value'].apply(lambda x: f"€{x:,.0f}")
        st.dataframe(top_holdings, use_container_width=True, hide_index=True)

elif page == "💼 Portfolio":
    st.title("Portfolio Management")
    
    # Enhanced add company form
    with st.expander("➕ Add New Company", expanded=not bool(len(st.session_state.portfolio_df))):
        with st.form("add_company", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Company Name*", help="Enter the company or startup name")
                isin = st.text_input("ISIN*", help="Custom identifier for the company")
                market_value = st.number_input("Current Market Value (€)*", min_value=0.0, format="%.2f")
                initial_investment = st.number_input("Initial Investment (€)", min_value=0.0, format="%.2f")
                ownership_pct = st.number_input("Ownership %", min_value=0.0, max_value=100.0, format="%.2f")
            
            with col2:
                investment_date = st.date_input("Investment Date", value=datetime.now().date())
                sector = st.selectbox("Sector", [
                    "", "Technology", "Healthcare", "Finance", "Consumer", "Industrial", 
                    "Real Estate", "Energy", "Materials", "Telecommunications", "Other"
                ])
                stage = st.selectbox("Investment Stage", [
                    "", "Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Growth", "Pre-IPO", "Other"
                ])
                notes = st.text_area("Notes", help="Additional information about the investment")
            
            submitted = st.form_submit_button("Add to Portfolio", type="primary")
            
            if submitted:
                errors = validate_company_data(name, isin, market_value)
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    new_row = {
                        "Company Name": name.strip(),
                        "ISIN": isin.strip(),
                        "Market Value": float(market_value),
                        "Investment Date": investment_date.strftime("%Y-%m-%d"),
                        "Sector": sector,
                        "Stage": stage,
                        "Ownership %": float(ownership_pct),
                        "Initial Investment": float(initial_investment),
                        "Notes": notes.strip()
                    }
                    st.session_state.portfolio_df = pd.concat([
                        st.session_state.portfolio_df, 
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
                    st.success(f"✅ Added {name} to portfolio")
                    st.rerun()
    
    # Portfolio table with editing capabilities
    if st.session_state.portfolio_df.empty:
        st.info("Your portfolio is empty. Add your first company using the form above!")
    else:
        st.subheader("Current Holdings")
        
        # Display options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_term = st.text_input("🔍 Search companies", placeholder="Enter company name or ISIN...")
        with col2:
            sort_column = st.selectbox("Sort by", ["Market Value", "Company Name", "Investment Date", "Sector"])
        with col3:
            sort_order = st.selectbox("Order", ["Descending", "Ascending"])
        
        # Filter and sort data
        df_display = st.session_state.portfolio_df.copy()
        
        if search_term:
            mask = (df_display['Company Name'].str.contains(search_term, case=False, na=False) |
                   df_display['ISIN'].str.contains(search_term, case=False, na=False))
            df_display = df_display[mask]
        
        ascending = sort_order == "Ascending"
        df_display = df_display.sort_values(sort_column, ascending=ascending)
        
        # Format display
        df_formatted = df_display.copy()
        df_formatted['Market Value'] = df_formatted['Market Value'].apply(lambda x: f"€{x:,.0f}")
        if 'Initial Investment' in df_formatted.columns:
            df_formatted['Initial Investment'] = df_formatted['Initial Investment'].apply(lambda x: f"€{x:,.0f}" if x > 0 else "-")
        if 'Ownership %' in df_formatted.columns:
            df_formatted['Ownership %'] = df_formatted['Ownership %'].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
        
        st.dataframe(df_formatted, use_container_width=True, hide_index=True)
        
        # Portfolio actions
        st.subheader("Portfolio Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Remove Last Entry"):
                if not st.session_state.portfolio_df.empty:
                    st.session_state.portfolio_df = st.session_state.portfolio_df.iloc[:-1].reset_index(drop=True)
                    st.success("Last entry removed")
                    st.rerun()
        
        with col2:
            if st.button("🧹 Clear All", type="secondary"):
                if st.button("Confirm Clear All", type="secondary"):
                    st.session_state.portfolio_df = create_empty_df()
                    st.success("Portfolio cleared")
                    st.rerun()
        
        with col3:
            href = download_link(st.session_state.portfolio_df)
            st.markdown(f"[📥 Download CSV]({href})", unsafe_allow_html=True)
        
        with col4:
            total_value = st.session_state.portfolio_df['Market Value'].sum()
            st.metric("Total Value", f"€{total_value:,.0f}")

elif page == "📈 Analytics":
    st.title("Portfolio Analytics")
    
    if st.session_state.portfolio_df.empty:
        st.info("Add companies to your portfolio to see detailed analytics")
    else:
        df = st.session_state.portfolio_df
        metrics = calculate_portfolio_metrics(df)
        
        # Performance metrics
        st.subheader("Performance Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"€{metrics['total_value']:,.0f}")
        with col2:
            if metrics['total_initial'] > 0:
                st.metric("Total Invested", f"€{metrics['total_initial']:,.0f}")
            else:
                st.metric("Companies", f"{metrics['num_companies']}")
        with col3:
            if metrics['total_initial'] > 0:
                st.metric("Unrealized P&L", f"€{metrics['total_gain_loss']:,.0f}")
            else:
                st.metric("Avg Holding", f"€{metrics['avg_holding']:,.0f}")
        with col4:
            if metrics['total_initial'] > 0:
                delta_color = "normal" if metrics['total_return_pct'] >= 0 else "inverse"
                st.metric("Total Return", f"{metrics['total_return_pct']:.1f}%")
            else:
                st.metric("Largest Holding", f"€{metrics['largest_holding']:,.0f}")
        
        # Sector Analysis
        st.subheader("Sector Analysis")
        if 'Sector' in df.columns and not df['Sector'].isna().all():
            sector_stats = create_sector_analysis(df)
            if sector_stats is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(sector_stats, use_container_width=True)
                
                with col2:
                    fig = px.treemap(
                        df,
                        path=['Sector', 'Company Name'],
                        values='Market Value',
                        title="Portfolio Treemap by Sector"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add sector information to companies for sector analysis")
        
        # Stage Analysis
        st.subheader("Investment Stage Analysis")
        if 'Stage' in df.columns and not df['Stage'].isna().all():
            stage_stats = create_stage_analysis(df)
            if stage_stats is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(stage_stats, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        stage_stats.reset_index(),
                        x='Stage',
                        y='Total Value',
                        title="Investment Value by Stage"
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add stage information to companies for stage analysis")
        
        # Risk Analysis
        st.subheader("Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Concentration risk
            st.write("**Concentration Risk**")
            df_sorted = df.sort_values('Market Value', ascending=False)
            df_sorted['Cumulative %'] = (df_sorted['Market Value'].cumsum() / df_sorted['Market Value'].sum() * 100)
            
            top_5_pct = df_sorted.head(5)['Market Value'].sum() / df_sorted['Market Value'].sum() * 100
            top_10_pct = df_sorted.head(min(10, len(df_sorted)))['Market Value'].sum() / df_sorted['Market Value'].sum() * 100
            
            st.metric("Top 5 Holdings", f"{top_5_pct:.1f}%")
            st.metric("Top 10 Holdings", f"{top_10_pct:.1f}%")
        
        with col2:
            # Portfolio distribution
            st.write("**Portfolio Distribution**")
            bins = [0, 50000, 250000, 1000000, float('inf')]
            labels = ['<€50K', '€50K-€250K', '€250K-€1M', '>€1M']
            df['Size Category'] = pd.cut(df['Market Value'], bins=bins, labels=labels, right=False)
            
            size_dist = df['Size Category'].value_counts()
            fig = px.bar(
                x=size_dist.index,
                y=size_dist.values,
                title="Holdings by Size Category"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

elif page == "📁 Import/Export":
    st.title("Import / Export Data")
    
    # Import section
    st.subheader("📤 Import Portfolio")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=["csv"],
            help="CSV must contain: Company Name, ISIN, Market Value"
        )
        
        if uploaded_file is not None:
            try:
                # Read and preview the uploaded file
                uploaded_df = pd.read_csv(uploaded_file)
                
                st.write("**File Preview:**")
                st.dataframe(uploaded_df.head(), use_container_width=True)
                
                # Column mapping
                st.write("**Column Mapping:**")
                available_cols = uploaded_df.columns.tolist()
                
                col_mapping = {}
                for required_col in ["Company Name", "ISIN", "Market Value"]:
                    col_mapping[required_col] = st.selectbox(
                        f"Map '{required_col}' to:", 
                        [""] + available_cols,
                        key=f"map_{required_col}"
                    )
                
                # Optional columns
                for optional_col in ["Investment Date", "Sector", "Stage", "Ownership %", "Initial Investment", "Notes"]:
                    if optional_col not in col_mapping:
                        col_mapping[optional_col] = st.selectbox(
                            f"Map '{optional_col}' to (optional):", 
                            [""] + available_cols,
                            key=f"map_{optional_col}"
                        )
                
                if st.button("Import Data", type="primary"):
                    # Validate required columns
                    required_mapped = all(col_mapping[col] for col in ["Company Name", "ISIN", "Market Value"])
                    
                    if required_mapped:
                        # Create mapped DataFrame
                        new_df = pd.DataFrame()
                        
                        for target_col, source_col in col_mapping.items():
                            if source_col and source_col in uploaded_df.columns:
                                new_df[target_col] = uploaded_df[source_col]
                        
                        # Clean and validate data
                        new_df['Market Value'] = pd.to_numeric(new_df['Market Value'], errors='coerce').fillna(0.0)
                        
                        # Remove rows with invalid data
                        valid_rows = (new_df['Company Name'].notna() & 
                                    new_df['ISIN'].notna() & 
                                    (new_df['Market Value'] > 0))
                        new_df = new_df[valid_rows]
                        
                        if not new_df.empty:
                            # Ask for merge or replace
                            import_action = st.radio(
                                "Import action:",
                                ["Replace existing portfolio", "Merge with existing portfolio"]
                            )
                            
                            if import_action == "Replace existing portfolio":
                                st.session_state.portfolio_df = new_df.reset_index(drop=True)
                            else:
                                st.session_state.portfolio_df = pd.concat([
                                    st.session_state.portfolio_df, 
                                    new_df
                                ], ignore_index=True)
                            
                            st.success(f"✅ Imported {len(new_df)} companies successfully!")
                            st.rerun()
                        else:
                            st.error("No valid data found after cleaning")
                    else:
                        st.error("Please map all required columns")
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.write("**Required Columns:**")
        st.code("""
Company Name
ISIN  
Market Value
        """)
        
        st.write("**Optional Columns:**")
        st.code("""
Investment Date
Sector
Stage  
Ownership %
Initial Investment
Notes
        """)
    
    # Export section
    st.subheader("📥 Export Portfolio")
    
    if not st.session_state.portfolio_df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Current portfolio export
            href = download_link(st.session_state.portfolio_df, "portfolio_export.csv")
            st.markdown(f"[📄 Download Full Portfolio]({href})", unsafe_allow_html=True)
        
        with col2:
            # Summary export
            if len(st.session_state.portfolio_df) > 0:
                summary_df = st.session_state.portfolio_df.groupby('Sector').agg({
                    'Market Value': 'sum',
                    'Company Name': 'count'
                }).rename(columns={'Company Name': 'Count'})
                href_summary = download_link(summary_df, "portfolio_summary.csv")
                st.markdown(f"[📊 Download Summary]({href_summary})", unsafe_allow_html=True)
        
        with col3:
            # Template export
            template_df = pd.DataFrame([{
                col: "" for col in DEFAULT_COLUMNS
            }])
            href_template = download_link(template_df, "portfolio_template.csv")
            st.markdown(f"[📋 Download Template]({href_template})", unsafe_allow_html=True)
    
    else:
        st.info("Add companies to your portfolio to enable export options")

elif page == "⚙️ Settings":
    st.title("Settings & Configuration")
    
    # Portfolio settings
    st.subheader("Portfolio Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        currency = st.selectbox("Default Currency", ["EUR", "USD", "GBP", "CHF"], index=0)
        date_format = st.selectbox("Date Format", ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"])
        
    with col2:
        decimal_places = st.number_input("Decimal Places for Values", min_value=0, max_value=4, value=2)
        show_percentages = st.checkbox("Show Percentages in Tables", value=True)
    
    # Data management
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset All Settings"):
            st.success("Settings reset to default")
    
    with col2:
        if st.button("📊 Recalculate All Metrics"):
            st.success("Metrics recalculated")
    
    # Export settings
    st.subheader("Export Preferences")
    
    export_format = st.radio("Default Export Format", ["CSV", "Excel", "JSON"])
    include_metadata = st.checkbox("Include Metadata in Exports", value=True)
    
    # About section
    st.subheader("About")
    st.info("""
    **Private Portfolio Analytics v2.0**
    
    An open-source platform for managing and analyzing private company investments.
    
    Features:
    - Portfolio management and tracking
    - Advanced analytics and visualizations  
    - Sector and stage analysis
    - Risk assessment tools
    - Import/export capabilities
    
    Built with Streamlit, Pandas, and Plotly.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🚀 Private Portfolio Analytics**")
st.sidebar.markdown("Open source investment tracking platform")
st.sidebar.markdown("[GitHub](https://github.com/your-repo) | [Documentation](https://docs.example.com)")
