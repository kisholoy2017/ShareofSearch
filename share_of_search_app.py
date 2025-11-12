import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import yaml
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Share of Search Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'df_avg' not in st.session_state:
    st.session_state.df_avg = None
if 'df_trends' not in st.session_state:
    st.session_state.df_trends = None
if 'df_keywords' not in st.session_state:
    st.session_state.df_keywords = None

# Country mapping (location IDs)
COUNTRY_MAP = {
    "Belgium": "2056",
    "United States": "2840",
    "United Kingdom": "2826",
    "Netherlands": "2528",
    "France": "2250",
    "Germany": "2276",
    "Spain": "2724",
    "Italy": "2380",
    "Canada": "2124",
    "Australia": "2036",
    "India": "2356",
    "Japan": "2392",
    "Brazil": "2076",
    "Mexico": "2484",
    "Switzerland": "2756",
    "Austria": "2040",
    "Poland": "2616",
    "Sweden": "2752",
    "Norway": "2578",
    "Denmark": "2208"
}

# Language mapping
LANGUAGE_MAP = {
    "English": "1000",
    "French": "1002",
    "German": "1003",
    "Dutch": "1010",
    "Spanish": "1003",
    "Italian": "1004",
    "Portuguese": "1014",
    "Japanese": "1005",
    "Swedish": "1015",
    "Danish": "1009",
    "Norwegian": "1013",
    "Polish": "1025"
}

# Helper functions
def get_date_range(months_back):
    """Get start and end dates based on months back"""
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months_back)
    
    return {
        'start_year': start_date.year,
        'start_month': start_date.month,
        'end_year': end_date.year,
        'end_month': end_date.month
    }

def map_locations_ids_to_resource_names(client, location_ids):
    """Converts location IDs to resource names"""
    geo_target_constant_service = client.get_service("GeoTargetConstantService")
    build_resource_name = geo_target_constant_service.geo_target_constant_path
    return [build_resource_name(location_id) for location_id in location_ids]

def get_brand_monthly_search_volume(client, customer_id, location_ids, language_id, brand_name, months_back, retry_count=0):
    """Get monthly search volume for a brand"""
    
    try:
        keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
        keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
        
        location_rns = map_locations_ids_to_resource_names(client, location_ids)
        google_ads_service = client.get_service("GoogleAdsService")
        language_rn = google_ads_service.language_constant_path(language_id)
        
        request = client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = customer_id
        request.language = language_rn
        request.geo_target_constants = location_rns
        request.include_adult_keywords = False
        request.keyword_plan_network = keyword_plan_network
        request.keyword_seed.keywords.extend([brand_name.lower()])
        
        # Add historical metrics
        date_range = get_date_range(months_back)
        request.historical_metrics_options.year_month_range.start.year = date_range['start_year']
        request.historical_metrics_options.year_month_range.start.month = date_range['start_month']
        request.historical_metrics_options.year_month_range.end.year = date_range['end_year']
        request.historical_metrics_options.year_month_range.end.month = date_range['end_month']
        
        keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
        
        monthly_data = {}
        keywords_found = []
        total_avg_volume = 0
        
        for idea in keyword_ideas:
            keyword = idea.text.lower()
            brand_lower = brand_name.lower()
            
            if brand_lower in keyword:
                metrics = idea.keyword_idea_metrics
                avg_volume = metrics.avg_monthly_searches if metrics.avg_monthly_searches else 0
                total_avg_volume += avg_volume
                
                if metrics.monthly_search_volumes:
                    for monthly_volume in metrics.monthly_search_volumes:
                        date_key = f"{monthly_volume.year}-{monthly_volume.month:02d}"
                        volume = monthly_volume.monthly_searches if monthly_volume.monthly_searches else 0
                        
                        if date_key in monthly_data:
                            monthly_data[date_key] += volume
                        else:
                            monthly_data[date_key] = volume
                
                keywords_found.append({
                    'keyword': idea.text,
                    'avg_volume': avg_volume,
                    'brand': brand_name
                })
        
        return total_avg_volume, monthly_data, keywords_found
    
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            if retry_count < 3:
                wait_time = 10 * (retry_count + 1)
                time.sleep(wait_time)
                return get_brand_monthly_search_volume(client, customer_id, location_ids, language_id, brand_name, months_back, retry_count + 1)
            else:
                return 0, {}, []
        else:
            st.error(f"Error for {brand_name}: {str(e)[:200]}")
            return 0, {}, []

def create_trend_chart(df_trends, target_brand):
    """Create interactive trend chart with Plotly"""
    
    fig = go.Figure()
    
    for brand in df_trends['brand'].unique():
        brand_data = df_trends[df_trends['brand'] == brand].sort_values('month')
        
        if brand == target_brand:
            fig.add_trace(go.Scatter(
                x=brand_data['month'],
                y=brand_data['share_of_search'],
                mode='lines+markers',
                name=brand,
                line=dict(width=3, color='#FF6B6B'),
                marker=dict(size=8)
            ))
        else:
            fig.add_trace(go.Scatter(
                x=brand_data['month'],
                y=brand_data['share_of_search'],
                mode='lines+markers',
                name=brand,
                line=dict(width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title='Share of Search Trends',
        xaxis_title='Month',
        yaxis_title='Share of Search (%)',
        hovermode='x unified',
        height=600,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig

def create_bar_chart(df_avg, target_brand):
    """Create interactive bar chart"""
    
    colors = ['#FF6B6B' if row['is_target'] else '#4ECDC4' for _, row in df_avg.iterrows()]
    
    fig = go.Figure(go.Bar(
        x=df_avg['avg_volume'],
        y=df_avg['brand'],
        orientation='h',
        marker_color=colors,
        text=df_avg['avg_volume'],
        texttemplate='%{text:,}',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Average Monthly Search Volume by Brand',
        xaxis_title='Search Volume',
        yaxis_title='Brand',
        height=500
    )
    
    return fig

def create_sos_bar_chart(df_avg, target_brand):
    """Create SoS percentage bar chart"""
    
    colors = ['#FF6B6B' if row['is_target'] else '#4ECDC4' for _, row in df_avg.iterrows()]
    
    fig = go.Figure(go.Bar(
        x=df_avg['share_of_search'],
        y=df_avg['brand'],
        orientation='h',
        marker_color=colors,
        text=df_avg['share_of_search'],
        texttemplate='%{text:.2f}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Share of Search Percentage',
        xaxis_title='Share of Search (%)',
        yaxis_title='Brand',
        height=500
    )
    
    return fig

def create_pie_chart(df_avg, target_brand):
    """Create pie chart"""
    
    df_top = df_avg.head(7).copy()
    others_volume = df_avg.iloc[7:]['avg_volume'].sum() if len(df_avg) > 7 else 0
    
    if others_volume > 0:
        df_top = pd.concat([df_top, pd.DataFrame({
            'brand': ['Others'],
            'avg_volume': [others_volume]
        })], ignore_index=True)
    
    colors = ['#FF6B6B' if brand == target_brand else '#4ECDC4' for brand in df_top['brand']]
    if len(colors) > 7:
        colors[-1] = '#95a5a6'
    
    fig = go.Figure(go.Pie(
        labels=df_top['brand'],
        values=df_top['avg_volume'],
        marker_colors=colors,
        hole=0.3
    ))
    
    fig.update_layout(
        title='Market Share Distribution',
        height=500
    )
    
    return fig

def run_analysis(client, customer_id, location_ids, language_id, target_brand, competitor_brands, months_back, delay_seconds):
    """Run the Share of Search analysis"""
    
    results = []
    all_monthly_data = []
    all_keywords = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_brands = len(competitor_brands) + 1
    current = 0
    
    # Target brand
    status_text.text(f"Analyzing {target_brand}...")
    target_avg, target_monthly, target_keywords = get_brand_monthly_search_volume(
        client, customer_id, location_ids, language_id, target_brand, months_back
    )
    
    results.append({
        'brand': target_brand,
        'avg_volume': target_avg,
        'is_target': True
    })
    
    for month, volume in target_monthly.items():
        all_monthly_data.append({
            'brand': target_brand,
            'month': month,
            'volume': volume
        })
    
    all_keywords.extend(target_keywords)
    current += 1
    progress_bar.progress(current / total_brands)
    
    time.sleep(delay_seconds)
    
    # Competitors
    for i, competitor in enumerate(competitor_brands, 1):
        status_text.text(f"Analyzing {competitor} ({i}/{len(competitor_brands)})...")
        
        comp_avg, comp_monthly, comp_keywords = get_brand_monthly_search_volume(
            client, customer_id, location_ids, language_id, competitor, months_back
        )
        
        results.append({
            'brand': competitor,
            'avg_volume': comp_avg,
            'is_target': False
        })
        
        for month, volume in comp_monthly.items():
            all_monthly_data.append({
                'brand': competitor,
                'month': month,
                'volume': volume
            })
        
        all_keywords.extend(comp_keywords)
        current += 1
        progress_bar.progress(current / total_brands)
        
        if i < len(competitor_brands):
            time.sleep(delay_seconds)
    
    status_text.text("Analysis complete!")
    
    # Create DataFrames
    df_avg = pd.DataFrame(results)
    df_monthly = pd.DataFrame(all_monthly_data)
    df_keywords = pd.DataFrame(all_keywords)
    
    # Calculate Share of Search
    total_avg = df_avg['avg_volume'].sum()
    
    if total_avg > 0:
        df_avg['share_of_search'] = (df_avg['avg_volume'] / total_avg * 100).round(2)
        df_avg = df_avg.sort_values('avg_volume', ascending=False).reset_index(drop=True)
        
        # Calculate monthly SoS
        df_trends = df_monthly.copy()
        
        if not df_trends.empty:
            monthly_totals = df_trends.groupby('month')['volume'].sum().reset_index()
            monthly_totals.columns = ['month', 'total_volume']
            
            df_trends = df_trends.merge(monthly_totals, on='month')
            df_trends['share_of_search'] = (df_trends['volume'] / df_trends['total_volume'] * 100).round(2)
            df_trends = df_trends.sort_values('month')
        
        return df_avg, df_trends, df_keywords
    else:
        return None, None, None

# =====================================================
# STREAMLIT APP
# =====================================================

st.markdown('<p class="main-header">📊 Share of Search Calculator</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# STEP 1: API Authentication
st.sidebar.subheader("Step 1: API Authentication")

config_method = st.sidebar.radio(
    "Choose authentication method:",
    ["Upload google-ads.yaml", "Manual Entry"],
    help="Upload your config file or enter credentials manually"
)

client = None
customer_id = None

if config_method == "Upload google-ads.yaml":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your google-ads.yaml", 
        type=['yaml', 'yml'],
        help="This file contains your Google Ads API credentials"
    )
    
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            client = GoogleAdsClient.load_from_storage(path=tmp_path, version="v22")
            
            # Try to extract customer ID from config
            with open(tmp_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'customer_id' in config:
                    customer_id = config['customer_id']
            
            st.sidebar.success("✅ API client initialized!")
            os.unlink(tmp_path)
            
        except Exception as e:
            st.sidebar.error(f"Error loading config: {str(e)}")

else:  # Manual Entry
    st.sidebar.info("Enter your Google Ads API credentials:")
    
    with st.sidebar.expander("📋 Where to find these credentials", expanded=False):
        st.markdown("""
        **Developer Token:** Google Ads API → Settings → Developer Token
        
        **OAuth Credentials:** Google Cloud Console → APIs & Services → Credentials
        
        **Customer ID:** Your Google Ads account ID (no dashes)
        
        **Login Customer ID:** Manager account ID (if using MCC)
        
        [📚 Full Setup Guide](https://developers.google.com/google-ads/api/docs/first-call/overview)
        """)
    
    dev_token = st.sidebar.text_input("Developer Token", type="password")
    client_id_input = st.sidebar.text_input("Client ID")
    client_secret = st.sidebar.text_input("Client Secret", type="password")
    refresh_token = st.sidebar.text_input("Refresh Token", type="password")
    login_customer_id = st.sidebar.text_input("Login Customer ID (MCC)", help="Leave empty if not using manager account")
    customer_id = st.sidebar.text_input("Customer ID", help="Your Google Ads account ID")
    
    if st.sidebar.button("Initialize Client"):
        if all([dev_token, client_id_input, client_secret, refresh_token, customer_id]):
            try:
                config = {
                    'developer_token': dev_token,
                    'client_id': client_id_input,
                    'client_secret': client_secret,
                    'refresh_token': refresh_token,
                    'use_proto_plus': True,
                }
                
                if login_customer_id:
                    config['login_customer_id'] = login_customer_id
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml', mode='w') as tmp_file:
                    yaml.dump(config, tmp_file)
                    tmp_path = tmp_file.name
                
                client = GoogleAdsClient.load_from_storage(path=tmp_path, version="v22")
                st.sidebar.success("✅ API client initialized!")
                os.unlink(tmp_path)
                
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.error("Please fill all required fields")

# Show customer ID input if not extracted from yaml
if client and not customer_id:
    customer_id = st.sidebar.text_input("Customer ID", help="Your Google Ads account ID")

st.sidebar.markdown("---")

# STEP 2: Analysis Parameters
st.sidebar.subheader("Step 2: Analysis Parameters")

# Target Brand
target_brand = st.sidebar.text_input(
    "🎯 Target Brand",
    value="LampTwist",
    help="The brand you want to analyze"
)

# Date Range
months_back = st.sidebar.slider(
    "📅 Time Window (Months)",
    min_value=3,
    max_value=24,
    value=12,
    help="How many months of historical data to analyze"
)

st.sidebar.info(f"📊 Analyzing: {months_back} months of data")

# Country Selection
country = st.sidebar.selectbox(
    "🌍 Country",
    options=list(COUNTRY_MAP.keys()),
    index=0,
    help="Select the geographic market to analyze"
)

location_id = COUNTRY_MAP[country]

# Language Selection
language = st.sidebar.selectbox(
    "🗣️ Language",
    options=list(LANGUAGE_MAP.keys()),
    index=0,
    help="Language for keyword searches"
)

language_id = LANGUAGE_MAP[language]

# Competitor Brands
st.sidebar.subheader("🏆 Competitor Brands")

competitor_input = st.sidebar.text_area(
    "Enter competitor brands (one per line):",
    value="MOHD\nLa Redoute\nwest elm\nWayfair\nTemple & Webster\nMade in Design FR\nKave Home\nSklum\nBulbSquare\nLumens\nLitfad.com\nFinnish Design\nSilvera\nSmallable",
    height=200,
    help="List each competitor brand on a new line"
)

competitor_brands = [brand.strip() for brand in competitor_input.split('\n') if brand.strip()]

st.sidebar.info(f"📊 Total competitors: {len(competitor_brands)}")

st.sidebar.markdown("---")

# STEP 3: Advanced Settings
st.sidebar.subheader("Step 3: Advanced Settings")

delay_seconds = st.sidebar.slider(
    "⏱️ Delay between API calls (seconds)",
    min_value=3,
    max_value=15,
    value=6,
    help="Higher values prevent rate limiting but take longer"
)

estimated_time = (len(competitor_brands) + 1) * delay_seconds / 60
st.sidebar.info(f"⏱️ Estimated time: {estimated_time:.1f} minutes")

st.sidebar.markdown("---")

# Run Analysis Button
run_button = st.sidebar.button(
    "🚀 Run Analysis",
    type="primary",
    use_container_width=True,
    disabled=not (client and customer_id and target_brand and competitor_brands)
)

# Main Content
if not client:
    st.info("👈 Please configure your Google Ads API credentials in the sidebar to begin.")
    
    with st.expander("📚 Getting Started Guide", expanded=True):
        st.markdown("""
        ## How to Use This App
        
        ### Step 1: Get Google Ads API Access
        
        You need:
        1. **Developer Token** - From your Google Ads account
        2. **OAuth Credentials** - From Google Cloud Console
        3. **Customer ID** - Your Google Ads account number
        
        [📖 Complete Setup Guide](https://developers.google.com/google-ads/api/docs/first-call/overview)
        
        ### Step 2: Configure Analysis
        
        1. **Target Brand** - The brand you want to track
        2. **Time Window** - How many months to analyze (3-24)
        3. **Country** - Geographic market
        4. **Competitors** - List of competing brands
        
        ### Step 3: Run Analysis
        
        Click "Run Analysis" and wait 2-5 minutes for results.
        
        ### What You'll Get
        
        - 📈 Trend chart showing Share of Search over time
        - 📊 Market share comparisons
        - 💾 CSV files with detailed data
        - 🎯 Key metrics and insights
        
        ---
        
        ### Need Help?
        
        - [Google Ads API Documentation](https://developers.google.com/google-ads/api)
        - [Authentication Guide](https://developers.google.com/google-ads/api/docs/oauth/overview)
        """)

elif not customer_id:
    st.warning("⚠️ Please enter your Customer ID in the sidebar.")

elif not target_brand:
    st.warning("⚠️ Please enter a target brand in the sidebar.")

elif not competitor_brands:
    st.warning("⚠️ Please enter at least one competitor brand in the sidebar.")

elif run_button:
    st.session_state.analysis_complete = False
    
    # Show analysis configuration
    st.subheader("📋 Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Brand", target_brand)
        st.metric("Competitors", len(competitor_brands))
    with col2:
        st.metric("Time Window", f"{months_back} months")
        st.metric("Country", country)
    with col3:
        st.metric("Language", language)
        st.metric("Est. Time", f"{estimated_time:.1f} min")
    
    st.markdown("---")
    
    with st.spinner("Running Share of Search analysis..."):
        try:
            df_avg, df_trends, df_keywords = run_analysis(
                client=client,
                customer_id=customer_id,
                location_ids=[location_id],
                language_id=language_id,
                target_brand=target_brand,
                competitor_brands=competitor_brands,
                months_back=months_back,
                delay_seconds=delay_seconds
            )
            
            if df_avg is not None:
                st.session_state.df_avg = df_avg
                st.session_state.df_trends = df_trends
                st.session_state.df_keywords = df_keywords
                st.session_state.target_brand = target_brand
                st.session_state.analysis_complete = True
                st.success("✅ Analysis complete!")
            else:
                st.error("❌ No data collected. Please check your settings.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display Results
if st.session_state.analysis_complete and st.session_state.df_avg is not None:
    df_avg = st.session_state.df_avg
    df_trends = st.session_state.df_trends
    df_keywords = st.session_state.df_keywords
    target_brand = st.session_state.target_brand
    
    # Key Metrics
    st.header("📊 Key Metrics")
    
    target_row = df_avg[df_avg['is_target']].iloc[0]
    leader_row = df_avg.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Target Brand SoS",
            f"{target_row['share_of_search']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Search Volume",
            f"{target_row['avg_volume']:,}"
        )
    
    with col3:
        rank = (df_avg['avg_volume'] > target_row['avg_volume']).sum() + 1
        st.metric(
            "Market Rank",
            f"#{rank} of {len(df_avg)}"
        )
    
    with col4:
        st.metric(
            "Market Leader",
            leader_row['brand'],
            f"{leader_row['share_of_search']:.2f}%"
        )
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Volume", "🥧 Market Share", "📋 Data"])
    
    with tab1:
        st.subheader("Share of Search Trends")
        if df_trends is not None and not df_trends.empty:
            fig_trend = create_trend_chart(df_trends, target_brand)
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Add insights
            st.markdown("### 💡 Trend Insights")
            
            # Calculate trend for target brand
            target_trend = df_trends[df_trends['brand'] == target_brand].sort_values('month')
            if len(target_trend) >= 2:
                first_sos = target_trend.iloc[0]['share_of_search']
                last_sos = target_trend.iloc[-1]['share_of_search']
                change = last_sos - first_sos
                
                if change > 0:
                    st.success(f"📈 {target_brand} Share of Search increased by {change:.2f} percentage points")
                elif change < 0:
                    st.warning(f"📉 {target_brand} Share of Search decreased by {abs(change):.2f} percentage points")
                else:
                    st.info(f"➡️ {target_brand} Share of Search remained stable")
        else:
            st.info("No monthly trend data available")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Search Volume")
            fig_bar = create_bar_chart(df_avg, target_brand)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Share of Search %")
            fig_sos = create_sos_bar_chart(df_avg, target_brand)
            st.plotly_chart(fig_sos, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Market Share Distribution")
            fig_pie = create_pie_chart(df_avg, target_brand)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Market Summary")
            st.metric("Total Brands", len(df_avg))
            st.metric("Total Search Volume", f"{df_avg['avg_volume'].sum():,}")
            st.metric("Average SoS", f"{df_avg['share_of_search'].mean():.2f}%")
            
            if target_brand != leader_row['brand']:
                gap = leader_row['avg_volume'] - target_row['avg_volume']
                st.metric("Gap to Leader", f"{gap:,}")
    
    with tab4:
        st.subheader("Brand Rankings")
        
        # Format the dataframe
        display_df = df_avg[['brand', 'avg_volume', 'share_of_search']].copy()
        display_df.columns = ['Brand', 'Avg Monthly Volume', 'Share of Search (%)']
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df = display_df[['Rank', 'Brand', 'Avg Monthly Volume', 'Share of Search (%)']]
        
        st.dataframe(
            display_df.style.format({
                'Avg Monthly Volume': '{:,.0f}',
                'Share of Search (%)': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        st.subheader(f"Top Keywords for {target_brand}")
        target_kw = df_keywords[df_keywords['brand'] == target_brand].sort_values('avg_volume', ascending=False).head(20)
        if not target_kw.empty:
            st.dataframe(
                target_kw[['keyword', 'avg_volume']].style.format({
                    'avg_volume': '{:,.0f}'
                }),
                use_container_width=True
            )
        else:
            st.info(f"No keywords found for {target_brand}")
    
    # Download Section
    st.markdown("---")
    st.header("💾 Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_summary = df_avg.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary CSV",
            data=csv_summary,
            file_name=f"sos_summary_{target_brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        if df_trends is not None and not df_trends.empty:
            csv_trends = df_trends.to_csv(index=False)
            st.download_button(
                label="📥 Download Trends CSV",
                data=csv_trends,
                file_name=f"sos_trends_{target_brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        csv_keywords = df_keywords.to_csv(index=False)
        st.download_button(
            label="📥 Download Keywords CSV",
            data=csv_keywords,
            file_name=f"sos_keywords_{target_brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
