import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

# Load data dan model
@st.cache_resource
def load_models():
    """Load semua model dan data yang sudah disimpan"""
    try:
        with open('dt_model.pkl', 'rb') as f:
            dt_model = pickle.load(f)
        
        with open('item_columns.pkl', 'rb') as f:
            item_columns = pickle.load(f)
        
        transaction_matrix = np.load('transaction_matrix.npy', allow_pickle=True)
        frequent_itemsets = pd.read_pickle('frequent_itemsets.pkl')
        association_rules = pd.read_pickle('association_rules.pkl')
        tree_rules = pd.read_pickle('tree_rules.pkl')
        
        return {
            'dt_model': dt_model,
            'item_columns': item_columns,
            'transaction_matrix': transaction_matrix,
            'frequent_itemsets': frequent_itemsets,
            'association_rules': association_rules,
            'tree_rules': tree_rules
        }
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        st.info("Pastikan Anda sudah menjalankan notebook preparation dan memiliki file yang diperlukan")
        return None

# Sidebar navigation
st.sidebar.title("🛒 Market Basket Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi",
    ["📊 Dashboard", "🔍 Association Rules", "🌳 Decision Tree", "🎯 Predict"]
)

# Load models
models_data = load_models()

if models_data is None:
    st.error("Gagal memuat model. Pastikan file model tersedia.")
    st.stop()

# Dashboard Page
if page == "📊 Dashboard":
    st.title("📊 Dashboard Analisis Market Basket")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Items",
            value=len(models_data['item_columns'])
        )
    
    with col2:
        st.metric(
            label="Total Transactions",
            value=models_data['transaction_matrix'].shape[0]
        )
    
    with col3:
        st.metric(
            label="Association Rules",
            value=len(models_data['association_rules'])
        )
    
    st.markdown("---")
    
    # Top items chart
    st.subheader("📈 Top 20 Items Paling Populer")
    
    # Hitung frekuensi items
    item_counts = models_data['transaction_matrix'].sum(axis=0)
    top_items_df = pd.DataFrame({
        'Item': models_data['item_columns'],
        'Count': item_counts
    }).sort_values('Count', ascending=False).head(20)
    
    fig = px.bar(
        top_items_df,
        x='Count',
        y='Item',
        orientation='h',
        title='Top 20 Items by Frequency',
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Frequent itemsets
    st.subheader("🛍️ Frequent Itemsets")
    
    min_support = st.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=0.1,
        value=0.02,
        step=0.01,
        format="%.3f"
    )
    
    filtered_itemsets = models_data['frequent_itemsets'][
        models_data['frequent_itemsets']['support'] >= min_support
    ]
    
    st.dataframe(
        filtered_itemsets.sort_values('support', ascending=False).head(20),
        use_container_width=True
    )

# Association Rules Page
elif page == "🔍 Association Rules":
    st.title("🔍 Association Rules Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Aturan Asosiasi")
        
        # Filter controls
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05
        )
        
        min_lift = st.slider(
            "Minimum Lift",
            min_value=1.0,
            max_value=5.0,
            value=1.5,
            step=0.1
        )
        
        # Filter rules
        filtered_rules = models_data['association_rules'][
            (models_data['association_rules']['confidence'] >= min_confidence) &
            (models_data['association_rules']['lift'] >= min_lift)
        ]
        
        st.write(f"**{len(filtered_rules)} rules found**")
        
        # Display rules
        for idx, row in filtered_rules.head(20).iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            
            with st.expander(f"Rule {idx+1}: IF {antecedents} THEN {consequents}"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Support", f"{row['support']:.3f}")
                with col_b:
                    st.metric("Confidence", f"{row['confidence']:.3f}")
                with col_c:
                    st.metric("Lift", f"{row['lift']:.3f}")
    
    with col2:
        st.subheader("Rules Summary")
        
        # Confidence distribution
        fig = px.histogram(
            models_data['association_rules'],
            x='confidence',
            nbins=20,
            title='Distribution of Confidence'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top rules by lift
        top_lift = models_data['association_rules'].nlargest(5, 'lift')
        st.write("**Top Rules by Lift:**")
        for _, row in top_lift.iterrows():
            st.write(f"{list(row['antecedents'])[0]} → {list(row['consequents'])[0]}")
            st.progress(row['lift'] / 5)
            st.caption(f"Lift: {row['lift']:.2f}")

# Decision Tree Page
elif page == "🌳 Decision Tree":
    st.title("🌳 Decision Tree Model")
    
    st.write("""
    Model Decision Tree ini memprediksi apakah item **'whole milk'** akan dibeli 
    berdasarkan item lain dalam transaksi.
    """)
    
    # Display tree rules
    st.subheader("📋 Rules from Decision Tree")
    
    if not models_data['tree_rules'].empty:
        tree_rules_df = models_data['tree_rules'].copy()
        
        # Format rules untuk display
        tree_rules_df['rule_display'] = tree_rules_df.apply(
            lambda x: f"IF {x['rule']} THEN {x['class']}",
            axis=1
        )
        
        # Filter untuk rules yang memprediksi 'whole milk'
        milk_rules = tree_rules_df[tree_rules_df['class'] == 'whole milk']
        
        st.write(f"**{len(milk_rules)} rules predict 'whole milk'**")
        
        # Display top rules
        for idx, row in milk_rules.head(10).iterrows():
            st.info(f"**Rule {idx+1}:** {row['rule_display']}")
            st.write(f"Support: {row['support']:.2%} ({int(row['samples'])} samples)")
            st.markdown("---")
    
    # Tree visualization
    st.subheader("🌲 Tree Visualization")
    
    # Karena plot_tree dari sklearn menghasilkan matplotlib figure,
    # kita perlu konversi ke image untuk Streamlit
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        models_data['dt_model'],
        feature_names=models_data['item_columns'],
        class_names=['Not whole milk', 'whole milk'],
        filled=True,
        rounded=True,
        max_depth=3,  # Limit depth untuk readability
        ax=ax
    )
    
    st.pyplot(fig)

# Prediction Page
elif page == "🎯 Predict":
    st.title("🎯 Prediction dengan Decision Tree")
    
    st.write("""
    Pilih beberapa item yang sudah dibeli untuk memprediksi apakah 
    **whole milk** akan dibeli juga.
    """)
    
    # Item selection
    selected_items = st.multiselect(
        "Select items already in basket:",
        options=models_data['item_columns'],
        help="Pilih item yang sudah ada di keranjang belanja"
    )
    
    if selected_items:
        # Create feature vector
        feature_vector = np.zeros(len(models_data['item_columns']))
        
        for item in selected_items:
            if item in models_data['item_columns']:
                idx = models_data['item_columns'].index(item)
                feature_vector[idx] = 1
        
        # Remove target item ('whole milk') from features
        target_idx = models_data['item_columns'].index('whole milk')
        feature_vector = np.delete(feature_vector, target_idx)
        
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        # Make prediction
        prediction = models_data['dt_model'].predict(feature_vector)
        prediction_prob = models_data['dt_model'].predict_proba(feature_vector)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Result")
            
            if prediction[0] == 1:
                st.success("✅ **Recommendation: BUY whole milk**")
                st.balloons()
            else:
                st.warning("⏸️ **Recommendation: Probably NOT buy whole milk**")
        
        with col2:
            st.subheader("Confidence Score")
            
            # Progress bar for probability
            prob_yes = prediction_prob[0][1]
            
            st.metric(
                "Probability of buying whole milk",
                f"{prob_yes:.1%}"
            )
            
            st.progress(prob_yes)
            
            if prob_yes > 0.7:
                st.info("High confidence recommendation")
            elif prob_yes > 0.4:
                st.info("Moderate confidence recommendation")
            else:
                st.info("Low confidence recommendation")
        
        # Show reasoning
        st.subheader("🤖 Reasoning")
        
        if selected_items:
            st.write("Berdasarkan item yang Anda pilih:")
            for item in selected_items:
                st.write(f"- {item}")
            
            st.write(f"\nModel memprediksi dengan **{prob_yes:.1%}** confidence bahwa whole milk akan dibeli.")
        
        # Show similar rules
        st.subheader("📖 Similar Association Rules")
        
        # Find rules with selected items as antecedents
        selected_set = set(selected_items)
        similar_rules = []
        
        for _, rule in models_data['association_rules'].iterrows():
            if selected_set.issubset(set(rule['antecedents'])):
                similar_rules.append(rule)
        
        if similar_rules:
            st.write(f"Ditemukan {len(similar_rules)} aturan yang relevan:")
            
            for i, rule in enumerate(similar_rules[:5], 1):
                consequents = ', '.join(list(rule['consequents']))
                st.write(f"{i}. **IF** {', '.join(list(rule['antecedents']))} **THEN** {consequents}")
                st.caption(f"Confidence: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f}")
        else:
            st.info("Tidak ditemukan aturan yang spesifik. Model menggunakan pola umum dari dataset.")
    
    else:
        st.info("Pilih minimal 1 item untuk mendapatkan prediksi")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Market Basket Analysis App**  
    Menggunakan Decision Tree dan Association Rules  
    Dataset: Groceries Transactions
    """
)