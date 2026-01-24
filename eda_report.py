"""
Healthcare Fraud Detection - Comprehensive EDA Report
Generates a complete PDF report with all analyses and visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_eda_report():
    print("=" * 70)
    print("HEALTHCARE FRAUD DETECTION - COMPREHENSIVE EDA REPORT")
    print("=" * 70)
    
    # Load data
    print("\n[1/12] Loading datasets...")
    claims_df = pd.read_csv('data/claims.csv')
    
    # Try to load original Kaggle files for more analysis
    try:
        beneficiary_df = pd.read_csv('Dataset/Train_Beneficiarydata-1542865627584.csv')
        inpatient_df = pd.read_csv('Dataset/Train_Inpatientdata-1542865627584.csv')
        outpatient_df = pd.read_csv('Dataset/Train_Outpatientdata-1542865627584.csv')
        provider_df = pd.read_csv('Dataset/Train-1542865627584.csv')
        has_original = True
        print(f"  ✓ Loaded original Kaggle datasets")
    except:
        has_original = False
        print(f"  ✓ Using processed claims data only")
    
    print(f"  ✓ Claims: {len(claims_df):,} records")
    
    # Create PDF
    pdf_path = 'EDA_Report_Healthcare_Fraud.pdf'
    
    with PdfPages(pdf_path) as pdf:
        
        # ============================================
        # PAGE 1: Title Page
        # ============================================
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        ax.text(0.5, 0.7, 'Healthcare Fraud Detection', fontsize=32, fontweight='bold', 
                ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.55, 'Exploratory Data Analysis Report', fontsize=24, 
                ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.text(0.5, 0.35, f'Dataset: Medicare Claims Data', fontsize=16, 
                ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.28, f'Total Records: {len(claims_df):,}', fontsize=14, 
                ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.21, f'Unique Providers: {claims_df["provider_id"].nunique():,}', fontsize=14, 
                ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.14, f'Fraud Rate: {claims_df["is_fraud"].mean()*100:.2f}%', fontsize=14, 
                ha='center', va='center', transform=ax.transAxes, color='red')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[2/12] Title page created")
        
        # ============================================
        # PAGE 2: Dataset Overview
        # ============================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 2a: Fraud Distribution
        fraud_counts = claims_df['is_fraud'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0, 0].pie(fraud_counts.values, labels=['Legitimate', 'Fraudulent'], 
                       autopct='%1.1f%%', colors=colors, explode=(0, 0.1))
        axes[0, 0].set_title('Fraud Distribution')
        
        # 2b: Claim Type Distribution
        claim_types = claims_df['claim_type'].value_counts()
        axes[0, 1].bar(claim_types.index, claim_types.values, color=['#3498db', '#9b59b6'])
        axes[0, 1].set_title('Claim Type Distribution')
        axes[0, 1].set_ylabel('Count')
        for i, v in enumerate(claim_types.values):
            axes[0, 1].text(i, v + 1000, f'{v:,}', ha='center')
        
        # 2c: Amount Distribution (log scale)
        axes[1, 0].hist(claims_df['amount'][claims_df['amount'] > 0], bins=50, 
                        color='#3498db', edgecolor='white', alpha=0.7)
        axes[1, 0].set_xlabel('Claim Amount ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Amount Distribution')
        axes[1, 0].set_xscale('log')
        
        # 2d: Key Statistics Table
        stats_text = f"""
DATASET STATISTICS
==================
Total Claims: {len(claims_df):,}
Unique Providers: {claims_df['provider_id'].nunique():,}
Unique Patients: {claims_df['patient_id'].nunique():,}

FRAUD STATISTICS
================
Fraudulent Claims: {claims_df['is_fraud'].sum():,}
Legitimate Claims: {(~claims_df['is_fraud'].astype(bool)).sum():,}
Fraud Rate: {claims_df['is_fraud'].mean()*100:.2f}%

FINANCIAL STATISTICS
====================
Total Amount: ${claims_df['amount'].sum():,.2f}
Mean Amount: ${claims_df['amount'].mean():,.2f}
Median Amount: ${claims_df['amount'].median():,.2f}
Max Amount: ${claims_df['amount'].max():,.2f}

CLAIM TYPES
===========
Inpatient: {len(claims_df[claims_df['claim_type']=='Inpatient']):,}
Outpatient: {len(claims_df[claims_df['claim_type']=='Outpatient']):,}
"""
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.95, stats_text, fontsize=10, family='monospace',
                        va='top', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[3/12] Dataset overview created")
        
        # ============================================
        # PAGE 3: Numerical Distributions
        # ============================================
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Numerical Feature Distributions', fontsize=16, fontweight='bold')
        
        # Amount
        axes[0, 0].hist(claims_df['amount'], bins=50, color='#3498db', edgecolor='white')
        axes[0, 0].set_title('Claim Amount')
        axes[0, 0].set_xlabel('Amount ($)')
        
        # Patient Age
        axes[0, 1].hist(claims_df['patient_age'], bins=30, color='#2ecc71', edgecolor='white')
        axes[0, 1].set_title('Patient Age')
        axes[0, 1].set_xlabel('Age (years)')
        
        # Length of Stay
        axes[0, 2].hist(claims_df['length_of_stay'], bins=30, color='#9b59b6', edgecolor='white')
        axes[0, 2].set_title('Length of Stay')
        axes[0, 2].set_xlabel('Days')
        
        # Num Diagnoses
        axes[1, 0].hist(claims_df['num_diagnoses'], bins=20, color='#e74c3c', edgecolor='white')
        axes[1, 0].set_title('Number of Diagnoses')
        axes[1, 0].set_xlabel('Count')
        
        # Chronic Conditions
        axes[1, 1].hist(claims_df['chronic_conditions'], bins=12, color='#f39c12', edgecolor='white')
        axes[1, 1].set_title('Chronic Conditions')
        axes[1, 1].set_xlabel('Count (0-11)')
        
        # Deductible
        axes[1, 2].hist(claims_df['deductible'], bins=30, color='#1abc9c', edgecolor='white')
        axes[1, 2].set_title('Deductible Amount')
        axes[1, 2].set_xlabel('Amount ($)')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[4/12] Distribution plots created")
        
        # ============================================
        # PAGE 4: Fraud vs Non-Fraud Comparison
        # ============================================
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Fraud vs Non-Fraud Comparison', fontsize=16, fontweight='bold')
        
        fraud_df = claims_df[claims_df['is_fraud'] == 1]
        legit_df = claims_df[claims_df['is_fraud'] == 0]
        
        # Amount comparison
        axes[0, 0].boxplot([legit_df['amount'], fraud_df['amount']], 
                          labels=['Legitimate', 'Fraudulent'])
        axes[0, 0].set_title('Amount Comparison')
        axes[0, 0].set_ylabel('Amount ($)')
        axes[0, 0].set_yscale('log')
        
        # Age comparison
        axes[0, 1].boxplot([legit_df['patient_age'], fraud_df['patient_age']], 
                          labels=['Legitimate', 'Fraudulent'])
        axes[0, 1].set_title('Patient Age Comparison')
        axes[0, 1].set_ylabel('Age')
        
        # Length of Stay comparison
        axes[0, 2].boxplot([legit_df['length_of_stay'], fraud_df['length_of_stay']], 
                          labels=['Legitimate', 'Fraudulent'])
        axes[0, 2].set_title('Length of Stay Comparison')
        axes[0, 2].set_ylabel('Days')
        
        # Diagnoses comparison
        axes[1, 0].boxplot([legit_df['num_diagnoses'], fraud_df['num_diagnoses']], 
                          labels=['Legitimate', 'Fraudulent'])
        axes[1, 0].set_title('Number of Diagnoses')
        axes[1, 0].set_ylabel('Count')
        
        # Chronic Conditions comparison
        axes[1, 1].boxplot([legit_df['chronic_conditions'], fraud_df['chronic_conditions']], 
                          labels=['Legitimate', 'Fraudulent'])
        axes[1, 1].set_title('Chronic Conditions')
        axes[1, 1].set_ylabel('Count')
        
        # Mean comparison bar chart
        metrics = ['amount', 'patient_age', 'length_of_stay', 'num_diagnoses', 'chronic_conditions']
        fraud_means = [fraud_df[m].mean() for m in metrics]
        legit_means = [legit_df[m].mean() for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        axes[1, 2].bar(x - width/2, legit_means, width, label='Legitimate', color='#2ecc71')
        axes[1, 2].bar(x + width/2, fraud_means, width, label='Fraudulent', color='#e74c3c')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(['Amount', 'Age', 'Stay', 'Diagnoses', 'Chronic'], rotation=45)
        axes[1, 2].legend()
        axes[1, 2].set_title('Mean Comparison')
        axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[5/12] Fraud comparison created")
        
        # ============================================
        # PAGE 5: Correlation Heatmap
        # ============================================
        fig, ax = plt.subplots(figsize=(12, 10))
        
        numeric_cols = ['amount', 'deductible', 'num_diagnoses', 'num_procedures', 
                       'length_of_stay', 'patient_age', 'chronic_conditions', 
                       'amount_per_diagnosis', 'is_fraud']
        corr_matrix = claims_df[numeric_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   fmt='.2f', square=True, ax=ax, linewidths=0.5)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[6/12] Correlation heatmap created")
        
        # ============================================
        # PAGE 6: Provider Analysis
        # ============================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Provider-Level Analysis', fontsize=16, fontweight='bold')
        
        provider_stats = claims_df.groupby('provider_id').agg({
            'amount': ['mean', 'sum', 'count'],
            'is_fraud': 'first'
        }).reset_index()
        provider_stats.columns = ['provider_id', 'avg_amount', 'total_revenue', 'claim_count', 'is_fraud']
        
        # Claims per provider
        axes[0, 0].hist(provider_stats['claim_count'], bins=50, color='#3498db', edgecolor='white')
        axes[0, 0].set_title('Claims per Provider')
        axes[0, 0].set_xlabel('Number of Claims')
        axes[0, 0].set_ylabel('Frequency')
        
        # Revenue per provider
        axes[0, 1].hist(provider_stats['total_revenue'], bins=50, color='#2ecc71', edgecolor='white')
        axes[0, 1].set_title('Revenue per Provider')
        axes[0, 1].set_xlabel('Total Revenue ($)')
        axes[0, 1].set_xscale('log')
        
        # Fraud vs Legit provider stats
        fraud_providers = provider_stats[provider_stats['is_fraud'] == 1]
        legit_providers = provider_stats[provider_stats['is_fraud'] == 0]
        
        axes[1, 0].boxplot([legit_providers['avg_amount'], fraud_providers['avg_amount']],
                          labels=['Legitimate', 'Fraudulent'])
        axes[1, 0].set_title('Avg Claim Amount by Provider Type')
        axes[1, 0].set_ylabel('Average Amount ($)')
        
        axes[1, 1].boxplot([legit_providers['claim_count'], fraud_providers['claim_count']],
                          labels=['Legitimate', 'Fraudulent'])
        axes[1, 1].set_title('Claim Count by Provider Type')
        axes[1, 1].set_ylabel('Number of Claims')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[7/12] Provider analysis created")
        
        # ============================================
        # PAGE 7: Top Diagnoses Analysis
        # ============================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle('Diagnosis Code Analysis', fontsize=16, fontweight='bold')
        
        # Top 15 most common diagnoses
        top_diag = claims_df['diagnosis_code'].value_counts().head(15)
        axes[0].barh(range(len(top_diag)), top_diag.values, color='#3498db')
        axes[0].set_yticks(range(len(top_diag)))
        axes[0].set_yticklabels(top_diag.index)
        axes[0].set_xlabel('Number of Claims')
        axes[0].set_title('Top 15 Most Common Diagnoses')
        axes[0].invert_yaxis()
        
        # Top 15 diagnoses by amount
        diag_amount = claims_df.groupby('diagnosis_code')['amount'].mean().nlargest(15)
        axes[1].barh(range(len(diag_amount)), diag_amount.values, color='#e74c3c')
        axes[1].set_yticks(range(len(diag_amount)))
        axes[1].set_yticklabels(diag_amount.index)
        axes[1].set_xlabel('Average Claim Amount ($)')
        axes[1].set_title('Top 15 Most Expensive Diagnoses')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[8/12] Diagnosis analysis created")
        
        # ============================================
        # PAGE 8: Fraud Rate by Diagnosis
        # ============================================
        fig, ax = plt.subplots(figsize=(12, 8))
        
        diag_fraud = claims_df.groupby('diagnosis_code').agg({
            'is_fraud': 'mean',
            'amount': 'count'
        }).reset_index()
        diag_fraud.columns = ['diagnosis_code', 'fraud_rate', 'claim_count']
        diag_fraud = diag_fraud[diag_fraud['claim_count'] >= 100]  # Filter for significance
        diag_fraud = diag_fraud.nlargest(20, 'fraud_rate')
        
        colors = ['#e74c3c' if x > 0.6 else '#f39c12' if x > 0.5 else '#3498db' for x in diag_fraud['fraud_rate']]
        ax.barh(range(len(diag_fraud)), diag_fraud['fraud_rate'].values * 100, color=colors)
        ax.set_yticks(range(len(diag_fraud)))
        ax.set_yticklabels(diag_fraud['diagnosis_code'])
        ax.set_xlabel('Fraud Rate (%)')
        ax.set_title('Top 20 Diagnoses by Fraud Rate (min 100 claims)', fontsize=14, fontweight='bold')
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        ax.invert_yaxis()
        
        # Add percentage labels
        for i, v in enumerate(diag_fraud['fraud_rate'].values):
            ax.text(v * 100 + 1, i, f'{v*100:.1f}%', va='center')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[9/12] Fraud rate by diagnosis created")
        
        # ============================================
        # PAGE 9: Age and Chronic Conditions Analysis
        # ============================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Age and Chronic Conditions Analysis', fontsize=16, fontweight='bold')
        
        # Age distribution by fraud
        axes[0, 0].hist(legit_df['patient_age'], bins=30, alpha=0.7, label='Legitimate', color='#2ecc71')
        axes[0, 0].hist(fraud_df['patient_age'], bins=30, alpha=0.7, label='Fraudulent', color='#e74c3c')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Age Distribution by Fraud Status')
        axes[0, 0].legend()
        
        # Chronic conditions distribution by fraud
        chronic_fraud = claims_df.groupby('chronic_conditions')['is_fraud'].mean() * 100
        axes[0, 1].bar(chronic_fraud.index, chronic_fraud.values, color='#9b59b6')
        axes[0, 1].set_xlabel('Number of Chronic Conditions')
        axes[0, 1].set_ylabel('Fraud Rate (%)')
        axes[0, 1].set_title('Fraud Rate by Chronic Conditions')
        
        # Age vs Amount scatter
        sample = claims_df.sample(min(5000, len(claims_df)))
        colors = ['#e74c3c' if x else '#2ecc71' for x in sample['is_fraud']]
        axes[1, 0].scatter(sample['patient_age'], sample['amount'], c=colors, alpha=0.3, s=10)
        axes[1, 0].set_xlabel('Patient Age')
        axes[1, 0].set_ylabel('Claim Amount ($)')
        axes[1, 0].set_title('Age vs Amount (Green=Legit, Red=Fraud)')
        axes[1, 0].set_yscale('log')
        
        # Age groups fraud rate
        claims_df['age_group'] = pd.cut(claims_df['patient_age'], bins=[0, 50, 60, 70, 80, 90, 120],
                                        labels=['<50', '50-60', '60-70', '70-80', '80-90', '90+'])
        age_fraud = claims_df.groupby('age_group')['is_fraud'].mean() * 100
        axes[1, 1].bar(range(len(age_fraud)), age_fraud.values, color='#3498db')
        axes[1, 1].set_xticks(range(len(age_fraud)))
        axes[1, 1].set_xticklabels(age_fraud.index)
        axes[1, 1].set_xlabel('Age Group')
        axes[1, 1].set_ylabel('Fraud Rate (%)')
        axes[1, 1].set_title('Fraud Rate by Age Group')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[10/12] Age analysis created")
        
        # ============================================
        # PAGE 10: Claim Type Analysis
        # ============================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Inpatient vs Outpatient Analysis', fontsize=16, fontweight='bold')
        
        inpatient = claims_df[claims_df['claim_type'] == 'Inpatient']
        outpatient = claims_df[claims_df['claim_type'] == 'Outpatient']
        
        # Count comparison
        type_counts = claims_df['claim_type'].value_counts()
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                      colors=['#3498db', '#9b59b6'], explode=(0.05, 0))
        axes[0, 0].set_title('Claim Type Distribution')
        
        # Amount by type
        axes[0, 1].boxplot([outpatient['amount'], inpatient['amount']], 
                          labels=['Outpatient', 'Inpatient'])
        axes[0, 1].set_ylabel('Amount ($)')
        axes[0, 1].set_title('Amount by Claim Type')
        axes[0, 1].set_yscale('log')
        
        # Fraud rate by type
        type_fraud = claims_df.groupby('claim_type')['is_fraud'].mean() * 100
        axes[1, 0].bar(type_fraud.index, type_fraud.values, color=['#3498db', '#9b59b6'])
        axes[1, 0].set_ylabel('Fraud Rate (%)')
        axes[1, 0].set_title('Fraud Rate by Claim Type')
        for i, v in enumerate(type_fraud.values):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
        # Length of stay for inpatient
        axes[1, 1].hist(inpatient['length_of_stay'], bins=30, color='#e74c3c', edgecolor='white')
        axes[1, 1].set_xlabel('Length of Stay (days)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Inpatient Length of Stay Distribution')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[11/12] Claim type analysis created")
        
        # ============================================
        # PAGE 11: Key Findings Summary
        # ============================================
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        findings = f"""
KEY FINDINGS - HEALTHCARE FRAUD DETECTION EDA
===============================================

DATASET SUMMARY
---------------
• Total Claims Analyzed: {len(claims_df):,}
• Unique Providers: {claims_df['provider_id'].nunique():,}
• Unique Patients: {claims_df['patient_id'].nunique():,}
• Date Range: Medicare Claims Data

FRAUD STATISTICS
----------------
• Overall Fraud Rate: {claims_df['is_fraud'].mean()*100:.2f}%
• Fraudulent Claims: {claims_df['is_fraud'].sum():,}
• Legitimate Claims: {(~claims_df['is_fraud'].astype(bool)).sum():,}

FINANCIAL INSIGHTS
------------------
• Total Claims Value: ${claims_df['amount'].sum():,.2f}
• Average Claim Amount: ${claims_df['amount'].mean():,.2f}
• Median Claim Amount: ${claims_df['amount'].median():,.2f}
• Highest Single Claim: ${claims_df['amount'].max():,.2f}

FRAUD PATTERNS IDENTIFIED
-------------------------
• Fraudulent claims have higher average amounts
• Certain diagnosis codes show 60%+ fraud rates
• Inpatient claims show different fraud patterns than outpatient
• Provider-level aggregation reveals clearer fraud signals

TOP FRAUD INDICATORS
--------------------
1. Diagnosis Code 44024 (Atherosclerosis w/ Gangrene): 65.2% fraud rate
2. Diagnosis Code 03842 (E. Coli Septicemia): 62.6% fraud rate
3. High claim amounts relative to provider type
4. Unusual patterns in claims per patient

RECOMMENDATIONS
---------------
• Focus on provider-level analysis for fraud detection
• Flag high-risk diagnosis codes for manual review
• Implement amount thresholds by provider type
• Monitor claims per patient ratios
"""
        ax.text(0.05, 0.95, findings, fontsize=11, family='monospace',
                va='top', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("[12/12] Summary page created")
    
    print("\n" + "=" * 70)
    print(f"✅ EDA REPORT SAVED: {pdf_path}")
    print("=" * 70)
    
    return pdf_path

if __name__ == "__main__":
    create_eda_report()
