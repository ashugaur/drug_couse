# %% Testing

## Dependencies
from drug_couse.arm.build import (
    simulated_patient_drug_combination,
    DrugAssociationAnalyzer,
    compare_file_sizes,
    load_parquet_data,
)
import os
from pathlib import Path

working_dir = Path("C:/my_disk/____tmp/drug_couse")
Path(working_dir).mkdir(parents=True, exist_ok=True)
os.chdir(working_dir)


## simulated_patient_drug_combination
df = simulated_patient_drug_combination(1000)
df.head()


## Automatically analyze ALL drug pairs with enhanced export
print("\n🚀 Creating Complete Drug Relationships Matrix:")
print("=" * 60)

## Initialize analyzer
analyzer = DrugAssociationAnalyzer(df)

# Step 1: Preprocess data
transactions = analyzer.preprocess_data()
print("\n📋 Transaction Matrix Shape: {transactions.shape}")
print("Sample of transaction matrix:")
print(transactions.head())

# Step 2: Get basic statistics
analyzer.get_drug_statistics()

# Step 3: Find frequent itemsets
# Start with low min_support since we have small dataset
frequent_itemsets = analyzer.find_frequent_itemsets(min_support=0.2)

if frequent_itemsets is not None:
    print("\n📊 Top 10 Frequent Itemsets:")
    print(frequent_itemsets.nlargest(10, "support"))

# Step 4: Generate association rules
rules = analyzer.generate_association_rules(metric="confidence", min_threshold=0.5)

all_pairs_matrix = analyzer.create_all_pairs_relationship_matrix(
    export_to_excel=True, filename="complete_drug_relationships_matrix.xlsx"
)

if all_pairs_matrix is not None:
    print("\n📈 Complete Analysis Summary:")
    print(f"   Total drug pairs analyzed: {len(all_pairs_matrix):,}")
    print(
        f"   High confidence relationships (60%+): {len(all_pairs_matrix[all_pairs_matrix['Confidence'] >= 0.6]):,}"
    )
    print(
        f"   Strong associations (Lift 2.0+): {len(all_pairs_matrix[all_pairs_matrix['Lift'] >= 2.0]):,}"
    )
    print(
        f"   High priority relationships: {len(all_pairs_matrix[all_pairs_matrix['Clinical_Priority'] == 'High']):,}"
    )

    # Compare file sizes for the complete matrix
    compare_file_sizes("complete_drug_relationships_matrix")

    print("\n📁 Output Files Generated:")
    print("   📊 Excel File: complete_drug_relationships_matrix.xlsx")
    print("      • Multiple sheets with filtered views")
    print("      • Pivot tables for easy analysis")
    print("      • Summary statistics")
    print("   📦 Parquet Files: Multiple .parquet files for different views")
    print("      • Faster loading for large datasets")
    print("      • Better compression than CSV")
    print("      • Preserves data types")

# Demonstration of loading parquet data
print("\n🔄 Demonstration: Loading Parquet Data")
sample_parquet = load_parquet_data("complete_drug_relationships_matrix.parquet")
if sample_parquet is not None:
    print("   Sample of loaded data:")
    print(
        f"   {sample_parquet.head(3)[['Drug_A', 'Drug_B', 'Confidence_%', 'Lift', 'Rule_Strength']].to_string(index=False)}"
    )

print("\n" + "=" * 60)
print("🎉 Enhanced Analysis Complete with Dual Export!")
print("📋 Key Features:")
print("   ✅ Excel export with multiple sheets")
print("   ✅ Parquet export for large datasets")
print("   ✅ Automatic handling of Excel row limits")
print("   ✅ File size comparisons")
print("   ✅ Easy data loading utilities")
print("=" * 60)


## Self run
"""
cd test
uv run .\arm_v2_test.py
"""
