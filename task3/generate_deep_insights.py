import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set dark theme for Intel/Ops aesthetic
plt.style.use('dark_background')
sns.set_palette("bright") # High contrast

def load_and_analyze():
    print("Loading dataset...")
    df = pd.read_csv("gtd_dataset.csv", encoding='ISO-8859-1', low_memory=False)
    
    # Basic cleanup
    df = df[['eventid', 'iyear', 'region_txt', 'country_txt', 'attacktype1_txt', 'nkill', 'nwound']].copy()
    
    # 1. Top 5 Countries by Attack Count
    top_countries = df['country_txt'].value_counts().head(5)
    
    plt.figure(figsize=(10, 6))
    ax = top_countries.plot(kind='bar', color='#ff3333', edgecolor='black') # Alert Red
    plt.title("TOP 5 HIGH-RISK ZONES (By Incident Volume)", color='#00ff41', fontname='monospace', fontsize=14, weight='bold') # Terminal Green
    plt.ylabel("Number of Incidents", color='#e0e0e0')
    plt.xticks(rotation=45, ha='right', color='#e0e0e0', fontname='monospace')
    plt.yticks(color='#e0e0e0', fontname='monospace')
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='#555')
    
    # Annotate bars
    for i, v in enumerate(top_countries):
        ax.text(i, v + 100, str(v), color='#00ff41', ha='center', fontname='monospace', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("figures/top_countries_dark.png")
    print("Saved figures/top_countries_dark.png")
    
    # 2. Attack Types in the Most Dangerous Country
    most_dangerous_country = top_countries.index[0]
    country_data = df[df['country_txt'] == most_dangerous_country]
    attack_dist = country_data['attacktype1_txt'].value_counts().head(5)
    
    plt.figure(figsize=(10, 6))
    # Using a donut chart for variety
    wedges, texts, autotexts = plt.pie(attack_dist, labels=attack_dist.index, autopct='%1.1f%%', 
                                       textprops=dict(color="#e0e0e0"), startangle=140, pctdistance=0.85, colors=sns.color_palette("bright"))
    
    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.70,fc='#0a0a0a')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title(f"TACTICAL BREAKDOWN: {most_dangerous_country.upper()}", color='#00ff41', fontname='monospace', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("figures/country_attack_breakdown.png")
    print("Saved figures/country_attack_breakdown.png")
    
    print("\n--- INTEL SUMMARY ---")
    print(f"Most Dangerous Country: {most_dangerous_country} with {top_countries[0]} incidents.")
    total_attacks = len(df)
    percent_top = (top_countries[0] / total_attacks) * 100
    print(f"{most_dangerous_country} accounts for {percent_top:.2f}% of all global incidents.")
    
    region_counts = df['region_txt'].value_counts()
    top_region = region_counts.index[0]
    top_region_pct = (region_counts[0] / total_attacks) * 100
    print(f"Top Region: {top_region} ({top_region_pct:.2f}% of total).")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    load_and_analyze()
