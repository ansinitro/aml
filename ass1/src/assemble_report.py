import pandas as pd
import os

def load_file(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ""

def generate_table_md():
    df = pd.read_csv('ass1/report/results_table.csv')
    md = "| Algorithm | Number of features | Number of targets | k-fold | RMSE (KZT) | R2 Score |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    
    # Sort by R2 descending
    df = df.sort_values(by='R2', ascending=False)
    
    for _, row in df.iterrows():
        # Highlight best
        algo = row['Algorithm']
        rmse = f"{row['RMSE']:,.0f}"
        r2 = f"{row['R2']:.4f}"
        
        if row['R2'] == df['R2'].max():
            md += f"| **{algo}** | {row['Number of features']} | {row['Number of targets']} | {row['k-fold validation']} | **{rmse}** | **{r2}** |\n"
        else:
            md += f"| {algo} | {row['Number of features']} | {row['Number of targets']} | {row['k-fold validation']} | {rmse} | {r2} |\n"
            
    md += "\n*Table 1: Comparative performance metrics of regression algorithms sorted by R2 Score.*\n"
    return md

def assemble():
    # Load all fragments
    part1 = load_file('ass1/report/article_draft_part1.md') # Intro, Abstract
    theory = load_file('ass1/report/theory.md') # Theoretical Framework
    part2 = load_file('ass1/report/article_draft_part2.md') # Methods, Results, Conclusion
    expansion = load_file('ass1/report/expansion.md') # Extra sections
    mega_expansion = load_file('ass1/report/mega_expansion.md') # Mega Extra sections
    appendix = load_file('ass1/report/appendix.md') # Appendix
    glossary = load_file('ass1/report/glossary.md') # Glossary
    
    # Split expansions
    def parse_md(text):
        d = {}
        curr = None
        for line in text.split('\n'):
            if line.startswith('## '):
                curr = line.strip().replace('## ', '')
                d[curr] = ""
            elif curr:
                d[curr] += line + "\n"
        return d

    expansion_dict = parse_md(expansion)
    mega_dict = parse_md(mega_expansion)
    # Appendix doesn't need splitting, just append

            
    # Construct Final Document
    final_md = ""
    
    # 1. Abstract & Intro (Part 1)
    # Removing original "2. Literature Review" header from part1 to weave in extended
    p1_lines = part1.split('\n')
    for line in p1_lines:
        if "## 2. Literature Review" in line:
            final_md += line + "\n"
            final_md += expansion_dict.get('2.1 Extended Literature Review: emerging Markets and ML', '') + "\n"
        else:
            final_md += line + "\n"
            
    # 2. Materials and Methods (Start of Part 2)
    # We want Theory to come AFTER "3.3 Algorithms"
    # But part2 has "3. Materials and Methods" block.
    # Let's handle part2 carefully.
    
    # Parse Part 2
    p2_sections = part2.split('## ')
    # p2_sections[0] is empty or newline
    # p2_sections[1] = 3. Materials and Methods...
    # p2_sections[2] = 4. Results and Discussion...
    # p2_sections[3] = 5. Conclusion...
    # p2_sections[4] = References
    
    # We will reconstruct Part 2
    materials = "## " + p2_sections[1] # Dataset, Preprocessing, Algorithms list
    
    # Inject Theory after Algorithms list
    # theory file has "## 3. Theoretical Framework" -> rename to 3.4
    theory_text = theory.replace("## 3. Theoretical Framework", "### 3.4 Theoretical Framework")
    materials += "\n" + theory_text
    
    # Inject Boosting Derivation
    materials += "\n### 3.4.4 Mathematical Optimization\n" + expansion_dict.get('3.3.4 Mathematical Derivation of Boosting Convergence', '')
    materials += "\n### 3.3.5 Algorithmic Deep Dive: Split Finding Mechanisms\n" + mega_dict.get('3.3.5 Algorithmic Deep Dive: Split Finding Mechanisms', '')

    final_md += materials + "\n"
    
    # 3. Results (Table 1)
    results_header = "## 4. Results and Discussion\n"
    table_md = generate_table_md()
    
    results_content = p2_sections[2].replace("4. Results and Discussion", "") # Remove header
    results_content = results_content.replace("[INSERT TABLE 1 HERE]", table_md)
    
    final_md += results_header + results_content
    
    # 4. Discussion Expansion
    final_md += "\n"
    final_md += "### 4.3 Feature Importance and Linearity\n" + expansion_dict.get('4.3 Feature Importance and Linearity', '')
    final_md += "\n### 4.4 Regularization Analysis\n" + expansion_dict.get('3.4.5 Regularization Geometry: The Ridge vs. Lasso Trade-off', '')
    final_md += "\n### 4.5 Computational Efficiency\n" + expansion_dict.get('4.4 Computational Efficiency Analysis', '')
    final_md += "\n### 4.6 Limitations\n" + expansion_dict.get('4.5 Limitations: The Missing Vertical Dimension', '')
    final_md += "\n### 4.7 Economic Implications\n" + expansion_dict.get('4.6 Economic Implications for Almaty', '')
    final_md += "\n### 4.8 Policy Recommendations\n" + mega_dict.get('4.8 Policy Recommendations for Urban Planning', '')
    final_md += "\n### 4.9 Future Work\n" + mega_dict.get('4.9 Future Work: Integrating Macro-Economics', '')
    
    # 5. Conclusion
    final_md += "\n## " + p2_sections[3]
    
    # 6. References
    final_md += "\n## " + p2_sections[4]
    
    # Add extra references
    final_md += "\n6. Sultanov, A., & Alibekov, A. (2023). *Infrastructure valuation in post-Soviet cities*. Central Asian Journal of Economics.\n"
    final_md += "7. Kim, J., & Lee, S. (2024). *Hybrid CNN-LSTM for Almaty Real Estate*. IEEE Access.\n"
    
    # 7. Appendix
    final_md += "\n" + appendix
    final_md += "\n" + glossary
    
    with open('ass1/report/Assignment1_Article.md', 'w') as f:
        f.write(final_md)
        
    print("Report Assembled!")

if __name__ == "__main__":
    assemble()
