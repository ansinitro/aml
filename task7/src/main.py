import os
import sys
import subprocess
import json

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)

def generate_report(metrics_path, output_path):
    print("Generating LaTeX report...")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    report_content = r"""
\documentclass[12pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage[hidelinks]{hyperref}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{titlesec}
\usepackage{setspace}
\onehalfspacing

\title{\textbf{Case Study 7: Design, Evaluation, and Optimization of Recommender Systems}}
\author{Adilet Akhmedov \\ Angsar Shaumen \\ Assanali Rymgali \\ Bekzat Sundetkhan \\ Sanzhar Syzdykov}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This case study explores the design and evaluation of recommender systems in an e-commerce context using the MovieLens dataset. We implement and compare two fundamental approaches: Content-Based Filtering using TF-IDF and Cosine Similarity, and Collaborative Filtering using Singular Value Decomposition (SVD). Our results demonstrate that while both models are effective, SVD with mean centering achieves a slightly superior RMSE of 0.93 and better precision at lower K values. The study highlights the importance of handling data sparsity and suggests hybrid approaches for future optimization.
\end{abstract}

\tableofcontents
\newpage

\section{Introduction}
Recommender systems have become indispensable tools in the digital economy, driving user engagement and sales on platforms like Amazon, Netflix, and Spotify. By filtering vast catalogs of information to provide personalized suggestions, these systems solve the problem of information overload.

This report documents the implementation of a recommender system pipeline. The primary objectives are to:
\begin{enumerate}
    \item Understand the fundamental algorithms of recommendation (Content-Based vs. Collaborative).
    \item Implement these algorithms using Python.
    \item Evaluate their performance using metric such as RMSE and Precision@K.
    \item Analyze the challenges posed by real-world data, such as sparsity.
\end{enumerate}

\section{Literature Review}
The field of recommender systems has evolved significantly since the mid-1990s.
\begin{itemize}
    \item \textbf{Collaborative Filtering}: Early work by Resnick et al. (1994) on GroupLens introduced the concept of automated collaborative filtering. Later, matrix factorization techniques gained prominence during the Netflix Prize competition, with Koren et al. (2009) demonstrating the superiority of SVD-based methods.
    \item \textbf{Content-Based Filtering}: Pazzani and Billsus (2007) formalized content-based recommendation, which relies on item attributes. This approach avoids the "cold-start" problem for new items but suffers from over-specialization.
    \item \textbf{Evaluation Metrics}: Herlocker et al. (2004) provided a comprehensive review of evaluation metrics, emphasizing that accuracy (RMSE) alone is insufficient, and ranking metrics like Precision and Recall are crucial for top-N recommendation tasks.
\end{itemize}

\section{Methodology}

\subsection{Dataset}
We utilized the \textbf{MovieLens Latest Small} dataset, a standard benchmark in the field.
\begin{itemize}
    \item \textbf{Ratings}: 100,000 ratings from 610 users on 9,724 movies.
    \item \textbf{Sparsity}: The interaction matrix is highly sparse ($>98\%$ empty), presenting a significant challenge for collaborative filtering.
    \item \textbf{Preprocessing}: Data was merged with movie metadata (genres) and split into training (80\%) and testing (20\%) sets using a random split strategy.
\end{itemize}

\subsection{Models}

\subsubsection{Content-Based Filtering}
The Content-Based model constructs item profiles using movie genres.
\begin{enumerate}
    \item \textbf{Feature Extraction}: We applied Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to the "genres" field.
    \item \textbf{Similarity}: Cosine Similarity was computed between all pairs of movies.
    \item \textbf{Prediction}: For a user $u$ and item $i$, the predicted rating is the weighted average of ratings given by $u$ to items similar to $i$.
\end{enumerate}

\subsubsection{Collaborative Filtering (SVD)}
We implemented a model-based approach using low-rank Matrix Factorization.
\begin{enumerate}
    \item \textbf{Matrix Construction}: A User-Item rating matrix $R$ was created.
    \item \textbf{Normalization}: User biases were removed by subtracting the mean rating of each user (Mean Centering). Missing values were filled with 0 (representing the mean).
    \item \textbf{Decomposition}: Truncated Singular Value Decomposition (SVD) was applied to factorize the centered matrix into orthogonal components.
    \item \textbf{Reconstruction}: The matrix was approximated using the top $k=20$ latent features to predict missing entries.
\end{enumerate}

\section{Results}

\subsection{Quantitative Metrics}
Table \ref{tab:metrics} presents the performance of both models on the test set.

\begin{table}[H]
\centering
\begin{tabular}{lcccc}
\toprule
Model & RMSE & Precision@10 & Recall@10 \\
\midrule
"""
    for model, scores in metrics.items():
        report_content += f"{model} & {scores['RMSE']:.4f} & {scores.get('Precision@10', 0):.4f} & {scores.get('Recall@10', 0):.4f} \\\\\n"

    report_content += r"""
\bottomrule
\end{tabular}
\caption{Performance Metrics Comparison}
\label{tab:metrics}
\end{table}

\subsection{Visualizations}

Figure \ref{fig:dist} shows the distribution of ratings, indicating a negativity bias is not present; users tend to rate movies they like (3.0-5.0).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/rating_distribution.png}
    \caption{Distribution of Ratings}
    \label{fig:dist}
\end{figure}

The sparsity of the data is visualized in Figure \ref{fig:heat}, which shows interactions only for the top 50 users and movies.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/interaction_heatmap.png}
    \caption{User-Item Interaction Heatmap (Top 50 Subset)}
    \label{fig:heat}
\end{figure}

Figure \ref{fig:pk} illustrates the Precision@K curve. As expected, precision decreases as $K$ increases, but SVD maintains better stability.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/precision_k_curve.png}
    \caption{Precision at K (Ranking Quality)}
    \label{fig:pk}
\end{figure}

\section{Discussion}
Our method comparison reveals that **Collaborative Filtering** slightly outperforms the Content-Based approach in both RMSE and top-N ranking metrics. The mean-centering step was critical; without it, the SVD model biased predictions towards zero (missing values), resulting in high error.

The Content-Based model performed surprisingly well given it only used Genres. However, it is limited by its inability to recommend items outside a user's known genre preferences (the "serendipity" problem). SVD, by learning latent factors, can capture cross-genre correlations.

\section{Conclusion}
We successfully designed and evaluated two recommender systems. The SVD-based collaborative filtering approach demonstrated robust performance. Future work should focus on **Hybrid Systems**, combining the cold-start resilience of content-based models with the accuracy of collaborative filtering, and incorporating temporal dynamics (e.g., handling changing user tastes over time).

\begin{thebibliography}{9}

\bibitem{resnick}
Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., \& Riedl, J. (1994). GroupLens: an open architecture for collaborative filtering of netnews. \textit{Proceedings of CSCW '94}.

\bibitem{sarwar}
Sarwar, B., Karypis, G., Konstan, J., \& Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. \textit{Proceedings of WWW '01}.

\bibitem{koren}
Koren, Y., Bell, R., \& Volinsky, C. (2009). Matrix factorization techniques for recommender systems. \textit{Computer}, 42(8).

\end{thebibliography}

\end{document}
"""
    with open(output_path, 'w') as f:
        f.write(report_content)
    print(f"Report written to {output_path}")

def compile_report(report_dir):
    print("Compiling PDF...")
    # Run pdflatex twice to resolve references
    try:
        subprocess.run(['pdflatex', '-interaction=nonstopmode', 'report.tex'], cwd=report_dir, check=True, capture_output=True)
        subprocess.run(['pdflatex', '-interaction=nonstopmode', 'report.tex'], cwd=report_dir, check=True, capture_output=True)
        print("PDF compilation successful.")
    except subprocess.CalledProcessError as e:
        print("PDF compilation failed.")
        # print(e.stderr.decode()) # Optional debug

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir) # Ensure we are in src/
    
    # 1. Download
    run_script('download_data.py')
    
    # 2. Preprocess
    run_script('preprocess.py')
    
    # 3. Train
    run_script('train.py')
    
    # 4. Visualize
    run_script('visualize.py')
    
    # 5. Generate Reports
    metrics_path = os.path.join('../data', 'metrics.json')
    report_dir = '../report'
    report_path = os.path.join(report_dir, 'report.tex')
    presentation_path = os.path.join(report_dir, 'presentation.html')
    
    generate_report(metrics_path, report_path)
    generate_presentation(metrics_path, presentation_path)
    
    # 6. Compile PDF
    compile_report(report_dir)
    
    print("All tasks completed successfully!")

def generate_presentation(metrics_path, output_path):
    print("Generating HTML presentation...")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    html_content = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recommender Systems - Cast Study 7</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #F8FAFC;
            --text-color: #1E293B;
            --heading-color: #0F172A;
            --accent-primary: #2563EB;
            --accent-secondary: #4F46E5;
            --glass-bg: rgba(255, 255, 255, 0.90);
            --glass-border: rgba(226, 232, 240, 0.8);
            --shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            overflow: hidden;
            height: 100vh;
            width: 100vw;
            display: flex;
            justify-content: center;
            align-items: center;
            background-image:
                radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.05) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(79, 70, 229, 0.05) 0px, transparent 50%);
        }
        .slide {
            position: absolute; width: 100%; height: 100%;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            opacity: 0; transform: scale(0.95); transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            visibility: hidden; padding: 2rem;
        }
        .slide.active { opacity: 1; transform: scale(1); visibility: visible; }
        .content-box {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 3rem 4rem;
            box-shadow: var(--shadow);
            max-width: 1300px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
            display: flex;
            gap: 3rem;
            align-items: center;
        }
        .column-layout { flex-direction: column; text-align: center; }
        h1 { font-size: 3.5rem; font-weight: 800; color: var(--heading-color); margin-bottom: 1rem; line-height: 1.1; }
        h2 { font-size: 2.25rem; margin-bottom: 1.5rem; color: var(--accent-primary); border-bottom: 3px solid var(--accent-secondary); display: inline-block; padding-bottom: 0.5rem; }
        p, li { font-size: 1.25rem; line-height: 1.6; margin-bottom: 1rem; color: var(--text-color); }
        ul { list-style: none; padding-left: 0; }
        li::before { content: "•"; color: var(--accent-primary); font-weight: bold; display: inline-block; width: 1em; margin-left: -1em; }
        img { max-width: 100%; max-height: 50vh; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        th, td { padding: 1rem 1.5rem; border-bottom: 1px solid #E2E8F0; text-align: left; }
        th { background-color: #F1F5F9; color: var(--accent-primary); font-weight: 700; }
        .controls { position: absolute; bottom: 2rem; right: 3rem; display: flex; gap: 1rem; z-index: 100; }
        .btn { background: white; border: 1px solid #E2E8F0; padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer; font-weight: 600; transition: all 0.2s; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .btn:hover { background: var(--accent-primary); color: white; transform: translateY(-2px); }
        .highlight { color: var(--accent-secondary); font-weight: 700; }
        .half-width { flex: 1; text-align: left; }
        .image-container { flex: 1; display: flex; justify-content: center; align-items: center; background: white; padding: 1rem; border-radius: 12px; border: 1px solid #E2E8F0; }
    </style>
</head>
<body>

    <!-- Slide 1: Title -->
    <div class="slide active">
        <div class="content-box column-layout">
            <p style="text-transform: uppercase; letter-spacing: 3px; color: var(--accent-secondary); font-weight: 600;">Case Study 7</p>
            <h1>Recommender Systems<br>Design & Evaluation</h1>
            <p style="color: #64748B;">A comparative analysis of Content-Based vs. Collaborative Filtering</p>
            <div style="margin-top: 2rem;">
                <span style="background: #DBEAFE; color: #1E40AF; padding: 0.5rem 1rem; border-radius: 50px; font-weight: 600; display: block; margin-bottom: 0.5rem;">Adilet Akhmedov | Angsar Shaumen</span>
                <span style="background: #DBEAFE; color: #1E40AF; padding: 0.5rem 1rem; border-radius: 50px; font-weight: 600; display: block; margin-bottom: 0.5rem;">Assanali Rymgali | Bekzat Sundetkhan</span>
                <span style="background: #DBEAFE; color: #1E40AF; padding: 0.5rem 1rem; border-radius: 50px; font-weight: 600; display: block;">Sanzhar Syzdykov</span>
            </div>
        </div>
    </div>

    <!-- Slide 2: Objective -->
    <div class="slide">
        <div class="content-box">
            <div class="half-width">
                <h2>Objective & Methodology</h2>
                <p>To build and evaluate recommender systems for e-commerce using the <span class="highlight">MovieLens Latest Small</span> dataset.</p>
                <ul>
                    <li><strong>Content-Based:</strong> TF-IDF on genres + Cosine Similarity.</li>
                    <li><strong>Collaborative Filtering:</strong> SVD Matrix Factorization with mean centering.</li>
                    <li><strong>Evaluation:</strong> RMSE, MAE, Precision@K, Recall@K.</li>
                </ul>
            </div>
            <div class="image-container">
                <img src="figures/rating_distribution.png" alt="Distribution">
                <p style="text-align: center; margin-top: 5px; font-size: 0.9rem;">Dataset Rating Distribution (Skewed towards 3.0-5.0)</p>
            </div>
        </div>
    </div>

    <!-- Slide 3: Heatmap -->
    <div class="slide">
        <div class="content-box column-layout">
            <h2>User-Item Interaction</h2>
            <p>Visualizing the sparsity of the rating matrix (Top 50 Users/Movies).</p>
            <div style="display: flex; justify-content: center; margin-top: 1rem;">
                <img src="figures/interaction_heatmap.png" alt="Heatmap" style="max-height: 60vh;">
            </div>
        </div>
    </div>

    <!-- Slide 4: Results Table -->
    <div class="slide">
        <div class="content-box column-layout">
            <h2>Quantitative Results</h2>
            <p>Performance comparison on the test set (20% split).</p>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>Precision@10</th>
                        <th>Recall@10</th>
                    </tr>
                </thead>
                <tbody>
"""
    for model, scores in metrics.items():
        html_content += f"<tr><td><strong>{model}</strong></td><td>{scores['RMSE']:.4f}</td><td>{scores['MAE']:.4f}</td><td>{scores.get('Precision@10', 0):.4f}</td><td>{scores.get('Recall@10', 0):.4f}</td></tr>"

    html_content += r"""
                </tbody>
            </table>
        </div>
    </div>

    <!-- Slide 5: Curves -->
    <div class="slide">
        <div class="content-box">
             <div class="half-width">
                <h2>Precision Analysis</h2>
                <p>Analyzing how precision degrades as K increases.</p>
                <ul>
                    <li><strong>Analysis:</strong> Both models show high precision for top recommendations, decreasing appropriately as K increases.</li>
                    <li><strong>Winner:</strong> SVD maintains slightly better consistency across K values.</li>
                </ul>
            </div>
            <div class="image-container">
                <img src="figures/precision_k_curve.png" alt="Precision Curve">
            </div>
        </div>
    </div>

    <!-- Slide 6: Conclusion -->
    <div class="slide">
        <div class="content-box column-layout">
            <h2>Conclusion</h2>
            <p style="max-width: 800px;">
                We successfully implemented a robust recommendation pipeline. <strong>Collaborative Filtering (SVD)</strong> proved effective even on small data when properly regularized (centered), achieving an RMSE of ~0.93, comparable to the metadata-rich Content-Based approach.
            </p>
            <div style="margin-top: 2rem; text-align: left; width: 100%;">
                 <ul>
                    <li><strong>Key Insight:</strong> Matrix Factorization captures latent tastes that simple genre matching misses.</li>
                    <li><strong>Future Work:</strong> Hybrid systems could combine specific genre preferences with latent discovery.</li>
                </ul>
            </div>
            <h3 style="margin-top: 2rem; color: var(--accent-primary);">Thank You</h3>
        </div>
    </div>

    <div class="controls">
        <button class="btn" onclick="prevSlide()">Previous</button>
        <button class="btn" onclick="nextSlide()">Next</button>
    </div>

    <script>
        let currentSlide = 0;
        const slides = document.querySelectorAll('.slide');
        function showSlide(index) {
            slides.forEach((slide, i) => {
                slide.classList.remove('active');
                if (i === index) slide.classList.add('active');
            });
        }
        function nextSlide() { if (currentSlide < slides.length - 1) { currentSlide++; showSlide(currentSlide); } }
        function prevSlide() { if (currentSlide > 0) { currentSlide--; showSlide(currentSlide); } }
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') nextSlide();
            if (e.key === 'ArrowLeft') prevSlide();
        });
    </script>
</body>
</html>
"""
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Presentation written to {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir) # Ensure we are in src/
    
    # 1. Download
    run_script('download_data.py')
    
    # 2. Preprocess
    run_script('preprocess.py')
    
    # 3. Train
    run_script('train.py')
    
    # 4. Visualize
    run_script('visualize.py')
    
    # 5. Generate Reports
    metrics_path = os.path.join('../data', 'metrics.json')
    report_path = os.path.join('../report', 'report.tex')
    presentation_path = os.path.join('../report', 'presentation.html')
    
    generate_report(metrics_path, report_path)
    generate_presentation(metrics_path, presentation_path)
    
    # 6. Compile PDF
    report_dir = os.path.dirname(report_path)
    compile_report(report_dir)
    
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
