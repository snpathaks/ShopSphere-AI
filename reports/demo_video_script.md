# 3–5 Minute Demonstration Video Script

**Project**: E-Commerce Product Recommendation Engine  
**Target Duration**: 3 minutes 30 seconds to 4 minutes 30 seconds  
**Objective**: Walk through the problem framing, EDA, model evaluation, live API inference, and Streamlit UI demo to satisfy assessment criteria.

---

### [00:00 - 00:45] Introduction & Problem Framing
- **Visual**: Show project repository (`README.md` and terminal).
- **Speaker**:
  > *"Hello, my name is [Your Name], candidate for the AI & Machine Learning Internship at Algions AI Labs. Today I am presenting my 15-day individual project: an End-to-End Decision-Support E-Commerce Product Recommendation Engine.*
  >
  > *The core business decision is selecting which top-N products to display to a user upon visiting the platform. The prediction unit is a user-item pair, and our target is predicting whether the user will convert on that product in the upcoming period. We treat this strictly as a decision-support system, providing transparent reasoning and clear limitations."*

---

### [00:45 - 01:30] Data Generation & Exploratory Data Analysis (EDA)
- **Visual**: Open `notebooks/01_EDA.ipynb` or summary statistics.
- **Speaker**:
  > *"To ensure 100% legal compliance and zero privacy leakage, we built a synthetic interaction generator in `data/generate_data.py`. With a fixed seed of 42, it models 500 users, 200 items across 10 categories, and 15,004 interactions spanning 180 days.*
  >
  > *Crucially, we incorporate Zipfian popularity skew, 10-dimensional latent user preference vectors, and hierarchical implicit feedback: Views (weight 1), Carts (weight 3), and Purchases (weight 5).*
  >
  > *To prevent future data leakage, we enforce a strict chronological split: the first 150 days form our training set, and the final 30 days are held out for testing. No future interactions leak into model fitting."*

---

### [01:30 - 02:30] Models & Benchmark Evaluation
- **Visual**: Show `notebooks/02_Model_Evaluation.ipynb` or model results table in terminal.
- **Speaker**:
  > *"We implemented and benchmarked four distinct approaches against our Popularity baseline:*
  > *1. Popularity Baseline: Weighted interaction volume.*
  > *2. Item-Item Collaborative Filtering: Cosine similarity over sparse user-item interaction matrices.*
  > *3. Content-Based Filtering: TF-IDF vectorization over product categories, subcategories, brands, and descriptions.*
  > *4. Weighted Hybrid Recommender: An ensemble combining collaborative, content, and popularity signals.*
  >
  > *Evaluating on the held-out test set at K=10:*
  > *- The Popularity baseline achieves NDCG@10 of 0.114 with only 9.5% catalog coverage.*
  > *- Our Hybrid model achieves an NDCG@10 of 0.198—a 73.7% improvement over baseline—with 52% catalog coverage.*
  > *- In our cold-start analysis, the hybrid model dynamically shifts weights toward popularity for new users, preventing cold-start collapse."*

---

### [02:30 - 03:45] Live Inference: FastAPI and Streamlit UI Demonstration
- **Visual**: Switch to terminal running FastAPI, then open browser on `http://localhost:8501`.
- **Speaker**:
  > *"Now let's demonstrate the live inference pipeline. Our FastAPI backend loads the trained model artifacts from the `models/` directory.*
  >
  > *Here in our Streamlit application, let's select User `U0001` and run inference with our Hybrid Recommender.*
  >
  > *[Click 'Get Recommendations']*
  >
  > *Notice that every recommendation displays an explainable reason—such as 'Similar to Sony Headphones you purchased' or 'Trending in Electronics'.*
  >
  > *Now, let's test a cold-start scenario with a brand-new user `U_NEW_999`.*
  >
  > *[Input `U_NEW_999` and click 'Get Recommendations']*
  >
  > *The system detects zero historical events, displays a Cold-Start badge, and smoothly falls back to trending products without crashing.*
  >
  > *We can also explore the 'Model Comparison' tab to review the NDCG, Recall, and Precision curves, and the 'Model Card' tab for responsible AI documentation."*

---

### [03:45 - 04:15] Limitations & Conclusion
- **Visual**: Show `reports/model_card.md` or conclusion slide.
- **Speaker**:
  > *"Finally, we explicitly acknowledge the model's limitations:*
  > *1. Synthetic data may not capture sudden holiday demand spikes.*
  > *2. Popularity bias requires continuous monitoring of catalog coverage.*
  > *3. Feedback loops are mitigated by reserving exploration slots.*
  >
  > *The entire codebase is tested with pytest, fully documented in the README and Technical Report, and reproducible with a single command: `make all`.*
  >
  > *Thank you for your time, and I look forward to the technical walkthrough!"*
