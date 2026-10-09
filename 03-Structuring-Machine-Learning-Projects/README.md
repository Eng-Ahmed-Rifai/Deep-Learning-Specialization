# Course 3: Structuring Machine Learning Projects

**DeepLearning.AI | Instructor: Andrew Ng**  
**Specialization:** Deep Learning Specialization  
**Learner:** Eng. Ahmed Mohamed Abdullatif Rifai  
**Credential:** [Official Certificate (ID: 3IDKEKSZSDB1)](https://www.coursera.org/account/accomplishments/verify/3IDKEKSZSDB1)  
**Completion Status:** 🏆 **100% Completed & Officially Certified**

---

## 🎯 Course Overview & Engineering Principles

This course distills Andrew Ng's years of experience building and shipping large-scale machine learning and deep learning systems into clear, actionable tactical frameworks for structuring machine learning projects.

```
                              ┌─────────────────────────────────────────┐
                              │     Orthogonalization Framework         │
                              └────────────────────┬────────────────────┘
                                                   │
         ┌─────────────────────────┬───────────────┴───────────────┬─────────────────────────┐
         ▼                         ▼                               ▼                         ▼
┌──────────────────┐      ┌──────────────────┐           ┌──────────────────┐      ┌──────────────────┐
│ Fit Training Set │      │   Fit Dev Set    │           │   Fit Test Set   │      │ Real-World Perf  │
│  (Avoid Bias)    │      │  (Avoid Variance)│           │ (Generalization) │      │  (Cost/Product)  │
└──────────────────┘      └──────────────────┘           └──────────────────┘      └──────────────────┘
```

---

## 📚 Curriculum Breakdown

### Week 1: ML Strategy (Part 1)
- **Orthogonalization:** Adjusting one knob at a time to solve independent problems without unintended side effects.
- **Single Real Number Evaluation Metric:** Using F1-Score or weighted metrics rather than precision and recall separately to iterate rapidly.
- **Optimizing vs. Satisficing Metrics:** E.g., maximize accuracy (optimizing) subject to running time $\le 100\text{ ms}$ (satisficing).
- **Train / Dev / Test Distribution Design:**
  - Dev and Test sets **must** come from the exact same distribution reflecting real target usage.
  - Size guidelines for modern big-data regimes ($98\% / 1\% / 1\%$).
- **Comparing to Human-Level Performance & Bayes Optimal Error:**
  - Avoidable Bias: Gap between Bayes error (approximated by human-level error) and training error.
  - Variance: Gap between training error and dev error.

### Week 2: ML Strategy (Part 2)
- **Error Analysis Ceiling:** Manually evaluating 100–500 misclassified examples to determine the maximum potential gain for each category before investing months in collecting new data.
- **Mislabeled Data:**
  - Deep learning algorithms are robust to random errors in the training set.
  - Dev and test sets must have corrected labels to preserve evaluation fidelity and identical distributions.
- **Addressing Data Mismatch:**
  - Using a Training-Dev set (same distribution as Training set) to distinguish between Variance and Data Mismatch.
  - Artificial data synthesis (e.g., adding fog/noise) caution: avoiding accidental over-fitting to the narrow synthesis distribution.
- **Transfer Learning vs. Multi-Task Learning:**
  - *Transfer Learning:* Task A $\to$ Task B when Task A has vastly more data, both share lower-level features (e.g. image edges, shapes), and initial weights are transferred.
  - *Multi-Task Learning:* Training a single network to predict multiple non-mutually exclusive outputs simultaneously with shared low-level representations.
- **End-to-End Deep Learning:**
  - Replaces multi-step handcrafted pipelines with a single unified neural network.
  - Crucial requirement: availability of massive quantities of end-to-end labeled data $(X, Y)$.

---

## 📊 Graded Assessments & Precision

| Module / Case Study | Type | Target Score | Achieved Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Bird Recognition in the City of Peacetopia** | Graded Quiz Case Study | 80.00% | **95.00%** | Passed ✅ |
| **Autonomous Driving Strategy Case Study** | Graded Quiz Case Study | 80.00% | **100.00%** | **Perfect (15/15)** 🏆 |
| **Course 3 Overall Average** | Official Course Grade | 80.00% | **97.50%** | **Certified** 🎓 |

---

## 📜 Verification & Credentials
- **Verification URL:** [https://www.coursera.org/account/accomplishments/verify/3IDKEKSZSDB1](https://www.coursera.org/account/accomplishments/verify/3IDKEKSZSDB1)
- **Archived Vector PDF:** `certificates/DeepLearning_AI_Course3_Certificate_3IDKEKSZSDB1.pdf`
- **Archived High-Res Image:** `certificates/DeepLearning_AI_Course3_Certificate_3IDKEKSZSDB1.jpeg`
