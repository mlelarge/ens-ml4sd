# ENS course: Machine Learning for Scientific Discovery: From Foundations to Applications

Marc Lelarge and Tony Bonnaire

This course introduces the foundations of machine learning, from statistical models to modern deep learning, with a focus on practical applications in scientific research. Students will learn core methods, computational tools, and workflows to apply machine learning techniques to data and problems in their own field of study.

After completing the core curriculum, each department will supervise (over a six-week period) the projects it has proposed. The purpose of the core curriculum is to provide a solid foundation in the fundamentals of statistical learning, along with the essential computing skills (sklearn – PyTorch) required across all projects.

**Mandatory registration [here](https://docs.google.com/forms/d/e/1FAIpQLScs--bQWrZO94LsgB5QG2txksC5G6_zuYEnO2uoDwvz5vQazQ/viewform?usp=dialog)**

Logistic:
- courses: Tuesday 4pm-6pm in room Camille Marbo (29 rue d'Ulm - U205) (except 30/9 in Galois (Ulm Immeuble Rataud))
- practicals: Friday 8:30am-10:30am in room Langevin (29 rue d'Ulm) (except 3/10 in Borel)

**Prerequisites**  
- Proficiency in Python (see [tutorial](https://cs231n.github.io/python-numpy-tutorial/) for review)  
- Basic knowledge of calculus and linear algebra  
- Basic knowledge of probability and statistics  

## Session 1: [Machine Learning for Scientific Discovery 2025 - Course Overview](https://docs.google.com/presentation/d/1Z7Zyjx7IS3zy2r7UhZxtVHOWhCIgRvogYnqKTgGGamk/edit?usp=sharing) (16/09)

### About the Course

**Machine Learning for Scientific Discovery** is an 8-week intensive course designed to equip scientists with practical machine learning skills for research applications. The course is taught by Marc Lelarge and Tony Bonnaire at ENS, running from September to November 2025, followed by 6 weeks of supervised research projects.


### Learning Objectives

- **Apply ML Tools in Research Contexts**: Master machine learning methodologies for scientific applications
- **Read and Evaluate ML Literature**: Develop critical analysis skills for research papers, including theoretical foundations in mathematics and statistics  
- **Implement and Customize Solutions**: Gain hands-on experience with Python/PyTorch for modifying and extending ML implementations

### Course Structure

#### Core Curriculum (8 weeks)
| Week | Topic |
|------|-------|
| 1-2 | Statistics foundations and data representation |
| 3 | Linear models |
| 4 | Optimization techniques |
| 5 | Tree-based methods and ensemble techniques |
| 6-8 | Deep learning and modern neural network architectures |

#### Projects (6 weeks)
Department-supervised research projects applying course concepts to domain-specific problems.

### Why This Course Matters

The field of machine learning is experiencing unprecedented growth:

- **Scale of Research**: AAAI-26 received ~29,000 submissions from 75,000+ unique authors
- **Scientific Impact**: ML is revolutionizing discovery across disciplines:
  - Protein structure prediction (AlphaFold - Nobel Prize 2024)
  - Drug discovery and antibiotic development
  - Climate modeling and materials science
- **Accessibility Gap**: Bridge between powerful ML tools and domain scientists without extensive CS backgrounds

### Key Concepts Covered

#### Fundamental Principles
- **Inductive Inference**: Understanding how patterns learned from training data generalize to new observations
- **Common Task Framework**: Reproducible research through standardized datasets and evaluation metrics
- **Data Interpretation**: Critical analysis skills - "data does not speak for itself"

#### Technical Content
- Supervised and unsupervised learning algorithms
- Deep learning architectures and training techniques
- Clustering methods (K-means, hierarchical clustering)
- Model evaluation and validation strategies

#### Historical Context
The course traces AI development from philosophical foundations (Descartes, 1637) through modern breakthroughs:
- Early AI: Turing Test (1950), Perceptron (1958)
- Deep Learning Revolution: Backpropagation (1986) → ImageNet (2012) → Modern LLMs
- Current challenges: Scaling laws, computational limits, data scarcity

### Practical Information

#### Prerequisites
- Basic programming experience (Python recommended)
- Undergraduate mathematics (linear algebra, calculus, statistics)
- Scientific research background in any domain

#### Tools and Technologies
- **Primary Language**: Python
- **ML Framework**: PyTorch  
- **Additional Libraries**: NumPy, scikit-learn, matplotlib

### Data Representation and Unsupervised Learning

The first lecture concludes with hands-on exploration of key statistical and machine learning concepts. **Simpson's Paradox** is demonstrated using the classic UC Berkeley admissions dataset, showing how aggregated statistics can be misleading - while overall admission rates appeared to favor men (44% vs 35% for women), departmental analysis reveals women disproportionately applied to more competitive programs, illustrating the critical importance of proper data stratification and causal reasoning. 

**Clustering algorithms** are introduced through practical implementations, including **K-means clustering** with its cost function minimization (sum of squared distances to cluster centroids) and applications in image quantization for color reduction. **Hierarchical clustering** methods are covered with various linkage criteria (single, complete, average linkage), demonstrating how different distance metrics affect cluster formation. These techniques serve as concrete examples of unsupervised learning while reinforcing the theme that algorithmic tools require domain expertise for meaningful interpretation.

_Note: This project was build with the help of Claude_