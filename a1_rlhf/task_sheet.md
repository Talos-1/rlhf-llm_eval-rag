# Assignment 1: Reinforcement Learning Human Feedback (RLHF) Implementation (Due: Tue, 17 Mar 2026 23:59)

## Implement a Reinforcement Learning with Human Feedback (RLHF) model and evaluate the model’s performance on open-source datasets 

Individual assignment
Weighting: 30%
Due date: 17 March 2026, Tue, Week 3.

## Assignment Overview
Welcome to your first assignment in Advanced Topics in Artificial Intelligence and Machine Learning! This assignment will give you hands-on experience in building and evaluating an RLHF Pipeline with open-source Large Language Models (LLMs).

You will select a range of open-source LLMs, fine-tune them with supervised instruction data, train a reward model using a range of open preference datasets, and then apply an RLHF algorithm (e.g., PPO - Proximal Policy Optimisation) to produce an aligned model. You will compare performance across a range of different base LLMs or a range of reward-dataset variants, evaluate alignment and utility, document experiments, and submit code, a short demo, and a report.

## Objectives
By the end of this assignment, you should be able to:
* Explain the RLHF pipeline: supervised fine-tuning, reward modelling and policy optimisation (e.g., PPO).
* Prepare and preprocess open preference datasets for reward modelling.
* Train a reward model from preference data and validate it.
* Evaluate and compare models on automatic metrics.

## Instructions
### Set up and baseline

1. Select a range of base open-source LLMs.
2. Run a supervised instruction-tuning step to obtain a baseline instruction-following model.
3. Conduct a baseline evaluation (qualitative samples and automated metrics).

### Reward modeling
1. Choose a range of open preference datasets (e.g., HH-RLHF, OpenAssistant preferences, HelpSteer2) or prepare your own dataset by web scraping.
2. Preprocess to [prompt, response_A, response_B, pref] format suitable for reward training (you need to perform this step if you choose to prepare your own dataset).
3. Train a (range of ) separate reward models using a (range of) open-source preference datasets. This model learns to predict human preferences based on the quality of generated responses.

### RLHF 
1. Use an RLHF framework (e.g., OpenRLHF, TRL) to fine-tune the language model using the trained reward model as a feedback signal. This typically involves algorithms like Proximal Policy Optimisation (PPO) or Direct Preference Optimisation (DPO).
2. Run a range of experiments that vary some major factors (e.g., different base LLMs, or reward dataset, etc.).
3. Keep track of rewards and sample outputs during training.

### Evaluation and analysis
1. Perform preference evaluation: have crowds (or at least TA/instructor + peers) do blind pairwise comparisons between model outputs (baseline vs RLHF variants). If human labels are not possible at scale, use a held-out preference dataset as proxy.
2. Report automatic metrics and human preference percentages.

### Submission Guidelines
* Submit your completed `assignment1_student id_surname.zip` file along with Jupyter notebook, video demonstration of PowerPoint slides and codes, and a PDF of the Overleaf report in a zip folder.
* Ensure that your code is well-commented and clearly organised. Include a brief description at the top of the file summarising what each section does.
* Plagiarism Warning: Ensure that all code and work is your own. If you use external resources, please cite them appropriately.
* Upload your submission before the due date.

### Advice and Resources
* You can read the following paper to learn more about how to implement RLHF:  Ouyang, L., Wu, J., Jiang, X. et al. (2022). Training language models to follow instructions with human feedback. arXiv:2203.02155. https://arxiv.org/abs/2203.02155
* The [following video](https://www.youtube.com/watch?v=K1UBOodkqEk) demonstrates how to implement RLHF.
* [Here](https://github.com/ash80/RLHF_in_notebooks) are the available codes to implement RLHF.
* [This article](https://huggingface.co/blog/rlhf) has provided some insights and further resources to implement RLHF.

### Additional Notes
* If you encounter any issues, please post questions on the course discussion board [insert discussion forum or contact information].
* You are strongly encouraged to experiment beyond the requirements of the assignment to further deepen your understanding.

### Academic Integrity and use of Generative AI
Students are permitted to use AI-assisted tools to support assignment code completion, provided that all outputs are critically reviewed and evaluated by the student. Human evaluation and reflection on AI-generated codes are required to ensure understanding and originality. However, for report writing, AI must not be used.

### Rubric

1. Structure, Content and Presentation (10%)
* Incorporates assignment scope, and objectives in detail throughout a thoughtful introduction.
* Presents findings clearly in a professional format, ideal for publication or academic dissemination. Visuals are high quality and well refined to deeply enhance understanding of concepts and ideas. 
* Presentation and delivery are exceptional with highly organised structure, comprehensive detail and strongly developed discussion for all ideas.

2. Literature Analysis (5%)
* Conducts a comprehensive literature analysis and provides deep reflections on a broad range of high-quality sources focusing on RLHF based systems and models and outlines clear relevance with detail. (2%)
* Descriptions have extremely strong links to relevant literature and a coherent, clear, logical structure.  (1%) 
* Articulates the positioning purposefully with connections to how prior work has informed, the intended goal and viable explanation of how it was achieved in the assignment. (3%)

3. Methodology & Experimental Depth (35%)
* Formulates highly relevant research questions with strong foundation of related methodologies and clear links to current/future applications. (1%) 
* Articulates method, approach and results comprehensively, with extensive alignment between demonstration, assignment scope and tasks given. (2%)
* Justifies all significant design choices (e.g., the type of model, dataset, LLM etc) in detail with high quality and highly credible evidence. (8%)
* Applies complex metrics extending on standard defaults and offers deep explanation. Collects and incorporates large scale and high-quality benchmarks with rigour and thoughtfully links to evidence. (9%)
* Experimental set ups are compared with several high-quality prior work and is woven into precise analysis with broad variety of contexts. (5%)

4. Implementation and Evaluation (50%)
* Articulates deep, structured error (limitation) analysis leading to actionable insights supported by highly credible research.
* Demonstrates significant, original and critical reflection on real-world deployment, including a range of technical, ethical, and societal implications.  Integrates a broad range of evidence from literature or industry to link and justify feasibility and impact.
* Provides strong justification with credible references for subjective decisions (e.g., model selection, hyperparameters, evaluation metrics selection etc.).
* Adds replication package, code, dataset with detailed comments and instructions for easy replication and share working repository.  Provides codes that is easy to deploy and test in different operating systems. The working repository is organised and shows signs of early and consistent effort.