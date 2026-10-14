# LLM Alignment, Trustworthiness, and Retrieval Augmented Generation

| # | Project | Question | Headline result |
| --- | --- | --- | --- |
| 1 | RLHF for cyber-threat classification | Do preference-tuning methods (DPO, GRPO, SimPO) improve a small LLM beyond plain fine-tuning? | Fine-tuning increased head-to-head win rate from **12.7% to 31.8%** |
| 2 | Trustworthiness of LLMs on autism-related prompts | How do open-weight LLMs treat autistic users across demographics? | One model **leaked a Medicare number**; fairness scores ranged from **0.27 to 0.77** |
| 3 | Red-team advisor RAG | Can a 4B local model give grounded red-team advice from a curated corpus? | Answer relevancy **0.96**, faithfulness **0.69** across 10 test queries |

**Stack:**

- Language: Python
- AI Libraries: PyTorch, Hugging Face TRL, Ollama
- Model: BGE-M3
- Optimisation: Unsloth, LoRA
- Evaluation: RAGAS, Detoxify
- Database: ChromaDB

```bash
.
├── a1_rlhf/        # Fine-tuned Llama 3.2 3B, comparison between fine-tuning, DPO, GRPO, SimPO using LLM judges.
├── a2_llm_eval/    # 8-dimension trustworthiness evaluation of three open-weight LLMs.
└── a3_rag/         # Hybrid-retrieval RAG over MITRE ATT&CK, NIST, CISA, and PTES.
```

---

## References

[1] L. Ouyang et al., "Training language models to follow instructions with human feedback," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 35, 2022, pp. 27730–27744.

[2] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, "Direct preference optimization: Your language model is secretly a reward model," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023, pp. 53728–53741.

[3] Y. Meng, M. Xia, and D. Chen, "SimPO: Simple preference optimization with a reference-free reward," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 37, 2024, pp. 124198–124235, doi: 10.52202/079017-3946.

[4] Z. Shao et al., "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," 2024, arXiv:2402.03300.

[5] B. Wang et al., "DecodingTrust: A comprehensive assessment of trustworthiness in GPT models," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track, vol. 36, 2023, pp. 31232–31339.

[6] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 9459–9474.

[7] W. Fan et al., "A survey on RAG meeting LLMs: Towards retrieval-augmented large language models," in Proc. 30th ACM SIGKDD Conf. Knowl. Discovery Data Mining (KDD), Barcelona, Spain, 2024, pp. 6491–6501, doi: 10.1145/3637528.3671470.

[8] V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), 2020, pp. 6769–6781, doi: 10.18653/v1/2020.emnlp-main.550.

[9] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, "M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation," in Findings Assoc. Comput. Linguistics: ACL 2024, Bangkok, Thailand, 2024, pp. 2318–2335, doi: 10.18653/v1/2024.findings-acl.137.

[10] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in Proc. 32nd Int. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, Boston, MA, USA, 2009, pp. 758–759, doi: 10.1145/1571941.1572114.

[11] M. Glass, G. Rossiello, M. F. M. Chowdhury, A. R. Naik, P. Cai, and A. Gliozzo, "Re2G: Retrieve, rerank, generate," in Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol. (NAACL-HLT), Seattle, WA, USA, 2022, pp. 2701–2715, doi: 10.18653/v1/2022.naacl-main.194.

[12] S. Es, J. James, L. Espinosa Anke, and S. Schockaert, "RAGAs: Automated evaluation of retrieval augmented generation," in Proc. 18th Conf. Eur. Chapter Assoc. Comput. Linguistics: Syst. Demonstrations, St. Julian's, Malta, 2024, pp. 150–158.

[13] C. Zhao et al., "CyberBOT: Ontology-grounded retrieval augmented generation for reliable cybersecurity education," in Proc. 34th ACM Int. Conf. Inf. Knowl. Manage. (CIKM), Seoul, Republic of Korea, 2025, pp. 6752–6756, doi: 10.1145/3746252.3761478.

[14] F. Blefari, C. Cosentino, F. A. Pironti, A. Furfaro, and F. Marozzo, "CyberRAG: An agentic RAG cyber attack classification and reporting tool," Future Gener. Comput. Syst., vol. 176, 2026, Art. no. 108186, doi: 10.1016/j.future.2025.108186.

[15] A. Yang et al., "Qwen3 technical report," 2025, arXiv:2505.09388.
