# Evaluation Results — NCERT Class 9 Physics RAG

**Corpus:** Ch 8–12 (Motion, Force, Gravitation, Work/Energy, Sound)

**Retrieval:** BM25 + Sentence Transformer (TF-IDF), Hybrid RRF fusion

**Temperature:** 0 (deterministic)

---

| ID | Ch | Type | Correctness | Grounding | Top Section | Score |
|----|----|------|-------------|-----------|-------------|-------|
| Q01 | Ch8 | direct | wrong | grounded | Introduction | 0.032 |
| Q02 | Ch8 | direct | partial | grounded | Introduction | 0.033 |
| Q03 | Ch8 | direct | correct | grounded | Introduction | 0.032 |
| Q04 | Ch8 | paraphrased | partial | grounded | Introduction | 0.032 |
| Q05 | Ch8 | paraphrased | wrong | grounded | Introduction | 0.033 |
| Q06 | Ch9 | direct | correct | grounded | Introduction | 0.033 |
| Q07 | Ch9 | direct | partial | grounded | Introduction | 0.033 |
| Q08 | Ch9 | direct | incorrect_refusal | na | Introduction | 0.033 |
| Q09 | Ch9 | paraphrased | partial | grounded | Introduction | 0.032 |
| Q10 | Ch9 | paraphrased | incorrect_refusal | na | Introduction | 0.033 |
| Q11 | Ch10 | direct | incorrect_refusal | na | Introduction | 0.033 |
| Q12 | Ch10 | direct | correct | grounded | Introduction | 0.032 |
| Q13 | Ch10 | direct | correct | grounded | Introduction | 0.032 |
| Q14 | Ch10 | paraphrased | wrong | grounded | Introduction | 0.032 |
| Q15 | Ch10 | direct | incorrect_refusal | na | Introduction | 0.030 |
| Q16 | Ch11 | direct | partial | grounded | Introduction | 0.032 |
| Q17 | Ch11 | direct | partial | partial | Introduction | 0.033 |
| Q18 | Ch11 | direct | incorrect_refusal | na | Introduction | 0.033 |
| Q19 | Ch11 | paraphrased | wrong | grounded | Introduction | 0.033 |
| Q20 | Ch12 | direct | correct | grounded | Introduction | 0.033 |
| Q21 | Ch12 | direct | correct | grounded | Introduction | 0.033 |
| Q22 | Ch12 | direct | wrong | grounded | Introduction | 0.032 |
| Q23 | Ch12 | paraphrased | partial | grounded | Introduction | 0.033 |
| Q24 | OOS | out_of_scope | correct_refusal | na | Introduction | 0.033 |
| Q25 | OOS | out_of_scope | missed_refusal | grounded | Introduction | 0.032 |
