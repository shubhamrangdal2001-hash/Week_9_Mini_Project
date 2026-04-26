# Evaluation Results — NCERT Class 9 Physics RAG

**Corpus:** Ch 8–12 (Motion, Force, Gravitation, Work/Energy, Sound)

**Retrieval:** BM25 + Sentence Transformer (TF-IDF), Hybrid RRF fusion

**Temperature:** 0 (deterministic)

---

| ID | Ch | Type | Correctness | Grounding | Top Section | Score |
|----|----|------|-------------|-----------|-------------|-------|
| Q01 | Ch8 | direct | wrong | grounded | 8.6 EQUATIONS OF MOTION FOR UN | 0.033 |
| Q02 | Ch8 | direct | wrong | grounded | 8.3 MEASURING THE RATE OF MOTI | 0.033 |
| Q03 | Ch8 | direct | correct | grounded | 8.3 MEASURING THE RATE OF MOTI | 0.033 |
| Q04 | Ch8 | paraphrased | correct | grounded | 8.6 EQUATIONS OF MOTION FOR UN | 0.033 |
| Q05 | Ch8 | paraphrased | partial | grounded | 8.4 RATE OF CHANGE OF VELOCITY | 0.033 |
| Q06 | Ch9 | direct | correct | grounded | 9.3 SECOND LAW OF MOTION | 0.033 |
| Q07 | Ch9 | direct | partial | grounded | 9.5 CONSERVATION OF MOMENTUM | 0.033 |
| Q08 | Ch9 | direct | wrong | grounded | 9.3 SECOND LAW OF MOTION | 0.033 |
| Q09 | Ch9 | paraphrased | partial | grounded | 9.3 SECOND LAW OF MOTION | 0.033 |
| Q10 | Ch9 | paraphrased | incorrect_refusal | na | 9.5 CONSERVATION OF MOMENTUM | 0.032 |
| Q11 | Ch10 | direct | wrong | grounded | 10.2 FREE FALL | 0.033 |
| Q12 | Ch10 | direct | correct | grounded | 10.2 FREE FALL | 0.032 |
| Q13 | Ch10 | direct | correct | grounded | 10.3 MASS AND WEIGHT | 0.033 |
| Q14 | Ch10 | paraphrased | wrong | grounded | 10.3 MASS AND WEIGHT | 0.033 |
| Q15 | Ch10 | direct | wrong | ungrounded | 10.5 BUOYANCY AND ARCHIMEDES'  | 0.033 |
| Q16 | Ch11 | direct | partial | grounded | 11.2 ENERGY | 0.033 |
| Q17 | Ch11 | direct | partial | grounded | 11.4 POWER | 0.033 |
| Q18 | Ch11 | direct | incorrect_refusal | na | 11.4 POWER | 0.033 |
| Q19 | Ch11 | paraphrased | incorrect_refusal | na | 11.2 ENERGY | 0.033 |
| Q20 | Ch12 | direct | correct | grounded | 12.4 REFLECTION OF SOUND — ECH | 0.033 |
| Q21 | Ch12 | direct | correct | grounded | 12.4 REFLECTION OF SOUND — ECH | 0.033 |
| Q22 | Ch12 | direct | wrong | grounded | 12.6 ULTRASOUND | 0.033 |
| Q23 | Ch12 | paraphrased | incorrect_refusal | na | 12.3 CHARACTERISTICS OF SOUND | 0.033 |
| Q24 | OOS | out_of_scope | correct_refusal | na | 8.3 MEASURING THE RATE OF MOTI | 0.032 |
| Q25 | OOS | out_of_scope | correct_refusal | na | 12.3 CHARACTERISTICS OF SOUND | 0.032 |
