"""
ethics_statement.py — Generate ethics and accessibility statement.

The proposal says the project should address ethics in terms of
accessibility and inclusivity. This script writes a structured
ethics statement to results/ethics_statement.txt and prints it.
"""

import os

OUTPUT_FILE = "results/ethics_statement.txt"
os.makedirs("results", exist_ok=True)

STATEMENT = """
ETHICS STATEMENT — ACCESSIBILITY AND INCLUSIVITY
=================================================

PROJECT CONTEXT
---------------
This project develops an AI-based sign language recognition system
using the WLASL (Word-Level American Sign Language) dataset. Sign
language recognition has direct relevance to accessibility and
inclusivity for the Deaf and Hard of Hearing community.

ACCESSIBILITY CONSIDERATIONS
-----------------------------
1. Sign language as a first language
   American Sign Language (ASL) is a complete, natural language used
   by an estimated 250,000–500,000 people in the US and Canada. Any
   automated recognition system must treat it with the same respect
   as any spoken language system — not as a simplified or secondary
   communication mode.

2. Limitations of the current system
   This system recognises a vocabulary of 10 isolated ASL words. It
   is not a translation system and cannot substitute for a human
   interpreter. Deploying it as such would be harmful and misleading.
   Any real-world application must clearly communicate these limits
   to users.

3. Dataset representation
   The WLASL dataset was collected from publicly available videos and
   may not represent the full diversity of signers — including
   variation by region, age, race, and individual signing style.
   Models trained on non-representative data risk performing worse
   for underrepresented groups, which would disproportionately
   affect the very community the system is meant to serve.

INCLUSIVITY CONSIDERATIONS
--------------------------
1. Community involvement
   Sign language technology should be developed with, not for, the
   Deaf community. The design decisions in this project (vocabulary
   selection, evaluation criteria) were driven by dataset availability
   rather than community input — a limitation that any production
   system should address.

2. Uncertainty as an accessibility feature
   This project's focus on uncertainty quantification is directly
   relevant to accessibility. A system that knows it is uncertain
   and communicates that to the user is more trustworthy than one
   that confidently produces wrong outputs. The entropy metrics and
   HMM-based uncertainty resolution implemented here are steps toward
   more reliable and honest AI-assisted communication tools.

3. Avoiding automation bias
   Users of any sign language recognition tool must be educated that
   AI predictions are probabilistic and can be wrong. Over-reliance
   on automated systems without human oversight creates risk,
   particularly in high-stakes contexts such as medical or legal
   interpretation.

RESPONSIBLE USE
---------------
This system is a research prototype intended for academic evaluation
of contextual modelling techniques. It is not suitable for deployment
as an accessibility aid without significant further development,
community consultation, and rigorous evaluation on diverse signers.

=================================================
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(STATEMENT)

print(STATEMENT)
print(f"Ethics statement saved → {OUTPUT_FILE}")