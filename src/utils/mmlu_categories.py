"""
MMLU subject to category mapping.

Maps the 57 MMLU subjects (excluding 'miscellaneous') to 10 target categories
for analysis of model capabilities across different domains.
"""

from typing import Dict, List

# Mapping from target category to list of MMLU subjects
MMLU_CATEGORY_MAP: Dict[str, List[str]] = {
    "Mathematics": [
        "abstract_algebra",
        "elementary_mathematics",
        "high_school_mathematics",
        "college_mathematics",
        "high_school_statistics",
        "formal_logic",
    ],
    "Physical Sciences": [
        "astronomy",
        "conceptual_physics",
        "high_school_physics",
        "college_physics",
        "high_school_chemistry",
        "college_chemistry",
    ],
    "Biological Sciences": [
        "anatomy",
        "cell_biology",
        "college_biology",
        "high_school_biology",
        "virology",
    ],
    "Social and Behavioral Sciences": [
        "high_school_psychology",
        "professional_psychology",
        "sociology",
        "high_school_geography",
        "moral_scenarios",
        "moral_disputes",
    ],
    "Engineering Sciences": [
        "electrical_engineering",
    ],
    "Computer Science and AI": [
        "computer_security",
        "high_school_computer_science",
        "machine_learning",
    ],
    "Medicine and Health": [
        "clinical_knowledge",
        "medical_genetics",
        "professional_medicine",
        "college_medicine",
        "human_aging",
        "nutrition",
    ],
    "Business and Economics": [
        "business_ethics",
        "econometrics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "management",
        "marketing",
        "professional_accounting",
    ],
    "Humanities and Arts": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "world_religions",
        "philosophy",
        "logical_fallacies",
        "prehistory",
        "global_facts",
    ],
    "Law and Government": [
        "international_law",
        "jurisprudence",
        "professional_law",
        "us_foreign_policy",
        "security_studies",
        "public_relations",
    ],
}

# Create reverse mapping: subject -> category
_SUBJECT_TO_CATEGORY: Dict[str, str] = {}
for category, subjects in MMLU_CATEGORY_MAP.items():
    for subject in subjects:
        _SUBJECT_TO_CATEGORY[subject] = category

# List of all MMLU subjects (excluding miscellaneous)
MMLU_SUBJECTS: List[str] = list(_SUBJECT_TO_CATEGORY.keys())

# List of all categories
MMLU_CATEGORIES: List[str] = list(MMLU_CATEGORY_MAP.keys())


def get_category_for_subject(subject: str) -> str:
    """
    Get the target category for a given MMLU subject.
    
    Args:
        subject: The MMLU subject name (e.g., 'abstract_algebra')
        
    Returns:
        The category name (e.g., 'Mathematics')
        
    Raises:
        KeyError: If the subject is not in the mapping (e.g., 'miscellaneous')
    """
    if subject not in _SUBJECT_TO_CATEGORY:
        raise KeyError(
            f"Subject '{subject}' not found in category mapping. "
            f"Note: 'miscellaneous' is excluded from evaluation."
        )
    return _SUBJECT_TO_CATEGORY[subject]


def get_subjects_for_category(category: str) -> List[str]:
    """
    Get all MMLU subjects belonging to a given category.
    
    Args:
        category: The category name (e.g., 'Mathematics')
        
    Returns:
        List of subject names in that category
        
    Raises:
        KeyError: If the category is not valid
    """
    if category not in MMLU_CATEGORY_MAP:
        raise KeyError(
            f"Category '{category}' not found. "
            f"Valid categories: {MMLU_CATEGORIES}"
        )
    return MMLU_CATEGORY_MAP[category]


if __name__ == "__main__":
    # Print summary of categories and subjects
    print(f"Total categories: {len(MMLU_CATEGORIES)}")
    print(f"Total subjects: {len(MMLU_SUBJECTS)}")
    print()
    for category in MMLU_CATEGORIES:
        subjects = MMLU_CATEGORY_MAP[category]
        print(f"{category} ({len(subjects)} subjects):")
        for subject in subjects:
            print(f"  - {subject}")
        print()
