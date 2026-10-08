# ==================================================
# REAL RESUME PIPELINE
# ==================================================

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import re

from section_dataset import section_data


# ==================================================
# 1. Resume File
# ==================================================

resume_file = "test_resume_2.pdf"


# ==================================================
# 2. Extract Resume Text
# ==================================================

reader = PdfReader(resume_file)

resume_text = ""

for page in reader.pages:

    text = page.extract_text()

    if text:

        resume_text += text + "\n"


# ==================================================
# 3. Canonical Resume Sections
# ==================================================

section_headings = [

    "Objective",

    "Education",

    "Skills",

    "Projects",

    "Work Experience",

    "Certifications",

    "Achievements",

    "Interests",

    "Languages"
]


# ==================================================
# 4. Display Dataset Information
# ==================================================

print(
    "\n===== RESUME SECTION DATASET =====\n"
)

print(
    f"Total examples: {len(section_data)}"
)


section_counts = {}


for text, label in section_data:

    section_counts[label] = (
        section_counts.get(label, 0) + 1
    )


print(
    "\nExamples per section:"
)


for section, count in section_counts.items():

    print(
        f"{section}: {count}"
    )


# ==================================================
# 5. Prepare Training Dataset
# ==================================================

training_texts = [

    text

    for text, label in section_data
]


training_labels = [

    label

    for text, label in section_data
]


# ==================================================
# 6. Load SentenceTransformer
# ==================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 7. Convert Training Dataset Into Embeddings
# ==================================================

training_embeddings = embedding_model.encode(

    training_texts,

    show_progress_bar=True
)


# ==================================================
# 8. Train Section Classifier
# ==================================================

classifier = make_pipeline(

    StandardScaler(),

    LogisticRegression(
        max_iter=1000
    )
)


classifier.fit(

    training_embeddings,

    training_labels
)


# ==================================================
# 9. Classify Heading Semantically
# ==================================================

def classify_heading(heading):

    heading_embedding = embedding_model.encode(
        [heading]
    )


    probabilities = classifier.predict_proba(
        heading_embedding
    )[0]


    classes = classifier.classes_


    sorted_indices = probabilities.argsort()[::-1]


    top_index = sorted_indices[0]

    second_index = sorted_indices[1]


    top_section = classes[top_index]

    top_score = probabilities[top_index]


    second_section = classes[second_index]

    second_score = probabilities[second_index]


    score_gap = (
        top_score - second_score
    )


    return (

        top_section,

        top_score,

        second_section,

        second_score,

        score_gap
    )


# ==================================================
# 10. Detect Whether a Line Looks Like a Heading
# ==================================================

def looks_like_heading(line):

    line = line.strip()


    if not line:

        return False


    # Very long lines are normally content

    if len(line) > 70:

        return False


    words = line.split()


    # Too many words usually means content

    if len(words) > 7:

        return False


    # Exact canonical heading

    if line in section_headings:

        return True


    # ALL CAPS heading

    if line.isupper():

        return True


    # Title Case heading

    title_case_words = 0


    for word in words:

        clean_word = re.sub(

            r"[^A-Za-z]",

            "",

            word
        )


        if (

            clean_word

            and

            clean_word[0].isupper()

        ):

            title_case_words += 1


    if (

        len(words) <= 5

        and

        title_case_words >= max(
            1,
            len(words) - 1
        )

    ):

        return True


    return False


# ==================================================
# 11. Smart Resume Section Segmentation
# ==================================================

sections = {}

current_section = None

lines = resume_text.splitlines()

i = 0


while i < len(lines):

    line = lines[i].strip()


    if not line:

        i += 1

        continue


    # --------------------------------------------------
    # Exact known heading
    # --------------------------------------------------

    if line in section_headings:

        current_section = line


        if current_section not in sections:

            sections[current_section] = ""


        i += 1

        continue


    # --------------------------------------------------
    # Possible unfamiliar heading
    # --------------------------------------------------

    if looks_like_heading(line):

        (

            predicted_section,

            top_score,

            second_section,

            second_score,

            score_gap

        ) = classify_heading(line)


        # Strong semantic heading

        if (

            top_score >= 0.30

            and

            score_gap >= 0.05

        ):

            current_section = predicted_section


            if current_section not in sections:

                sections[current_section] = ""


            i += 1

            continue


    # --------------------------------------------------
    # Normal content
    # --------------------------------------------------

    if current_section:

        sections[current_section] += (
            line + " "
        )


    i += 1


# ==================================================
# 12. Remove Empty Sections
# ==================================================

sections = {

    section: content.strip()

    for section, content in sections.items()

    if content.strip()
}


# ==================================================
# 13. Safety Check
# ==================================================

if not sections:

    print(
        "\n===== SECTION DETECTION ERROR =====\n"
    )


    print(
        "No resume sections could be detected."
    )


    print(
        "\nExtracted Resume Text:\n"
    )


    print(
        resume_text[:3000]
    )


    raise SystemExit


# ==================================================
# 14. Integrated Real Resume Pipeline
# ==================================================

print(
    "\n===== INTEGRATED REAL RESUME PIPELINE =====\n"
)


for section_name, content in sections.items():

    print(
        f"Section: {section_name}"
    )


    print(
        f"Content: {content}"
    )


    print(
        "-" * 60
    )


# ==================================================
# 15. Final Section Classification
# ==================================================

section_names = list(
    sections.keys()
)


section_texts = [

    sections[section]

    for section in section_names
]


section_embeddings = embedding_model.encode(

    section_texts,

    show_progress_bar=True
)


predictions = classifier.predict(

    section_embeddings
)


probabilities = classifier.predict_proba(

    section_embeddings
)


classes = classifier.classes_


# ==================================================
# 16. Confidence-Aware Classification
# ==================================================

print(
    "\n===== FINAL SECTION CLASSIFICATION =====\n"
)


for (

    section_name,

    prediction,

    probability

) in zip(

    section_names,

    predictions,

    probabilities

):


    sorted_indices = probability.argsort()[::-1]


    top_index = sorted_indices[0]

    second_index = sorted_indices[1]


    top_section = classes[top_index]

    top_score = probability[top_index]


    second_section = classes[second_index]

    second_score = probability[second_index]


    score_gap = (
        top_score - second_score
    )


    # --------------------------------------------------
    # Confidence Decision
    # --------------------------------------------------

    if (

        top_score >= 0.60

        and

        score_gap >= 0.10

    ):

        decision = "ACCEPT"


    elif (

        top_score >= 0.40

        and

        score_gap >= 0.10

    ):

        decision = "REVIEW"


    else:

        decision = "UNCERTAIN"


    print(
        f"Detected Heading: {section_name}"
    )


    print(
        f"Predicted Section: {prediction}"
    )


    print(
        f"Top Score: {top_score:.3f}"
    )


    print(
        f"Second Section: {second_section}"
    )


    print(
        f"Second Score: {second_score:.3f}"
    )


    print(
        f"Score Gap: {score_gap:.3f}"
    )


    print(
        f"Decision: {decision}"
    )


    print(
        "-" * 60
    )


# ==================================================
# 17. Improved Skill Extraction
# ==================================================

def extract_skills(skills_text):

    skills = []


    # --------------------------------------------------
    # Skill Vocabulary
    # --------------------------------------------------

    skill_patterns = [

        # Programming Languages

        "Python",

        "Java",

        "C++",

        "C",

        "SQL",

        "JavaScript",


        # Web Technologies

        "HTML",

        "HTML5",

        "CSS",


        # Data Science

        "NumPy",

        "Pandas",

        "Scikit-learn",

        "Data Analysis",

        "Data Preprocessing",

        "Data Visualization",

        "Statistics",

        "Data Science",


        # Machine Learning

        "Machine Learning",

        "Deep Learning",

        "Artificial Intelligence",

        "AI/ML",

        "AIML",

        "Regression",

        "Classification",

        "KNN",

        "CNN",


        # NLP

        "NLP",

        "Natural Language Processing",

        "Text Processing",

        "Tokenization",


        # Deep Learning Frameworks

        "TensorFlow",

        "Keras",

        "PyTorch",


        # Development Tools

        "Jupyter Notebook",

        "VS Code",

        "Streamlit",

        "Git",

        "GitHub",

        "Docker",


        # Cloud

        "AWS",

        "Azure",


        # APIs / Data Formats

        "REST API",

        "APIs",

        "JSON",


        # IoT / Embedded

        "IoT",

        "Embedded Systems",

        "Embedded System"
    ]


    # --------------------------------------------------
    # Normalize Whitespace
    # --------------------------------------------------

    text = re.sub(

        r"\s+",

        " ",

        skills_text

    ).strip()


    # --------------------------------------------------
    # Extract Skills
    # --------------------------------------------------

    for skill in skill_patterns:

        if re.search(

            r"(?<![A-Za-z0-9+#])"

            + re.escape(skill)

            + r"(?![A-Za-z0-9+#])",

            text,

            re.IGNORECASE

        ):

            skills.append(skill)


    # --------------------------------------------------
    # Remove Duplicates
    # --------------------------------------------------

    skills = list(

        dict.fromkeys(
            skills
        )

    )


    return skills


# ==================================================
# 18. Extract Skills From Real Resume
# ==================================================

if "Skills" in sections:

    extracted_skills = extract_skills(

        sections["Skills"]
    )


    print(
        "\n===== EXTRACTED SKILLS =====\n"
    )


    for skill in extracted_skills:

        print(skill)


    print(
        f"\nTotal Skills: "
        f"{len(extracted_skills)}"
    )


else:

    extracted_skills = []


    print(
        "\nSkills section not found."
    )


# ==================================================
# 19. Skill Normalization
# ==================================================

def normalize_skill(skill):

    skill = skill.strip().lower()


    normalization_map = {

        "aiml":
            "AI/ML",

        "ai ml":
            "AI/ML",

        "ai/ml":
            "AI/ML",

        "artificial intelligence":
            "Artificial Intelligence",

        "artificial intelligence and machine learning":
            "AI/ML",

        "machine learning":
            "Machine Learning",

        "deep learning":
            "Deep Learning",

        "data analysis":
            "Data Analysis",

        "data preprocessing":
            "Data Preprocessing",

        "data visualization":
            "Data Visualization",

        "data science":
            "Data Science",

        "python":
            "Python",

        "java":
            "Java",

        "c++":
            "C++",

        "c":
            "C",

        "sql":
            "SQL",

        "html":
            "HTML",

        "html5":
            "HTML",

        "css":
            "CSS",

        "javascript":
            "JavaScript",

        "numpy":
            "NumPy",

        "pandas":
            "Pandas",

        "scikit-learn":
            "Scikit-learn",

        "tensorflow":
            "TensorFlow",

        "keras":
            "Keras",

        "pytorch":
            "PyTorch",

        "nlp":
            "NLP",

        "natural language processing":
            "Natural Language Processing",

        "text processing":
            "Text Processing",

        "tokenization":
            "Tokenization",

        "cnn":
            "CNN",

        "knn":
            "KNN",

        "regression":
            "Regression",

        "classification":
            "Classification",

        "statistics":
            "Statistics",

        "jupyter notebook":
            "Jupyter Notebook",

        "vs code":
            "VS Code",

        "streamlit":
            "Streamlit",

        "git":
            "Git",

        "github":
            "GitHub",

        "docker":
            "Docker",

        "aws":
            "AWS",

        "azure":
            "Azure",

        "rest api":
            "REST API",

        "apis":
            "APIs",

        "json":
            "JSON",

        "iot":
            "IoT",

        "embedded system":
            "Embedded System",

        "embedded systems":
            "Embedded Systems"
    }


    if skill in normalization_map:

        return normalization_map[skill]


    return skill.title()


# ==================================================
# 20. Normalize Extracted Skills
# ==================================================

normalized_skills = [

    normalize_skill(skill)

    for skill in extracted_skills
]


# Remove duplicates while preserving order

normalized_skills = list(

    dict.fromkeys(

        normalized_skills

    )

)


print(
    "\n===== NORMALIZED SKILLS =====\n"
)


for skill in normalized_skills:

    print(skill)


print(
    f"\nTotal Normalized Skills: "
    f"{len(normalized_skills)}"
)


# ==================================================
# 21. Job Skill Matching
# ==================================================

required_skills = [

    "Python",

    "Data Analysis",

    "AI/ML",

    "Machine Learning",

    "SQL",

    "TensorFlow",

    "Git"
]


# ==================================================
# 22. Skill Equivalence
# ==================================================

skill_equivalents = {

    "AI/ML": [

        "AI/ML",

        "AIML",

        "Artificial Intelligence",

        "Machine Learning"
    ],


    "Software Engineering": [

        "Software Engineering",

        "Software Engineer"
    ],


    "HTML": [

        "HTML",

        "HTML5"
    ]
}


# ==================================================
# 23. Check Whether Two Skills Are Equivalent
# ==================================================

def skills_match(

    resume_skill,

    job_skill

):

    resume_skill = normalize_skill(

        resume_skill
    )


    job_skill = normalize_skill(

        job_skill
    )


    # --------------------------------------------------
    # Exact Match
    # --------------------------------------------------

    if resume_skill == job_skill:

        return True


    # --------------------------------------------------
    # Equivalence Groups
    # --------------------------------------------------

    for equivalent_skills in (

        skill_equivalents.values()

    ):

        normalized_group = [

            normalize_skill(skill)

            for skill in equivalent_skills
        ]


        if (

            resume_skill in normalized_group

            and

            job_skill in normalized_group

        ):

            return True


    return False


# ==================================================
# 24. Improved Job Skill Matching
# ==================================================

matched_skills = []

missing_skills = []


for job_skill in required_skills:

    found_match = False


    for resume_skill in normalized_skills:

        if skills_match(

            resume_skill,

            job_skill

        ):

            found_match = True

            break


    if found_match:

        matched_skills.append(

            job_skill

        )

    else:

        missing_skills.append(

            job_skill

        )


# ==================================================
# 25. Calculate Match Score
# ==================================================

if required_skills:

    match_percentage = (

        len(matched_skills)

        /

        len(required_skills)

    ) * 100

else:

    match_percentage = 0


# ==================================================
# 26. Display Job Matching Results
# ==================================================

print(
    "\n===== IMPROVED JOB SKILL MATCHING =====\n"
)


print(
    "Required Skills:"
)


for skill in required_skills:

    print(skill)


print(
    "\nMatched Skills:"
)


for skill in matched_skills:

    print(skill)


print(
    "\nMissing Skills:"
)


for skill in missing_skills:

    print(skill)


print(

    f"\nImproved Skill Match Score: "
    f"{match_percentage:.1f}%"

)