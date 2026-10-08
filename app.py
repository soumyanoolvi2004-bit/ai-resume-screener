import streamlit as st
import re
import os

from pypdf import PdfReader
from pypdf.errors import EmptyFileError

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from section_dataset import section_data


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# SECTION DISPLAY NAMES
# ============================================================

SECTION_DISPLAY_NAMES = {
    "summary": "Summary",
    "objective": "Objective",
    "education": "Education",
    "skills": "Skills",
    "projects": "Projects",
    "experience": "Work Experience",
    "certifications": "Certifications",
    "achievements": "Achievements",
    "interests": "Interests",
    "languages": "Languages"
}


# ============================================================
# SECTION KEYWORDS
# ============================================================

SECTION_KEYWORDS = {

    "summary": [
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "career summary",
        "career profile",
        "about me"
    ],

    "objective": [
        "objective",
        "career objective",
        "professional objective"
    ],

    "skills": [
        "skills",
        "technical skills",
        "technical expertise",
        "core competencies",
        "programming skills",
        "technical competencies",
        "technical knowledge",
        "key skills"
    ],

    "projects": [
        "projects",
        "project",
        "academic projects",
        "personal projects",
        "key projects",
        "project experience"
    ],

    "education": [
        "education",
        "educational background",
        "academic background",
        "academic qualifications",
        "qualifications",
        "education qualifications"
    ],

    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "career experience",
        "prior experience",
        "previous experience",
        "internship experience",
        "internships",
        "work"
    ],

    "certifications": [
        "certifications",
        "certificates",
        "certification",
        "courses",
        "certifications & courses",
        "certificates & courses",
        "courses & certifications"
    ],

    "achievements": [
        "achievements",
        "awards",
        "honors",
        "accomplishments"
    ],

    "interests": [
        "interests",
        "hobbies",
        "areas of interest"
    ],

    "languages": [
        "languages",
        "language",
        "language proficiency",
        "languages known"
    ]
}


# ============================================================
# SECTION LABEL NORMALIZATION
# ============================================================

def normalize_section_label(label):

    if not label:
        return None

    label = str(label).strip().lower()

    mapping = {

        "summary": "summary",
        "profile": "summary",

        "objective": "objective",

        "education": "education",
        "academic": "education",

        "skills": "skills",

        "projects": "projects",
        "project": "projects",

        "experience": "experience",
        "work experience": "experience",
        "professional experience": "experience",
        "internships": "experience",
        "internship": "experience",

        "certifications": "certifications",
        "certificates": "certifications",
        "certification": "certifications",
        "courses": "certifications",

        "achievements": "achievements",
        "awards": "achievements",

        "interests": "interests",

        "languages": "languages",
        "language": "languages"
    }

    return mapping.get(label)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text(uploaded_file):

    try:

        uploaded_file.seek(0)

        reader = PdfReader(uploaded_file)

        if len(reader.pages) == 0:
            return ""

        full_text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                full_text += page_text + "\n"

        return full_text.strip()

    except EmptyFileError:
        return ""

    except Exception:
        return ""


# ============================================================
# PYMUPDF FALLBACK
# ============================================================

def extract_text_with_pymupdf(uploaded_file):

    try:

        import pymupdf

        uploaded_file.seek(0)

        pdf_bytes = uploaded_file.read()

        if not pdf_bytes:
            return ""

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        full_text = ""

        for page in document:

            page_text = page.get_text("text")

            if page_text:
                full_text += page_text + "\n"

        document.close()

        return full_text.strip()

    except Exception:
        return ""


# ============================================================
# OCR FALLBACK
# ============================================================

def extract_text_with_ocr(uploaded_file):

    try:

        import pymupdf
        import pytesseract

        from PIL import Image

        possible_tesseract_paths = [

            r"C:\Program Files\Tesseract-OCR\tesseract.exe",

            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]

        for path in possible_tesseract_paths:

            if os.path.exists(path):

                pytesseract.pytesseract.tesseract_cmd = path

                break

        uploaded_file.seek(0)

        pdf_bytes = uploaded_file.read()

        if not pdf_bytes:
            return ""

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        full_text = ""

        for page in document:

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False
            )

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            page_text = pytesseract.image_to_string(
                image,
                config="--psm 6"
            )

            if page_text.strip():

                full_text += (
                    page_text +
                    "\n"
                )

        document.close()

        return full_text.strip()

    except Exception:
        return ""


# ============================================================
# ROBUST PDF EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):

    # --------------------------------------------------------
    # 1. PyPDF
    # --------------------------------------------------------

    text = extract_text(uploaded_file)

    if len(text.strip()) >= 50:

        return text, "PyPDF"


    # --------------------------------------------------------
    # 2. PyMuPDF
    # --------------------------------------------------------

    text = extract_text_with_pymupdf(
        uploaded_file
    )

    if len(text.strip()) >= 50:

        return text, "PyMuPDF"


    # --------------------------------------------------------
    # 3. OCR
    # --------------------------------------------------------

    text = extract_text_with_ocr(
        uploaded_file
    )

    if len(text.strip()) >= 20:

        return text, "OCR"


    return "", "Failed"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NORMALIZE HEADING
# ============================================================

def normalize_heading(line):

    line = line.strip()

    line = re.sub(
        r"\s+",
        " ",
        line
    )

    line = line.lower()

    line = line.replace("–", "-")

    line = line.replace("—", "-")

    line = line.strip(" :|-")

    return line


# ============================================================
# DETECT EXACT KNOWN SECTION
# ============================================================

def detect_known_section(line):

    normalized = normalize_heading(line)

    for section, keywords in SECTION_KEYWORDS.items():

        for keyword in keywords:

            if normalized == keyword:

                return section

    return None


# ============================================================
# FIND SECTION KEYWORD MATCHES
# ============================================================

def get_section_keyword_matches(line):

    normalized = normalize_heading(line)

    matches = []

    for section, keywords in SECTION_KEYWORDS.items():

        for keyword in keywords:

            if keyword in normalized:

                matches.append(section)

                break

    return list(
        dict.fromkeys(matches)
    )


# ============================================================
# MIXED HEADING
# ============================================================

def is_mixed_heading(line):

    matches = get_section_keyword_matches(line)

    return len(matches) >= 2


# ============================================================
# CONTENT PROTECTION
# ============================================================

def is_likely_content_line(line):

    text = line.strip()

    if not text:
        return True

    normalized = normalize_heading(text)


    # Known headings
    if detect_known_section(text):

        return False


    # Email
    email_pattern = (
        r"[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}"
    )

    if re.search(
        email_pattern,
        text
    ):

        return True


    # URL
    if re.search(
        r"(linkedin\.com|github\.com|"
        r"https?://|www\.)",
        text,
        re.IGNORECASE
    ):

        return True


    # Phone
    compact = re.sub(
        r"[\s()-]",
        "",
        text
    )

    if re.fullmatch(
        r"(?:\+91)?[6-9]\d{9}",
        compact
    ):

        return True


    # Bullets
    if re.match(
        r"^[•●▪◦\-*]\s*",
        text
    ):

        return True


    # Content labels
    content_labels = {

        "libraries",
        "library",
        "techniques",
        "domains",
        "classification",
        "processing",
        "tools",
        "programming",
        "programming & tools",
        "technologies",
        "technology",
        "software",
        "frameworks",
        "platforms",
        "colab, vs code, streamlit",
        "processing, iot"
    }

    if normalized in content_labels:

        return True


    # Organizations / project content
    content_patterns = [

        r".*\bbleep education\b.*",
        r".*\bvisionastraa\b.*",
        r".*\buniversity\b.*",
        r".*\bcollege\b.*",
        r".*\binstitute\b.*",
        r".*\bcompletion certificate\b.*",
        r".*\binternship program\b.*",
        r".*\binternship\b.*",
        r".*\bcertificate\b.*",
        r".*\bclassification\b.*",
        r".*\bprocessing\b.*",
        r".*\blibraries\b.*",
        r".*\btechniques\b.*"
    ]

    for pattern in content_patterns:

        if re.fullmatch(
            pattern,
            normalized,
            re.IGNORECASE
        ):

            return True


    # Technical terms
    technical_terms = [

        "numpy",
        "pandas",
        "tensorflow",
        "keras",
        "scikit-learn",
        "sentence transformer",
        "sentence transformers",
        "supervised learning",
        "text processing",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "embedded systems",
        "ev charging",
        "streamlit",
        "python",
        "sql",
        "github",
        "colab",
        "vs code"
    ]

    for term in technical_terms:

        if term in normalized:

            return True

    return False


# ============================================================
# HEADING DETECTION
# ============================================================

def looks_like_heading(line):

    text = line.strip()

    if not text:
        return False

    if detect_known_section(text):

        return True

    if is_mixed_heading(text):

        return True

    if is_likely_content_line(text):

        return False

    if len(text) > 60:

        return False

    words = text.split()

    if len(words) > 5:

        return False


    # ALL CAPS
    if text.isupper() and len(words) <= 4:

        return True


    # Title Case
    title_case_count = 0

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

            title_case_count += 1

    if (
        2 <= len(words) <= 3
        and
        title_case_count == len(words)
    ):

        return True

    return False


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# TRAIN SECTION CLASSIFIER
# ============================================================

@st.cache_resource
def train_section_classifier():

    texts = [
        text
        for text, label in section_data
    ]

    labels = [
        label
        for text, label in section_data
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )

    model.fit(
        embeddings,
        labels
    )

    return model


classifier = train_section_classifier()


# ============================================================
# NLP CLASSIFICATION
# ============================================================

def classify_text(text):

    embedding = embedding_model.encode(
        [text]
    )

    probabilities = (
        classifier.predict_proba(
            embedding
        )[0]
    )

    classes = classifier.classes_

    sorted_indices = (
        probabilities.argsort()[::-1]
    )

    top_index = sorted_indices[0]

    second_index = (
        sorted_indices[1]
        if len(sorted_indices) > 1
        else sorted_indices[0]
    )

    top_section = classes[top_index]

    top_probability = probabilities[top_index]

    second_section = classes[second_index]

    second_probability = probabilities[second_index]

    probability_gap = (
        top_probability
        -
        second_probability
    )

    return (
        top_section,
        top_probability,
        second_section,
        second_probability,
        probability_gap
    )


# ============================================================
# SECTION SEGMENTATION
# ============================================================

def detect_sections(
    resume_text,
    allow_ml_headings=True
):

    sections = {}

    current_section = None

    lines = resume_text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue


        # ----------------------------------------------------
        # KNOWN SECTION
        # ----------------------------------------------------

        known_section = detect_known_section(line)

        if known_section:

            current_section = known_section

            if current_section not in sections:

                sections[current_section] = ""

            continue


        # ----------------------------------------------------
        # UNKNOWN HEADING
        # ----------------------------------------------------

        if allow_ml_headings:

            if looks_like_heading(line):

                if is_mixed_heading(line):

                    continue

                (
                    predicted_section,
                    top_probability,
                    second_section,
                    second_probability,
                    probability_gap
                ) = classify_text(line)

                mapped_section = normalize_section_label(
                    predicted_section
                )


                # Conservative ML acceptance
                if (
                    mapped_section
                    and
                    top_probability >= 0.70
                    and
                    probability_gap >= 0.20
                ):

                    current_section = mapped_section

                    if current_section not in sections:

                        sections[current_section] = ""

                    continue


        # ----------------------------------------------------
        # NORMAL CONTENT
        # ----------------------------------------------------

        if current_section:

            sections[current_section] += (
                line + " "
            )


    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    sections = {

        section: content.strip()

        for section, content
        in sections.items()

        if content.strip()
    }

    return sections


# ============================================================
# SKILL DATABASE
# ============================================================

SKILL_DATABASE = [

    "python",
    "sql",
    "mysql",
    "nosql",
    "html",
    "css",
    "javascript",

    "tensorflow",
    "keras",
    "pytorch",

    "machine learning",
    "deep learning",
    "artificial intelligence",

    "data science",
    "data analysis",
    "data analytics",
    "data preprocessing",
    "data visualization",

    "statistics",

    "scikit-learn",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",

    "regression",
    "classification",
    "knn",
    "cnn",

    "nlp",
    "text processing",
    "tokenization",

    "computer vision",
    "opencv",

    "hugging face",
    "transformers",
    "large language models",
    "llm",

    "retrieval-augmented generation",
    "rag",
    "generative ai",

    "autoencoders",
    "variational autoencoders",
    "vae",
    "diffusion models",

    "jupyter notebook",
    "vs code",
    "streamlit",

    "json",
    "apis",
    "rest api",

    "git",
    "github",

    "tkinter",
    "geopy",

    "iot",
    "embedded systems"
]


# ============================================================
# SKILL DISPLAY NAMES
# ============================================================

SKILL_DISPLAY_NAMES = {

    "python": "Python",
    "sql": "SQL",
    "mysql": "MySQL",
    "nosql": "NoSQL",

    "html": "HTML",
    "css": "CSS",
    "javascript": "JavaScript",

    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "pytorch": "PyTorch",

    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "artificial intelligence": "Artificial Intelligence",

    "data science": "Data Science",
    "data analysis": "Data Analysis",
    "data analytics": "Data Analytics",
    "data preprocessing": "Data Preprocessing",
    "data visualization": "Data Visualization",

    "statistics": "Statistics",

    "scikit-learn": "Scikit-learn",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",

    "regression": "Regression",
    "classification": "Classification",
    "knn": "KNN",
    "cnn": "CNN",

    "nlp": "NLP",
    "text processing": "Text Processing",
    "tokenization": "Tokenization",

    "computer vision": "Computer Vision",
    "opencv": "OpenCV",

    "hugging face": "Hugging Face",
    "transformers": "Transformers",

    "large language models": "Large Language Models",
    "llm": "LLM",

    "retrieval-augmented generation":
        "Retrieval-Augmented Generation",

    "rag": "RAG",
    "generative ai": "Generative AI",

    "autoencoders": "Autoencoders",

    "variational autoencoders":
        "Variational Autoencoders",

    "vae": "VAE",

    "diffusion models": "Diffusion Models",

    "jupyter notebook": "Jupyter Notebook",
    "vs code": "VS Code",
    "streamlit": "Streamlit",

    "json": "JSON",
    "apis": "APIs",
    "rest api": "REST API",

    "git": "Git",
    "github": "GitHub",

    "tkinter": "Tkinter",
    "geopy": "Geopy",

    "iot": "IoT",
    "embedded systems": "Embedded Systems"
}


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    text = re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()

    found = []

    for skill in SKILL_DATABASE:

        pattern = (
            r"(?<![a-zA-Z0-9])"
            +
            re.escape(skill)
            +
            r"(?![a-zA-Z0-9])"
        )

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):

            found.append(skill)

    return list(
        dict.fromkeys(
            SKILL_DISPLAY_NAMES.get(
                skill,
                skill.title()
            )
            for skill in found
        )
    )


# ============================================================
# SKILL NORMALIZATION
# ============================================================

def normalize_skill(skill):

    skill = skill.strip().lower()

    mapping = {

        "aiml": "AI/ML",
        "ai ml": "AI/ML",
        "ai/ml": "AI/ML",

        "artificial intelligence":
            "Artificial Intelligence",

        "machine learning":
            "Machine Learning",

        "data analysis":
            "Data Analysis",

        "data analytics":
            "Data Analysis",

        "python": "Python",
        "sql": "SQL",
        "mysql": "MySQL",

        "tensorflow": "TensorFlow",
        "keras": "Keras",
        "pytorch": "PyTorch",

        "github": "GitHub",
        "git": "Git",

        "nlp": "NLP",
        "iot": "IoT"
    }

    return mapping.get(
        skill,
        skill.title()
    )


# ============================================================
# SKILL EQUIVALENCE
# ============================================================

SKILL_EQUIVALENTS = {

    "AI/ML": [
        "AI/ML",
        "Artificial Intelligence",
        "Machine Learning"
    ],

    "Data Analysis": [
        "Data Analysis",
        "Data Analytics"
    ]
}


# ============================================================
# SKILL MATCHING
# ============================================================

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

    if resume_skill == job_skill:

        return True

    for group in SKILL_EQUIVALENTS.values():

        normalized_group = [
            normalize_skill(skill)
            for skill in group
        ]

        if (
            resume_skill in normalized_group
            and
            job_skill in normalized_group
        ):

            return True

    return False


# ============================================================
# DEFAULT JOB SKILLS
# ============================================================

DEFAULT_REQUIRED_SKILLS = [

    "Python",
    "Data Analysis",
    "AI/ML",
    "Machine Learning",
    "SQL",
    "TensorFlow",
    "Git"
]


# ============================================================
# MATCH JOB SKILLS
# ============================================================

def match_job_skills(
    resume_skills,
    required_skills
):

    matched_skills = []

    missing_skills = []

    for job_skill in required_skills:

        found = False

        for resume_skill in resume_skills:

            if skills_match(
                resume_skill,
                job_skill
            ):

                found = True

                break

        if found:

            matched_skills.append(
                job_skill
            )

        else:

            missing_skills.append(
                job_skill
            )


    if required_skills:

        match_percentage = (
            len(matched_skills)
            /
            len(required_skills)
        ) * 100

    else:

        match_percentage = 0


    return (
        matched_skills,
        missing_skills,
        match_percentage
    )


# ============================================================
# CANDIDATE SCORE
# ============================================================

def calculate_candidate_score(
    sections,
    resume_skills,
    required_skills
):

    (
        matched_skills,
        missing_skills,
        skill_match_percentage
    ) = match_job_skills(
        resume_skills,
        required_skills
    )


    # --------------------------------------------------------
    # SKILLS - 40
    # --------------------------------------------------------

    skill_score = (
        skill_match_percentage * 0.40
    )


    # --------------------------------------------------------
    # PROJECTS - 20
    # --------------------------------------------------------

    project_score = 0

    project_text = sections.get(
        "projects",
        ""
    )

    if project_text:

        project_words = len(
            project_text.split()
        )

        project_skill_hits = 0

        project_skills = extract_skills(
            project_text
        )

        for job_skill in required_skills:

            for project_skill in project_skills:

                if skills_match(
                    project_skill,
                    job_skill
                ):

                    project_skill_hits += 1

                    break


        if project_words >= 50:

            project_base = 12

        elif project_words >= 25:

            project_base = 9

        elif project_words >= 10:

            project_base = 6

        else:

            project_base = 3


        relevance_bonus = min(
            8,
            project_skill_hits * 2
        )

        project_score = min(
            20,
            project_base + relevance_bonus
        )


    # --------------------------------------------------------
    # EXPERIENCE - 15
    # --------------------------------------------------------

    experience_score = 0

    experience_text = sections.get(
        "experience",
        ""
    )

    if experience_text:

        experience_words = len(
            experience_text.split()
        )

        if experience_words >= 50:

            experience_score = 15

        elif experience_words >= 25:

            experience_score = 12

        elif experience_words >= 10:

            experience_score = 8

        else:

            experience_score = 5


    # --------------------------------------------------------
    # EDUCATION - 10
    # --------------------------------------------------------

    education_score = (
        10
        if sections.get("education")
        else 0
    )


    # --------------------------------------------------------
    # CERTIFICATIONS - 5
    # --------------------------------------------------------

    certification_score = (
        5
        if sections.get("certifications")
        else 0
    )


    # --------------------------------------------------------
    # SECTION QUALITY - 5
    # --------------------------------------------------------

    important_sections = [

        "education",
        "skills",
        "projects",
        "experience",
        "certifications"
    ]

    detected_sections = sum(

        1

        for section in important_sections

        if sections.get(section)
    )

    section_score = (
        detected_sections
        /
        len(important_sections)
    ) * 5


    # --------------------------------------------------------
    # COMPLETENESS - 5
    # --------------------------------------------------------

    completeness_sections = [

        "summary",
        "education",
        "skills",
        "projects",
        "experience",
        "certifications",
        "achievements",
        "languages"
    ]

    detected_completeness = sum(

        1

        for section in completeness_sections

        if sections.get(section)
    )

    completeness_score = (
        detected_completeness
        /
        len(completeness_sections)
    ) * 5


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    total_score = (

        skill_score
        +
        project_score
        +
        experience_score
        +
        education_score
        +
        certification_score
        +
        section_score
        +
        completeness_score
    )


    return {

        "total_score":
            round(total_score, 1),

        "skill_score":
            round(skill_score, 1),

        "project_score":
            round(project_score, 1),

        "experience_score":
            round(experience_score, 1),

        "education_score":
            round(education_score, 1),

        "certification_score":
            round(certification_score, 1),

        "section_score":
            round(section_score, 1),

        "completeness_score":
            round(completeness_score, 1),

        "matched_skills":
            matched_skills,

        "missing_skills":
            missing_skills,

        "skill_match_percentage":
            round(
                skill_match_percentage,
                1
            )
    }


# ============================================================
# CANDIDATE NAME EXTRACTION
# ============================================================

def extract_candidate_name(text):

    lines = [

        line.strip()

        for line in text.splitlines()

        if line.strip()
    ]


    # --------------------------------------------------------
    # EMAIL
    # --------------------------------------------------------

    email_pattern = (
        r"[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}"
    )

    email_match = re.search(
        email_pattern,
        text
    )


    # --------------------------------------------------------
    # FIRST NAME-LIKE LINE
    # --------------------------------------------------------

    for line in lines[:15]:

        if (

            len(line) <= 50

            and

            not re.search(
                r"@|linkedin|github|http|www\.",
                line,
                re.IGNORECASE
            )

            and

            not detect_known_section(line)
        ):

            words = line.split()

            if 2 <= len(words) <= 4:

                valid_words = all(

                    re.fullmatch(
                        r"[A-Za-z][A-Za-z.'-]*",
                        word
                    )

                    for word in words
                )

                if valid_words:

                    if not any(

                        keyword in line.lower()

                        for keyword in [

                            "education",
                            "skills",
                            "experience",
                            "project",
                            "certificate",
                            "internship",
                            "engineer",
                            "developer"
                        ]
                    ):

                        return line


    # --------------------------------------------------------
    # EMAIL USERNAME FALLBACK
    # --------------------------------------------------------

    if email_match:

        username = email_match.group(0).split("@")[0]

        username = re.sub(
            r"[._-]+",
            " ",
            username
        )

        if username:

            return username.title()


    return "Unknown Candidate"


# ============================================================
# RESUME VALIDATION
# ============================================================

def validate_resume(
    text,
    sections
):

    text_lower = text.lower()

    score = 0


    # --------------------------------------------------------
    # SECTION SIGNAL
    # --------------------------------------------------------

    important_sections = [

        "education",
        "skills",
        "projects",
        "experience",
        "certifications",
        "summary",
        "objective"
    ]

    detected_sections = sum(

        1

        for section in important_sections

        if sections.get(section)
    )

    if detected_sections >= 2:

        score += 3

    elif detected_sections == 1:

        score += 2


    # --------------------------------------------------------
    # EMAIL
    # --------------------------------------------------------

    email_pattern = (
        r"[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}"
    )

    if re.search(
        email_pattern,
        text
    ):

        score += 2


    # --------------------------------------------------------
    # PHONE
    # --------------------------------------------------------

    phone_pattern = (
        r"(?:\+91[\s-]?)?"
        r"[6-9]\d{9}"
    )

    if re.search(
        phone_pattern,
        text
    ):

        score += 2


    # --------------------------------------------------------
    # LINKEDIN
    # --------------------------------------------------------

    if "linkedin" in text_lower:

        score += 1


    # --------------------------------------------------------
    # GITHUB
    # --------------------------------------------------------

    if "github" in text_lower:

        score += 1


    # --------------------------------------------------------
    # RESUME-SPECIFIC TERMS
    # --------------------------------------------------------

    resume_terms = [

        "education",
        "skills",
        "projects",
        "experience",
        "internship",
        "certification",
        "objective",
        "summary",
        "bachelor",
        "degree",
        "engineering",
        "university",
        "college",
        "cgpa",
        "technical skills"
    ]

    resume_term_hits = sum(

        1

        for term in resume_terms

        if term in text_lower
    )


    if resume_term_hits >= 5:

        score += 3

    elif resume_term_hits >= 3:

        score += 2

    elif resume_term_hits >= 1:

        score += 1


    # --------------------------------------------------------
    # TECHNICAL SKILLS
    # --------------------------------------------------------

    technical_hits = 0

    for skill in SKILL_DATABASE:

        pattern = (

            r"(?<![a-zA-Z0-9])"
            +
            re.escape(skill)
            +
            r"(?![a-zA-Z0-9])"
        )

        if re.search(
            pattern,
            text_lower
        ):

            technical_hits += 1


    if technical_hits >= 5:

        score += 3

    elif technical_hits >= 3:

        score += 2

    elif technical_hits >= 1:

        score += 1


    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    return score >= 5


# ============================================================
# RESUME COMPLETENESS SCORE
# ============================================================

def calculate_resume_score(
    text,
    sections
):

    score = 0


    # Email
    email_pattern = (
        r"[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}"
    )

    if re.search(
        email_pattern,
        text
    ):

        score += 10


    # Phone
    phone_pattern = (
        r"(?:\+91[\s-]?)?"
        r"[6-9]\d{9}"
    )

    if re.search(
        phone_pattern,
        text
    ):

        score += 10


    if sections.get("education"):

        score += 10


    if sections.get("skills"):

        score += 10


    if sections.get("projects"):

        score += 10


    if sections.get("experience"):

        score += 5


    if sections.get("certifications"):

        score += 5


    if "linkedin" in text.lower():

        score += 2


    if "github" in text.lower():

        score += 2


    return score


# ============================================================
# PROCESS ONE RESUME
# ============================================================

def process_resume(
    uploaded_file,
    required_skills
):

    resume_text, extraction_method = (
        extract_resume_text(
            uploaded_file
        )
    )


    if not resume_text:

        return {

            "valid": False,

            "filename":
                uploaded_file.name,

            "name":
                "Unknown Candidate",

            "error":
                "Could not extract readable text."
        }


    cleaned_text = clean_text(
        resume_text
    )


    sections = detect_sections(
        resume_text,
        allow_ml_headings=True
    )


    resume_valid = validate_resume(
        cleaned_text,
        sections
    )


    resume_score = calculate_resume_score(
        cleaned_text,
        sections
    )


    resume_skills = extract_skills(
        resume_text
    )


    candidate_score = calculate_candidate_score(
        sections,
        resume_skills,
        required_skills
    )


    candidate_name = extract_candidate_name(
        resume_text
    )


    return {

        "valid":
            resume_valid,

        "filename":
            uploaded_file.name,

        "name":
            candidate_name,

        "text":
            resume_text,

        "sections":
            sections,

        "skills":
            resume_skills,

        "resume_score":
            resume_score,

        "candidate_score":
            candidate_score,

        "extraction_method":
            extraction_method
    }


# ============================================================
# USER DASHBOARD
# ============================================================

def user_dashboard():

    st.title(
        "👤 User Dashboard"
    )

    st.write(
        "Upload your resume to analyze "
        "skills, sections and job compatibility."
    )


    uploaded_file = st.file_uploader(

        "Upload Your Resume",

        type=["pdf"],

        accept_multiple_files=False,

        key="user_resume"
    )


    if uploaded_file is None:

        st.info(
            "Upload one PDF resume to begin."
        )

        return


    with st.spinner(
        "Analyzing your resume..."
    ):

        result = process_resume(
            uploaded_file,
            DEFAULT_REQUIRED_SKILLS
        )


    if not result["valid"]:

        st.error(
            "This document could not be confidently "
            "identified as a resume."
        )

        if result.get("error"):

            st.warning(
                result["error"]
            )

        return


    st.success(
        f"Resume analyzed using "
        f"{result['extraction_method']}."
    )


    # ========================================================
    # TOP METRICS
    # ========================================================

    candidate_score = result[
        "candidate_score"
    ]


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Candidate Score",
            f"{candidate_score['total_score']}/100"
        )


    with col2:

        st.metric(
            "Resume Score",
            f"{result['resume_score']}/64"
        )


    with col3:

        st.metric(
            "Skill Match",
            f"{candidate_score['skill_match_percentage']:.1f}%"
        )


    with col4:

        st.metric(
            "Detected Sections",
            len(result["sections"])
        )


    # ========================================================
    # SCORE BREAKDOWN
    # ========================================================

    st.header(
        "🎯 Candidate Score Breakdown"
    )


    score_cols = st.columns(4)


    with score_cols[0]:

        st.metric(
            "Skills",
            f"{candidate_score['skill_score']}/40"
        )


    with score_cols[1]:

        st.metric(
            "Projects",
            f"{candidate_score['project_score']}/20"
        )


    with score_cols[2]:

        st.metric(
            "Experience",
            f"{candidate_score['experience_score']}/15"
        )


    with score_cols[3]:

        st.metric(
            "Education",
            f"{candidate_score['education_score']}/10"
        )


    st.write(
        f"Certifications: "
        f"**{candidate_score['certification_score']}/5**"
    )

    st.write(
        f"Section Quality: "
        f"**{candidate_score['section_score']}/5**"
    )

    st.write(
        f"Resume Completeness: "
        f"**{candidate_score['completeness_score']}/5**"
    )


    # ========================================================
    # DETECTED SECTIONS
    # ========================================================

    st.header(
        "📑 Detected Resume Sections"
    )


    if result["sections"]:

        for section in result["sections"]:

            st.write(
                "• "
                +
                SECTION_DISPLAY_NAMES.get(
                    section,
                    section.title()
                )
            )

    else:

        st.warning(
            "No reliable sections detected."
        )


    # ========================================================
    # SKILLS
    # ========================================================

    st.header(
        "🛠️ Skills Analysis"
    )


    if result["skills"]:

        st.write(
            ", ".join(
                result["skills"]
            )
        )

    else:

        st.info(
            "No recognized skills detected."
        )


    # ========================================================
    # JOB MATCHING
    # ========================================================

    st.header(
        "💼 Job Skill Matching"
    )


    st.write(
        "**Required skills:**"
    )

    st.write(
        ", ".join(
            DEFAULT_REQUIRED_SKILLS
        )
    )


    matched = candidate_score[
        "matched_skills"
    ]

    missing = candidate_score[
        "missing_skills"
    ]


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "✅ Matched Skills"
        )

        for skill in matched:

            st.write(
                f"✓ {skill}"
            )


    with col2:

        st.subheader(
            "❌ Missing Skills"
        )

        for skill in missing:

            st.write(
                f"✗ {skill}"
            )


    # ========================================================
    # SECTION CONTENT
    # ========================================================

    st.header(
        "📂 Resume Content"
    )


    for section, content in result[
        "sections"
    ].items():

        display_name = (
            SECTION_DISPLAY_NAMES.get(
                section,
                section.title()
            )
        )


        with st.expander(
            display_name
        ):

            st.write(
                content
            )


    # ========================================================
    # RAW TEXT
    # ========================================================

    with st.expander(
        "View Extracted Resume Text"
    ):

        st.text(
            result["text"]
        )


# ============================================================
# HR DASHBOARD
# ============================================================

def hr_dashboard():

    st.title(
        "🧑‍💼 HR Dashboard"
    )

    st.write(
        "Upload multiple resumes and rank candidates "
        "against the selected job requirements."
    )


    # ========================================================
    # JOB REQUIREMENTS
    # ========================================================

    st.header(
        "💼 Job Requirements"
    )


    job_skill_text = st.text_input(

        "Required skills",

        value=", ".join(
            DEFAULT_REQUIRED_SKILLS
        ),

        help="Enter skills separated by commas."
    )


    required_skills = [

        skill.strip()

        for skill in job_skill_text.split(",")

        if skill.strip()
    ]


    required_skills = list(
        dict.fromkeys(
            required_skills
        )
    )


    st.write(
        "**Skills used for ranking:**"
    )

    st.write(
        ", ".join(
            required_skills
        )
    )


    # ========================================================
    # HR RESUME UPLOAD
    # ========================================================

    st.header(
        "📄 Candidate Resumes"
    )


    uploaded_files = st.file_uploader(

        "Upload Candidate Resumes",

        type=["pdf"],

        accept_multiple_files=True,

        key="hr_resumes"
    )


    if not uploaded_files:

        st.info(
            "Upload one or more candidate PDFs "
            "to generate the ranking."
        )

        return


    # ========================================================
    # PROCESS CANDIDATES
    # ========================================================

    candidates = []

    invalid_files = []


    progress = st.progress(0)

    total_files = len(
        uploaded_files
    )


    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        with st.spinner(
            f"Analyzing {uploaded_file.name}..."
        ):

            result = process_resume(

                uploaded_file,

                required_skills
            )


        if result.get("valid"):

            candidates.append(
                result
            )

        else:

            invalid_files.append(
                uploaded_file.name
            )


        progress.progress(
            (index + 1)
            /
            total_files
        )


    progress.empty()


    # ========================================================
    # INVALID FILES
    # ========================================================

    if invalid_files:

        st.warning(
            "The following files were not confidently "
            "identified as resumes:"
        )

        for filename in invalid_files:

            st.write(
                f"• {filename}"
            )


    # ========================================================
    # NO VALID CANDIDATES
    # ========================================================

    if not candidates:

        st.error(
            "No valid resumes could be analyzed."
        )

        return


    # ========================================================
    # RANK CANDIDATES
    # ========================================================

    candidates.sort(

        key=lambda candidate:
            candidate[
                "candidate_score"
            ][
                "total_score"
            ],

        reverse=True
    )


    # ========================================================
    # ASSIGN RANK
    # ========================================================

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        candidate["rank"] = index


    # ========================================================
    # RECRUITMENT SUMMARY
    # ========================================================

    st.header(
        "📊 Recruitment Summary"
    )


    highest_score = candidates[0][
        "candidate_score"
    ][
        "total_score"
    ]


    top_candidate = candidates[0][
        "name"
    ]


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Candidates Analyzed",
            len(candidates)
        )


    with col2:

        st.metric(
            "Highest Score",
            f"{highest_score}/100"
        )


    with col3:

        st.metric(
            "Top Candidate",
            top_candidate
        )


    # ========================================================
    # CANDIDATE RANKING
    # ========================================================

    st.header(
        "🏆 Candidate Ranking"
    )


    for candidate in candidates:

        score = candidate[
            "candidate_score"
        ]

        rank = candidate[
            "rank"
        ]


        if rank == 1:

            medal = "🥇"

        elif rank == 2:

            medal = "🥈"

        elif rank == 3:

            medal = "🥉"

        else:

            medal = "👤"


        st.subheader(
            f"{medal} #{rank} "
            f"{candidate['name']}"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Candidate Score",
                f"{score['total_score']}/100"
            )


        with col2:

            st.metric(
                "Skill Match",
                f"{score['skill_match_percentage']:.1f}%"
            )


        with col3:

            st.metric(
                "Resume Score",
                candidate["resume_score"]
            )


        with col4:

            st.metric(
                "Sections",
                len(candidate["sections"])
            )


        st.divider()


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    st.header(
        "📊 Candidate Comparison"
    )


    table_header = (

        "| Rank | Candidate | Score | "
        "Skill Match | Projects | Experience | Education |\n"

        "|---:|---|---:|---:|---:|---:|---:|\n"
    )


    table_rows = ""


    for candidate in candidates:

        score = candidate[
            "candidate_score"
        ]


        table_rows += (

            f"| {candidate['rank']} "

            f"| {candidate['name']} "

            f"| {score['total_score']}/100 "

            f"| {score['skill_match_percentage']:.1f}% "

            f"| {score['project_score']}/20 "

            f"| {score['experience_score']}/15 "

            f"| {score['education_score']}/10 |\n"
        )


    st.markdown(
        table_header + table_rows
    )


    # ========================================================
    # SHORTLIST
    # ========================================================

    st.header(
        "⭐ Shortlist"
    )


    # IMPORTANT:
    # Streamlit slider cannot have min == max.
    # Therefore, handle one candidate separately.

    if len(candidates) == 1:

        shortlist_count = 1

        st.info(
            "Only one candidate was analyzed, "
            "so the candidate is automatically shortlisted."
        )

    else:

        shortlist_count = st.slider(

            "Number of candidates to shortlist",

            min_value=1,

            max_value=len(candidates),

            value=min(
                3,
                len(candidates)
            )
        )


    shortlisted = candidates[
        :shortlist_count
    ]


    for candidate in shortlisted:

        score = candidate[
            "candidate_score"
        ]


        st.success(

            f"#{candidate['rank']} "
            f"{candidate['name']} — "
            f"{score['total_score']}/100"
        )


    # ========================================================
    # DETAILED CANDIDATE REVIEW
    # ========================================================

    st.header(
        "🔎 Detailed Candidate Review"
    )


    candidate_names = [

        candidate["name"]

        for candidate in candidates
    ]


    selected_name = st.selectbox(

        "Select candidate",

        candidate_names
    )


    selected_candidate = next(

        candidate

        for candidate in candidates

        if candidate["name"] == selected_name
    )


    selected_score = selected_candidate[
        "candidate_score"
    ]


    st.subheader(

        f"{selected_candidate['name']} "
        f"— Rank #{selected_candidate['rank']}"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "Candidate Score",

            f"{selected_score['total_score']}/100"
        )


    with col2:

        st.metric(

            "Skill Match",

            f"{selected_score['skill_match_percentage']:.1f}%"
        )


    with col3:

        st.metric(

            "Resume Score",

            selected_candidate["resume_score"]
        )


    # ========================================================
    # SCORE BREAKDOWN
    # ========================================================

    st.subheader(
        "Score Breakdown"
    )


    st.write(
        f"Skills: "
        f"{selected_score['skill_score']}/40"
    )


    st.write(
        f"Projects: "
        f"{selected_score['project_score']}/20"
    )


    st.write(
        f"Experience: "
        f"{selected_score['experience_score']}/15"
    )


    st.write(
        f"Education: "
        f"{selected_score['education_score']}/10"
    )


    st.write(
        f"Certifications: "
        f"{selected_score['certification_score']}/5"
    )


    st.write(
        f"Section Quality: "
        f"{selected_score['section_score']}/5"
    )


    st.write(
        f"Completeness: "
        f"{selected_score['completeness_score']}/5"
    )


    # ========================================================
    # MATCHED / MISSING
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "✅ Matched Skills"
        )


        if selected_score[
            "matched_skills"
        ]:

            for skill in selected_score[
                "matched_skills"
            ]:

                st.write(
                    f"✓ {skill}"
                )

        else:

            st.info(
                "No required skills matched."
            )


    with col2:

        st.subheader(
            "❌ Missing Skills"
        )


        if selected_score[
            "missing_skills"
        ]:

            for skill in selected_score[
                "missing_skills"
            ]:

                st.write(
                    f"✗ {skill}"
                )

        else:

            st.success(
                "All required skills matched."
            )


    # ========================================================
    # DETECTED SECTIONS
    # ========================================================

    st.subheader(
        "📑 Detected Sections"
    )


    for section in selected_candidate[
        "sections"
    ]:

        st.write(

            "• "

            +

            SECTION_DISPLAY_NAMES.get(

                section,

                section.title()
            )
        )


    # ========================================================
    # SECTION DETAILS
    # ========================================================

    st.subheader(
        "📂 Section Details"
    )


    for section, content in selected_candidate[
        "sections"
    ].items():

        display_name = (

            SECTION_DISPLAY_NAMES.get(

                section,

                section.title()
            )
        )


        with st.expander(
            display_name
        ):

            st.write(
                content
            )


    # ========================================================
    # RAW TEXT
    # ========================================================

    with st.expander(
        "View Extracted Resume Text"
    ):

        st.text(
            selected_candidate["text"]
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📄 AI Resume Screener"
)


dashboard = st.sidebar.radio(

    "Choose Dashboard",

    [

        "👤 User Dashboard",

        "🧑‍💼 HR Dashboard"
    ]
)


# ============================================================
# RUN DASHBOARD
# ============================================================

if dashboard == "👤 User Dashboard":

    user_dashboard()

else:

    hr_dashboard()