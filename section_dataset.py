# ============================================================
# RESUME SECTION TRAINING DATASET
# ============================================================
# Each tuple contains:
# (resume text / heading, section label)

section_data = [

    # ========================================================
    # SUMMARY
    # ========================================================

    ("Summary", "Summary"),
    ("Professional Summary", "Summary"),
    ("Career Summary", "Summary"),
    ("Professional Profile", "Summary"),
    ("Career Profile", "Summary"),
    ("Profile", "Summary"),
    ("About Me", "Summary"),
    ("About", "Summary"),
    ("Personal Profile", "Summary"),
    ("Executive Summary", "Summary"),
    ("Professional Overview", "Summary"),
    ("Career Overview", "Summary"),
    ("Profile Summary", "Summary"),
    ("Professional Introduction", "Summary"),
    ("Overview", "Summary"),

    ("Motivated engineering graduate with strong programming and machine learning skills", "Summary"),
    ("Aspiring AI engineer with knowledge of Python and machine learning", "Summary"),
    ("Electronics and communication engineering graduate interested in artificial intelligence", "Summary"),
    ("Computer science graduate with experience in Python and data analysis", "Summary"),
    ("Detail-oriented graduate with strong analytical and problem-solving abilities", "Summary"),
    ("Enthusiastic technology graduate seeking opportunities in artificial intelligence", "Summary"),
    ("Engineering graduate with hands-on experience in software and IoT projects", "Summary"),
    ("Passionate learner interested in machine learning, deep learning and generative AI", "Summary"),
    ("Entry-level engineer with practical experience developing software applications", "Summary"),
    ("Results-oriented graduate with strong technical and analytical skills", "Summary"),

    # ========================================================
    # OBJECTIVE
    # ========================================================

    ("Objective", "Objective"),
    ("Career Objective", "Objective"),
    ("Professional Objective", "Objective"),
    ("Career Goal", "Objective"),
    ("Job Objective", "Objective"),
    ("Objective Statement", "Objective"),
    ("Career Aim", "Objective"),
    ("Seeking an opportunity to start my career in the IT industry", "Objective"),
    ("Motivated graduate looking for a challenging career opportunity", "Objective"),
    ("Looking for an opportunity to apply my technical skills", "Objective"),
    ("Aspiring AI engineer seeking an entry-level position", "Objective"),
    ("Looking to begin a career in software and artificial intelligence", "Objective"),
    ("Seeking a position where I can apply my programming knowledge", "Objective"),
    ("Goal is to grow as a machine learning engineer", "Objective"),
    ("Interested in joining an organization to develop technical expertise", "Objective"),
    ("Looking for a challenging role to start my professional journey", "Objective"),
    ("Seeking a position where I can contribute to software development", "Objective"),
    ("To obtain an entry-level position in artificial intelligence", "Objective"),

    # ========================================================
    # EDUCATION
    # ========================================================

    ("Education", "Education"),
    ("Educational Background", "Education"),
    ("Academic Background", "Education"),
    ("Academic Qualifications", "Education"),
    ("Educational Qualifications", "Education"),
    ("Academic Profile", "Education"),
    ("Academic Credentials", "Education"),
    ("Education Qualifications", "Education"),
    ("Academic History", "Education"),
    ("Educational History", "Education"),
    ("Qualifications", "Education"),
    ("Academic Record", "Education"),
    ("Bachelor of Engineering in Electronics and Communication Engineering", "Education"),
    ("B.E. in Computer Science Engineering", "Education"),
    ("Bachelor of Technology in Information Technology", "Education"),
    ("Master of Business Administration", "Education"),
    ("Completed PUC with 82%", "Education"),
    ("Academic qualifications include a degree in Computer Science", "Education"),
    ("Graduated with a Bachelor of Technology degree", "Education"),
    ("Completed undergraduate studies in Electronics Engineering", "Education"),
    ("University education in Information Science and Engineering", "Education"),
    ("Higher secondary education with distinction", "Education"),
    ("Bachelor of Engineering with a CGPA of 8.02", "Education"),
    ("Completed SSLC with 85 percent", "Education"),
    ("Completed PUC Science with 78 percent", "Education"),
    ("Bachelor's degree with a CGPA of 7.5", "Education"),
    ("B.Tech Computer Science Engineering VTU", "Education"),
    ("B.E. Electronics and Communication Engineering", "Education"),
    ("Bachelor of Engineering 2022-2026", "Education"),
    ("Graduated from Visvesvaraya Technological University", "Education"),

    # ========================================================
    # SKILLS
    # ========================================================

    ("Skills", "Skills"),
    ("Technical Skills", "Skills"),
    ("Technical Expertise", "Skills"),
    ("Technical Knowledge", "Skills"),
    ("Core Competencies", "Skills"),
    ("Core Skills", "Skills"),
    ("Key Skills", "Skills"),
    ("Programming Skills", "Skills"),
    ("Technology Skills", "Skills"),
    ("Technical Competencies", "Skills"),
    ("Tools and Technologies", "Skills"),
    ("Technology Stack", "Skills"),
    ("Areas of Expertise", "Skills"),
    ("Professional Skills", "Skills"),
    ("Software Skills", "Skills"),
    ("IT Skills", "Skills"),
    ("Skills and Technologies", "Skills"),
    ("Technical Proficiency", "Skills"),

    ("Python, SQL, TensorFlow, Pandas, NumPy", "Skills"),
    ("Technical Skills: Python, Java, C++, SQL", "Skills"),
    ("Machine Learning, Deep Learning, Git and GitHub", "Skills"),
    ("Programming Languages: Python and Java", "Skills"),
    ("Technology Stack: Python, SQL, TensorFlow", "Skills"),
    ("Proficient in Python, Java, SQL and data analysis", "Skills"),
    ("Expertise in machine learning and data visualization", "Skills"),
    ("Tools and technologies include Git, Docker and TensorFlow", "Skills"),
    ("Core competencies include programming and problem solving", "Skills"),
    ("Knowledge of Python, APIs, databases and cloud computing", "Skills"),
    ("Python, TensorFlow, Keras, Scikit-learn", "Skills"),
    ("Machine Learning, NLP, Computer Vision", "Skills"),
    ("SQL, MySQL, Pandas and NumPy", "Skills"),
    ("Git, GitHub, Docker and AWS", "Skills"),

    # ========================================================
    # PROJECTS
    # ========================================================

    ("Projects", "Projects"),
    ("Project", "Projects"),
    ("Academic Projects", "Projects"),
    ("Personal Projects", "Projects"),
    ("Key Projects", "Projects"),
    ("Selected Projects", "Projects"),
    ("Project Experience", "Projects"),
    ("Project Portfolio", "Projects"),
    ("Technical Projects", "Projects"),
    ("Major Projects", "Projects"),
    ("Academic Project Work", "Projects"),
    ("Selected Academic Projects", "Projects"),
    ("Projects and Applications", "Projects"),
    ("Project Highlights", "Projects"),

    ("Developed an IoT-based patient monitoring system", "Projects"),
    ("Built a facial recognition attendance system", "Projects"),
    ("Personal Projects: Student tracking system using LoRa", "Projects"),
    ("Developed a machine learning model for prediction", "Projects"),
    ("Created a web application using Python and Flask", "Projects"),
    ("Created an AI-based resume screening application", "Projects"),
    ("Designed and implemented an automatic attendance system", "Projects"),
    ("Built a web-based application for data analysis", "Projects"),
    ("Implemented a deep learning image classification system", "Projects"),
    ("Developed a smart monitoring solution using IoT", "Projects"),
    ("Built a personal expense tracker using Python and Flask", "Projects"),
    ("Developed an IoT-based home automation system as an academic project", "Projects"),
    ("Created a machine learning model to predict student performance", "Projects"),
    ("Built a facial recognition attendance system using Python", "Projects"),
    ("Developed a personal portfolio website using HTML, CSS and JavaScript", "Projects"),
    ("Developed an AI resume screening system using NLP", "Projects"),
    ("Built an EV charging station finder application", "Projects"),
    ("Created a battery range estimation system for electric vehicles", "Projects"),
    ("Developed a chatbot using Python and NLP", "Projects"),
    ("Implemented a computer vision based attendance system", "Projects"),

    # ========================================================
    # WORK EXPERIENCE
    # ========================================================

    ("Experience", "Work Experience"),
    ("Work Experience", "Work Experience"),
    ("Professional Experience", "Work Experience"),
    ("Employment History", "Work Experience"),
    ("Work History", "Work Experience"),
    ("Career Experience", "Work Experience"),
    ("Career History", "Work Experience"),
    ("Professional Background", "Work Experience"),
    ("Industry Experience", "Work Experience"),
    ("Internship Experience", "Work Experience"),
    ("Internships", "Work Experience"),
    ("Relevant Experience", "Work Experience"),
    ("Work Profile", "Work Experience"),
    ("Employment Experience", "Work Experience"),
    ("Career Background", "Work Experience"),

    ("Worked as a Software Engineer Intern", "Work Experience"),
    ("Professional Experience: Software Developer Intern", "Work Experience"),
    ("Worked as a Data Science Intern", "Work Experience"),
    ("Internship at an embedded systems company", "Work Experience"),
    ("Developed applications during my internship", "Work Experience"),
    ("Served as a Python Developer Intern", "Work Experience"),
    ("Gained industry experience through a software development internship", "Work Experience"),
    ("Worked with a team to build and test web applications", "Work Experience"),
    ("Responsible for developing and maintaining software solutions", "Work Experience"),
    ("Completed an internship focused on artificial intelligence", "Work Experience"),
    ("Worked as a software engineering intern and developed web applications", "Work Experience"),
    ("Assisted the development team in maintaining production software", "Work Experience"),
    ("Worked at a company as a Python developer intern", "Work Experience"),
    ("Collaborated with senior developers during my internship", "Work Experience"),
    ("Developed and maintained applications as part of a software engineering role", "Work Experience"),
    ("Software Engineer at TCS", "Work Experience"),
    ("Python Developer Intern at an IT company", "Work Experience"),
    ("Worked on Python development and machine learning projects", "Work Experience"),
    ("Developed software applications as part of an internship", "Work Experience"),

    # ========================================================
    # CERTIFICATIONS
    # ========================================================

    ("Certifications", "Certifications"),
    ("Certification", "Certifications"),
    ("Certificates", "Certifications"),
    ("Professional Certifications", "Certifications"),
    ("Certifications and Courses", "Certifications"),
    ("Courses and Certifications", "Certifications"),
    ("Training and Certifications", "Certifications"),
    ("Professional Credentials", "Certifications"),
    ("Credentials", "Certifications"),
    ("Certifications and Training", "Certifications"),
    ("Training Credentials", "Certifications"),
    ("Courses", "Certifications"),
    ("Online Certifications", "Certifications"),
    ("Technical Certifications", "Certifications"),

    ("Completed Python certification from Infosys", "Certifications"),
    ("AWS Cloud Fundamentals certification", "Certifications"),
    ("Certified in Machine Learning", "Certifications"),
    ("Completed TensorFlow certification course", "Certifications"),
    ("Professional certifications and courses", "Certifications"),
    ("Earned a certificate in Data Analytics", "Certifications"),
    ("Successfully completed an Artificial Intelligence course", "Certifications"),
    ("Received certification in cloud computing", "Certifications"),
    ("Completed an online course in Python programming", "Certifications"),
    ("Training credentials include machine learning and AWS", "Certifications"),
    ("Earned a certificate after completing an AI course", "Certifications"),
    ("Received certification in Artificial Intelligence", "Certifications"),
    ("Completed a certified AI training program", "Certifications"),
    ("Professional certification in Machine Learning", "Certifications"),
    ("Certificate of completion for an AI course", "Certifications"),
    ("IBM Dev Con Certificate, 2025", "Certifications"),
    ("Python for Data Science certification", "Certifications"),
    ("Machine Learning certification from Coursera", "Certifications"),

    # ========================================================
    # ACHIEVEMENTS
    # ========================================================

    ("Achievements", "Achievements"),
    ("Awards", "Achievements"),
    ("Honors", "Achievements"),
    ("Honours", "Achievements"),
    ("Accomplishments", "Achievements"),
    ("Awards and Achievements", "Achievements"),
    ("Academic Achievements", "Achievements"),
    ("Professional Achievements", "Achievements"),
    ("Recognition", "Achievements"),
    ("Achievements and Awards", "Achievements"),
    ("College Achievements", "Achievements"),

    ("Won first place in a college technical competition", "Achievements"),
    ("Received first prize in a coding competition", "Achievements"),
    ("Awarded best project in the department", "Achievements"),
    ("Secured second position in a hackathon", "Achievements"),
    ("Achievements and awards received during college", "Achievements"),
    ("Recognized for outstanding performance in a technical event", "Achievements"),
    ("Received an award for academic excellence", "Achievements"),
    ("Selected as a finalist in a national hackathon", "Achievements"),
    ("Earned recognition for completing a successful project", "Achievements"),
    ("Honoured for exceptional performance during college", "Achievements"),
    ("Won a technical quiz competition", "Achievements"),
    ("Received recognition for academic performance", "Achievements"),

    # ========================================================
    # INTERESTS
    # ========================================================

    ("Interests", "Interests"),
    ("Interest", "Interests"),
    ("Hobbies", "Interests"),
    ("Hobbies and Interests", "Interests"),
    ("Personal Interests", "Interests"),
    ("Areas of Interest", "Interests"),
    ("Leisure Activities", "Interests"),
    ("Personal Interests and Hobbies", "Interests"),
    ("Activities and Interests", "Interests"),
    ("Outside Work", "Interests"),

    ("Reading and dancing", "Interests"),
    ("Hobbies include reading books and travelling", "Interests"),
    ("Interested in photography and music", "Interests"),
    ("Enjoy playing sports and listening to music", "Interests"),
    ("Personal interests and hobbies", "Interests"),
    ("Enjoy reading, drawing and exploring new technologies", "Interests"),
    ("My hobbies include music, travelling and photography", "Interests"),
    ("Interested in creative activities and technology", "Interests"),
    ("Outside academics, I enjoy dancing and reading", "Interests"),
    ("Leisure activities include sports and music", "Interests"),
    ("Enjoy photography, travelling and reading", "Interests"),
    ("Interested in sports and creative activities", "Interests"),

    # ========================================================
    # LANGUAGES
    # ========================================================

    ("Languages", "Languages"),
    ("Language", "Languages"),
    ("Languages Known", "Languages"),
    ("Languages Known", "Languages"),
    ("Language Proficiency", "Languages"),
    ("Language Skills", "Languages"),
    ("Languages Spoken", "Languages"),
    ("Known Languages", "Languages"),
    ("Communication Languages", "Languages"),
    ("Language Competencies", "Languages"),

    ("English, Kannada and Hindi", "Languages"),
    ("Languages known: English and Kannada", "Languages"),
    ("Fluent in English and Kannada", "Languages"),
    ("English, Hindi, Kannada", "Languages"),
    ("Language proficiency: English and Hindi", "Languages"),
    ("Can communicate in Kannada, English and Hindi", "Languages"),
    ("Known languages include Kannada and English", "Languages"),
    ("Multilingual with proficiency in English and Kannada", "Languages"),
    ("Able to speak and write in English and Kannada", "Languages"),
    ("Language skills include Hindi, English and Kannada", "Languages"),
    ("English and Kannada", "Languages"),
    ("English, Hindi and Kannada languages", "Languages")
]


# ============================================================
# DATASET INFORMATION
# ============================================================

print("===== RESUME SECTION DATASET =====\n")

print(f"Total examples: {len(section_data)}")

print("\nExamples per section:")

section_counts = {}

for text, label in section_data:
    section_counts[label] = section_counts.get(label, 0) + 1

for label, count in section_counts.items():
    print(f"{label}: {count}")