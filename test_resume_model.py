from transformers import pipeline

print("Loading resume section model...")

classifier = pipeline(
    "text-classification",
    model="amosify/resume-section-classifier-v1"
)

test_lines = [
    "Bachelor of Engineering in Electronics and Communication Engineering",
    "Python, TensorFlow, Keras, NLP",
    "Developed an AI Resume Screening System",
    "Worked as Software Engineer Intern at TCS",
    "Machine Learning Projects",
    "Professional Experience",
    "Certifications and Courses",
    "Languages Known",
    "Data Analysis and Visualization",
    "Built an IoT based patient monitoring system",
    "Received certification in Artificial Intelligence"
]

print("\n===== PRETRAINED RESUME MODEL TEST =====\n")

for line in test_lines:

    result = classifier(line)[0]

    print(
        f"{line}"
        f" -> {result['label']}"
        f" ({result['score'] * 100:.2f}%)"
    )