from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from section_dataset import section_data


# ==================================================
# 1. LOAD DATASET
# ==================================================

texts = [text for text, label in section_data]
labels = [label for text, label in section_data]


# ==================================================
# 2. SPLIT DATASET
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


print("===== DATASET SPLIT =====")
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))


# ==================================================
# 3. LOAD SENTENCE TRANSFORMER
# ==================================================

print("\nLoading SentenceTransformer...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 4. CREATE EMBEDDINGS
# ==================================================

print("Creating embeddings...")

X_train_embeddings = embedding_model.encode(
    X_train,
    show_progress_bar=True
)

X_test_embeddings = embedding_model.encode(
    X_test,
    show_progress_bar=True
)


# ==================================================
# 5. TRAIN LOGISTIC REGRESSION
# ==================================================

print("\nTraining Logistic Regression...")

classifier = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        random_state=42
    )
)

classifier.fit(
    X_train_embeddings,
    y_train
)


# ==================================================
# 6. PREDICTION
# ==================================================

y_pred = classifier.predict(
    X_test_embeddings
)


# ==================================================
# 7. ACCURACY
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===== MODEL ACCURACY =====")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==================================================
# 8. CLASSIFICATION REPORT
# ==================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==================================================
# 9. CONFUSION MATRIX
# ==================================================

print("\n===== CONFUSION MATRIX =====")

labels_order = classifier.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels_order
)

print("\nLabels:")

print(
    list(labels_order)
)

print("\nMatrix:")

print(cm)


# ==================================================
# 10. TEST REALISTIC RESUME HEADINGS
# ==================================================

print("\n===== REALISTIC HEADING TEST =====")

test_headings = [
    "Professional Summary",
    "Technical Skills",
    "Academic Projects",
    "Work Experience",
    "Educational Qualifications",
    "Certifications",
    "Achievements",
    "Languages Known",
    "Career Objective",
    "Personal Interests",
    "Machine Learning Projects",
    "Professional Experience",
    "Programming Languages",
    "Academic Background"
]


predictions = classifier.predict(
    embedding_model.encode(test_headings)
)

probabilities = classifier.predict_proba(
    embedding_model.encode(test_headings)
)


for heading, prediction, probability in zip(
    test_headings,
    predictions,
    probabilities
):

    confidence = max(probability) * 100

    print(
        f"{heading} -> "
        f"{prediction} "
        f"({confidence:.1f}%)"
    )