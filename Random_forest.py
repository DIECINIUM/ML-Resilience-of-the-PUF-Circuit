import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- LFSR transformation ---
def lfsr(challenge, seed=0b1011, taps=[3, 2]):
    reg = seed
    out = []
    for _ in range(len(challenge)):
        bit = 0
        for t in taps:
            bit ^= (reg >> t) & 1
        out.append(challenge[_] ^ (reg & 1))
        reg = ((reg << 1) | bit) & 0xF  # Keep 4-bit LFSR
    return np.array(out)

# --- Arbiter PUF model ---
def arbiter_puf(challenge, weights):
    phi = np.zeros(len(challenge) + 1)
    phi[0] = 1
    for i in range(len(challenge)):
        phi[i+1] = phi[i] * (1 - 2 * challenge[i])
    delay = np.dot(weights, phi)
    return 1 if delay >= 0 else 0

# --- CRP generation ---
def generate_crps(num_samples, num_stages=4):
    weights = np.random.randn(num_stages + 1)  # Secret PUF weights
    crps = []
    for _ in range(num_samples):
        challenge = np.random.randint(0, 2, num_stages)
        transformed_challenge = lfsr(challenge)
        response = arbiter_puf(transformed_challenge, weights)
        crps.append((challenge, response))
    return zip(*crps)

# --- Feature transformation (phi vector) ---
def phi_transform(challenge):
    phi = np.zeros(len(challenge) + 1)
    phi[0] = 1
    for i in range(len(challenge)):
        phi[i+1] = phi[i] * (1 - 2 * challenge[i])
    return phi

# --- Generate and process CRPs ---
X_raw, y = generate_crps(5000)
X_raw = np.array(list(X_raw))
y = np.array(y)
X = np.array([phi_transform(lfsr(ch)) for ch in X_raw])

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- ML models ---
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

# --- Train and evaluate each model ---
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc * 100:.2f}%")
