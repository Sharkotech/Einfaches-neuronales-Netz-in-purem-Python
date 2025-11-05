import math
import random
import matplotlib.pyplot as plt

def sigmoid(z):
    if z < -60:
        return 0.0
    elif z > 60:
        return 1.0
    else:
        return 1 / (1 + math.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# Trainingsdaten
train_data = []
for i in range(1):
    for x in range(-10, 21):
        y = 1 if 1 <= x <= 10 else 0
        train_data.append((x, y))

w1 = random.uniform(-0.01, 0.01)
b1 = random.uniform(-0.01, 0.01)
w2 = random.uniform(-0.01, 0.01)
b2 = random.uniform(-0.01, 0.01)
print(f"Initiale Gewichte: w1={w1}, b1={b1}, w2={w2}, b2={b2}")
eta = 0.000005  # Lernrate

for epoch in range(500_000):
    total_loss = 0

    for x, y_true in train_data:
        # Vorwärts
        z1 = w1 * x + b1
        z2 = w2 * x + b2
        n1 = sigmoid(z1)
        n2 = sigmoid(-z2 + 10)
        y_pred = n1 * n2

        # Fehler
        loss = (y_true - y_pred) ** 2
        total_loss += loss

        # Backprop
        error = (y_pred - y_true)

        # Lokale Gradienten
        d_n1 = error * n2
        d_n2 = error * n1

        d_z1 = d_n1 * sigmoid_deriv(z1)
        d_z2 = d_n2 * sigmoid_deriv(z2)

        # Gewichtsanpassung
        w1 -= eta * d_z1 * x
        b1 -= eta * d_z1
        w2 -= eta * d_z2 * x
        b2 -= eta * d_z2

    if epoch % 500 == 0:
        print(f"Epoche {epoch} Fehler: {round(total_loss, 3)} Gewichte: w1={round(w1, 4)}, b1={round(b1, 4)}, w2={round(w2, 4)}, b2={round(b2, 4)}")

y_predplt = []
print("\n=== Test ===")

x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for test_x in x:
    z1 = w1 * test_x + b1
    z2 = w2 * test_x + b2
    n1 = sigmoid(z1)
    n2 = sigmoid(-z2 + 10)
    y_pred = n1 * n2
    print(f"x={test_x:>3} → KI-Ausgabe={y_pred:.2f}")
    y_predplt.append(y_pred)


y_true = []
for val in x:
    if 1 <= val <= 10:
        y_true.append(1.0)
    else:
        y_true.append(0.0)

# Plotten
plt.figure(figsize=(9,5))
plt.plot(x, y_predplt, 'o-', color='blue', label='KI-Ausgabe (Vorhersage)')
plt.step(x, y_true, where='mid', color='green', linestyle='--', label='Soll-Kurve (1 im Bereich 1–10)')
plt.axvspan(1, 10, color='green', alpha=0.1, label='Zielbereich')
plt.axhline(y=0.5, color='red', linestyle=':', linewidth=1, label='Grenze 0.5')
plt.title("KI-Ausgabe vs. Soll-Kurve")
plt.xlabel("Eingabewert (x)")
plt.ylabel("Ausgabe / Wahrscheinlichkeit")
plt.grid(True)
plt.legend()
plt.show()