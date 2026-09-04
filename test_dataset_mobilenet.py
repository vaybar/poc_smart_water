from pathlib import Path

dataset = Path("dataset_mobilenet/train")

print("Clase  | Cantidad | Barra")
print("-------+----------+------")
for clase in sorted(dataset.iterdir(), key=lambda p: p.name):
    n   = len(list(clase.glob("*.png")))
    bar = "█" * (n // 100)
    print(f"  {clase.name}    | {n:>6}   | {bar}")