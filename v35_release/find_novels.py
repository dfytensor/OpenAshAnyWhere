import os, glob

novel_dir = r"F:\小说\女生小说"
files = []
for f in os.listdir(novel_dir):
    if f.endswith('.txt'):
        path = os.path.join(novel_dir, f)
        size = os.path.getsize(path)
        files.append((f, size))

files.sort(key=lambda x: x[1])

print("Smallest (< 500KB):")
for name, size in files[:10]:
    print(f"  {name}: {size/1024:.0f} KB")

print(f"\nMedium (100KB - 2MB):")
mid = [f for f in files if 100*1024 < f[1] < 2*1024*1024]
for name, size in mid[:10]:
    print(f"  {name}: {size/1024:.0f} KB")

print(f"\nLarge (> 10MB):")
big = [f for f in files if f[1] > 10*1024*1024]
for name, size in big[:10]:
    print(f"  {name}: {size/1024/1024:.1f} MB")

# Find specific novels
targets = ["告白", "难哄", "黑月光", "水月洞天", "盗墓"]
print(f"\nTargeted search:")
for name, size in files:
    for t in targets:
        if t in name:
            print(f"  '{t}' -> {name}: {size/1024:.0f} KB")
            break
