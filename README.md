# ======================================================
# THUẬT TOÁN TÔ MÀU ĐỒ THỊ (GRAPH COLORING - GREEDY)
# + THUẬT TOÁN AKT (KRUSKAL - CÂY KHUNG NHỎ NHẤT)
# Ngôn ngữ: Python
# Chạy trên Google Colab / Python thường
# ======================================================


# ======================================================
# PHẦN 1: THUẬT TOÁN TÔ MÀU ĐỒ THỊ
# ======================================================

import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------
# NHẬP DỮ LIỆU ĐỒ THỊ
# ------------------------------------------------------
n = int(input("Nhập số đỉnh (Tô màu): "))
m = int(input("Nhập số cạnh (Tô màu): "))

G = nx.Graph()
G.add_nodes_from(range(n))

print("Nhập các cạnh (u v):")
for i in range(m):
    u, v = map(int, input(f"Cạnh {i+1}: ").split())
    G.add_edge(u, v)

# ------------------------------------------------------
# THUẬT TOÁN TÔ MÀU GREEDY
# Ý tưởng:
# - Duyệt từng đỉnh
# - Gán màu nhỏ nhất chưa bị trùng với các đỉnh kề
# ------------------------------------------------------
colors = {}

for node in G.nodes():
    # Lấy các màu đã dùng ở các đỉnh kề
    used_colors = set(colors.get(neigh) for neigh in G.neighbors(node))

    # Tìm màu nhỏ nhất chưa dùng
    color = 0
    while color in used_colors:
        color += 1

    colors[node] = color

# ------------------------------------------------------
# HIỂN THỊ KẾT QUẢ TÔ MÀU
# ------------------------------------------------------
print("\nKết quả tô màu đồ thị:")
for node in colors:
    print(f"Đỉnh {node} → Màu {colors[node]}")

# ------------------------------------------------------
# VẼ ĐỒ THỊ SAU KHI TÔ MÀU
# ------------------------------------------------------
color_list = [colors[node] for node in G.nodes()]
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(8, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=color_list,
    cmap=plt.cm.Set3,
    node_size=1000,
    font_size=12
)
plt.title("Đồ thị sau khi tô màu (Greedy Coloring)")
plt.show()


# ======================================================
# PHẦN 2: THUẬT TOÁN AKT (KRUSKAL - MINIMUM SPANNING TREE)
# ======================================================

# ------------------------------------------------------
# CẤU TRÚC DISJOINT SET (UNION-FIND)
# ------------------------------------------------------
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        # Tìm đại diện của tập
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Hợp hai tập nếu khác nhau
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px
            return True
        return False


# ------------------------------------------------------
# THUẬT TOÁN KRUSKAL
# Ý tưởng:
# - Sắp xếp các cạnh theo trọng số tăng dần
# - Lần lượt chọn cạnh nhỏ nhất không tạo chu trình
# ------------------------------------------------------
def kruskal(n, edges):
    edges.sort(key=lambda x: x[2])  # sắp theo trọng số
    ds = DisjointSet(n)
    mst = []

    for u, v, w in edges:
        if ds.union(u, v):
            mst.append((u, v, w))

    return mst


# ------------------------------------------------------
# NHẬP DỮ LIỆU ĐỒ THỊ CÓ TRỌNG SỐ
# ------------------------------------------------------
n = int(input("\nNhập số đỉnh (Kruskal): "))
m = int(input("Nhập số cạnh (Kruskal): "))

edges = []
for i in range(m):
    u, v, w = map(int, input(f"Cạnh {i+1} (u v w): ").split())
    edges.append((u, v, w))

# ------------------------------------------------------
# KẾT QUẢ CÂY KHUNG NHỎ NHẤT
# ------------------------------------------------------
mst = kruskal(n, edges)

print("\nCây khung nhỏ nhất (MST):")
total = 0
for u, v, w in mst:
    print(f"{u} - {v} : {w}")
    total += w

print("Tổng trọng số:", total)


# ======================================================
# TÓM TẮT
# ------------------------------------------------------
# 1. Tô màu đồ thị:
#    - Dùng thuật toán Greedy
#    - Hai đỉnh kề nhau không trùng màu
#
# 2. Kruskal (AKT):
#    - Tìm cây khung nhỏ nhất
#    - Dùng Union-Find để tránh chu trình
#
# ======================================================
# THUẬT TOÁN KNN (K-NEAREST NEIGHBORS)
# Ngôn ngữ: Python
# Chạy được trên Google Colab / Python thường
# Nhập dữ liệu động từ bàn phím
# ======================================================

import numpy as np
from collections import Counter

# ------------------------------------------------------
# HÀM TÍNH KHOẢNG CÁCH EUCLID
# Công thức:
# d(a, b) = sqrt((a1-b1)^2 + (a2-b2)^2 + ... + (an-bn)^2)
# ------------------------------------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# ------------------------------------------------------
# HÀM DỰ ĐOÁN KNN
# X_train : tập dữ liệu huấn luyện
# y_train : nhãn tương ứng
# x_test  : điểm cần phân loại
# k       : số láng giềng gần nhất
# ------------------------------------------------------
def knn_predict(X_train, y_train, x_test, k):
    distances = []

    # Tính khoảng cách từ x_test tới từng điểm huấn luyện
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    # Sắp xếp theo khoảng cách tăng dần
    distances.sort(key=lambda x: x[0])

    # Lấy nhãn của k điểm gần nhất
    labels = [label for _, label in distances[:k]]

    # Bỏ phiếu: nhãn xuất hiện nhiều nhất
    return Counter(labels).most_common(1)[0][0]


# ======================================================
# PHẦN NHẬP DỮ LIỆU ĐỘNG
# ======================================================

# Nhập số điểm huấn luyện
n = int(input("Nhập số điểm huấn luyện: "))

# Nhập số chiều của mỗi điểm
d = int(input("Nhập số chiều: "))

X_train = []
y_train = []

# Nhập từng điểm huấn luyện và nhãn
for i in range(n):
    x = list(map(float, input(f"Nhập điểm {i+1} ({d} số): ").split()))
    label = input(f"Nhãn của điểm {i+1}: ")
    X_train.append(x)
    y_train.append(label)

# Chuyển sang numpy array để tính toán
X_train = np.array(X_train)

# Nhập điểm cần phân loại
x_test = np.array(list(map(float, input("Nhập điểm cần phân loại: ").split())))

# Nhập k
k = int(input("Nhập k: "))


# ======================================================
# KẾT QUẢ
# ======================================================
result = knn_predict(X_train, y_train, x_test, k)
print("Kết quả phân loại theo KNN:", result)


# ======================================================
# GIẢI THÍCH NGẮN GỌN THUẬT TOÁN
# ------------------------------------------------------
# 1. KNN là thuật toán học có giám sát.
# 2. Không cần huấn luyện, chỉ lưu dữ liệu.
# 3. Khi có điểm mới:
#    - Tính khoảng cách đến tất cả điểm huấn luyện
#    - Chọn k điểm gần nhất
#    - Nhãn xuất hiện nhiều nhất là kết quả
# 4. Độ phức tạp: O(n * d)
# ======================================================

# 3. Chạy được trên Google Colab
# ======================================================
