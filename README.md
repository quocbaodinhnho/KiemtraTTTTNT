#Thuat toan to mau
Tạo đồ thị với n đỉnh và m cạnh.
Duyệt từng đỉnh:
Lấy tập các màu đã dùng bởi các đỉnh kề.
Gán cho đỉnh hiện tại màu nhỏ nhất không nằm trong tập đó.
Lưu màu của từng đỉnh vào colors.
Vẽ đồ thị, mỗi đỉnh có màu tương ứng
nx.Graph() – tạo đồ thị
G.add_nodes_from() – thêm đỉnh
G.add_edge(u, v) – thêm cạnh
G.nodes() – lấy danh sách đỉnh
G.neighbors(node) – lấy các đỉnh kề
nx.spring_layout() – xác định vị trí vẽ
nx.draw() – vẽ đồ thị
plt.show() – hiển thị đồ thị


#thuat toan AKT
Sắp xếp các cạnh theo trọng số tăng dần.
Duyệt từng cạnh theo thứ tự:
Nếu hai đỉnh của cạnh thuộc hai tập khác nhau → thêm cạnh đó vào cây khung.
Nếu cùng tập → bỏ qua (tránh chu trình).
Dùng cấu trúc Disjoint Set (Union–Find) để:
find: tìm tập của đỉnh
union: gộp hai tập lại với nhau
Kết thúc khi đã chọn đủ n-1 cạnh.
Lớp DisjointSet
__init__(n)
Khởi tạo mảng cha cho mỗi đỉnh
find(x)
Tìm đại diện của tập chứa x (nén đường đi)
union(x, y)
Gộp hai tập nếu khác nhau (tránh chu trình)
Hàm chính
kruskal(n, edges)
Thực hiện thuật toán Kruskal để tìm cây khung nhỏ nhất

#Thuat toan KNN
Nhập các điểm huấn luyện X_train và nhãn tương ứng y_train.
Khi có điểm cần phân loại x_test:
Tính khoảng cách Euclid từ x_test đến từng điểm huấn luyện.
Sắp xếp các điểm theo khoảng cách tăng dần.
Chọn k điểm gần nhất.
Đếm nhãn xuất hiện nhiều nhất → đó là kết quả phân loại.
euclidean_distance(a, b)
Tính khoảng cách Euclid giữa hai điểm
knn_predict(X_train, y_train, x_test, k)
Dự đoán nhãn của điểm mới bằng KNN
