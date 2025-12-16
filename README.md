#Thuat toan to mau
Tạo đồ thị với n đỉnh và m cạnh.
Duyệt từng đỉnh:
Lấy tập các màu đã dùng bởi các đỉnh kề.
Gán cho đỉnh hiện tại màu nhỏ nhất không nằm trong tập đó.
Lưu màu của từng đỉnh vào colors.
Vẽ đồ thị, mỗi đỉnh có màu tương ứng

#thuat toan AKT
Sắp xếp các cạnh theo trọng số tăng dần.
Duyệt từng cạnh theo thứ tự:
Nếu hai đỉnh của cạnh thuộc hai tập khác nhau → thêm cạnh đó vào cây khung.
Nếu cùng tập → bỏ qua (tránh chu trình).
Dùng cấu trúc Disjoint Set (Union–Find) để:
find: tìm tập của đỉnh
union: gộp hai tập lại với nhau
Kết thúc khi đã chọn đủ n-1 cạnh.

#Thuat toan KNN
Nhập các điểm huấn luyện X_train và nhãn tương ứng y_train.
Khi có điểm cần phân loại x_test:
Tính khoảng cách Euclid từ x_test đến từng điểm huấn luyện.
Sắp xếp các điểm theo khoảng cách tăng dần.
Chọn k điểm gần nhất.
Đếm nhãn xuất hiện nhiều nhất → đó là kết quả phân loại.
